# src/vm_tft/cli/explain.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

# ---- headless matplotlib (no GUI popups) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from darts.explainability import TFTExplainer

from vm_tft.cfg import (
    ALL_FILE, TEST_FILE, FREQ,
    TIME_COL, GROUP_COL, TARGET,
    STATIC_CATS, STATIC_REALS,
    PAST_REALS, FUTURE_REALS,
    ARTIFACTS, INPUT_CHUNK
)
from vm_tft.features import build_future_covariates_well_age_full
from vm_tft.features import build_future_covariates_dca_full
from vm_tft.io_utils import ensure_dir, artifact_path, set_seeds


# =========================
# Helpers
# =========================

def _as_list(x):
    if x is None:
        return None
    return list(x) if not isinstance(x, list) else x

def _maybe_ts_from_group_df(df: pd.DataFrame, cols: list[str] | None) -> Optional[List[TimeSeries]]:
    if not cols:
        return None
    return TimeSeries.from_group_dataframe(
        df=df[[TIME_COL, GROUP_COL] + cols].copy(),
        time_col=TIME_COL,
        group_cols=GROUP_COL,
        value_cols=cols,
        freq=FREQ,
    )

def _build_statics_transformer():
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scale", MinMaxScaler())])
    cat_enc  = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    tr = StaticCovariatesTransformer(
        transformer_num=num_pipe,
        transformer_cat=cat_enc,
        cols_num=STATIC_REALS,
        cols_cat=STATIC_CATS,
    )
    return tr

def _save_mpl(ax_or_fig, out_path: Path, figsize: tuple[float, float] | None = None) -> None:
    """
    Save a Matplotlib plot robustly whether the plotting function returned:
      - a Figure
      - an Axes (we grab its .figure)
      - None (we use plt.gcf())
    Always closes the figure afterwards.
    """
    try:
        if ax_or_fig is None:
            fig = plt.gcf()
        elif hasattr(ax_or_fig, "savefig"):  # Figure
            fig = ax_or_fig
        elif hasattr(ax_or_fig, "figure"):   # Axes
            fig = ax_or_fig.figure
        else:
            fig = plt.gcf()

        if figsize is not None and hasattr(fig, "set_size_inches"):
            fig.set_size_inches(*figsize)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
    finally:
        try:
            plt.close(fig)
        except Exception:
            plt.close("all")

def _explain_one(explainer: TFTExplainer, s, pc, fc):
    """
    Robust single-series explain() with a fallback that trims the last step
    (to avoid occasional off-by-one horizon issues in some Darts builds).
    """
    try:
        return explainer.explain(
            foreground_series=s,
            foreground_past_covariates=pc,
            foreground_future_covariates=fc,
        )
    except Exception:
        try:
            if len(s) > 1:
                return explainer.explain(
                    foreground_series=s[:-1],
                    foreground_past_covariates=pc[:-1] if (pc is not None and len(pc) == len(s)) else pc,
                    foreground_future_covariates=fc[:-1] if (fc is not None and len(fc) >= len(s)) else fc,
                )
        except Exception:
            pass
        raise

def _fi_rows_from_result(expl_res, series_idx: int | None = None, well_id: int | None = None):
    rows = []
    fi = expl_res.get_feature_importances()  # dict: block -> {feature: value}
    for block, data in fi.items():
        for feat, val in data.items():
            rows.append({
                "series_idx": series_idx,
                "well_id": well_id,
                "block": block,
                "feature": feat,
                "importance": float(val),
            })
    return rows

def _series_idx_to_well_id(series_list: List[TimeSeries]) -> list[int]:
    """
    Extract well_id from each TimeSeries static covariate.
    Assumes GROUP_COL was included in static_cols during creation.
    """
    out = []
    for s in series_list:
        try:
            gid = s.static_covariates[GROUP_COL].values[0]
            gid = int(gid) if not isinstance(gid, (int, np.integer)) else gid
        except Exception:
            gid = None
        out.append(gid)
    return out

def _well_ids_to_indices(series_list: List[TimeSeries], desired_ids: list[int]) -> list[int]:
    map_ids = _series_idx_to_well_id(series_list)
    id_to_idx = {gid: i for i, gid in enumerate(map_ids)}
    idxs = []
    for wid in desired_ids:
        if wid in id_to_idx:
            idxs.append(id_to_idx[wid])
        else:
            print(f"⚠️ well_id {wid} not found in series list; skipping.")
    return idxs


# =========================
# Exporters
# =========================

def export_local_vi_and_attention(
    explainer: TFTExplainer,
    series_scaled_list: List[TimeSeries],
    past_covs_scaled_list: Optional[List[TimeSeries]],
    future_covs_scaled_list: Optional[List[TimeSeries]],
    indices: list[int],
    save_by: str,  # "well" | "index"
    out_dir: Path,
    attention_types: set[str],  # {"all", "heatmap"}
):
    """
    For the selected indices, export local variable importance and attention plots.
    Returns DataFrame with concatenated local VI rows.
    """
    out_local = out_dir / "local"
    out_attn  = out_dir / "attention"
    ensure_dir(out_local); ensure_dir(out_attn)

    # map idx -> well id (if available)
    idx_to_well = _series_idx_to_well_id(series_scaled_list)

    rows, succeeded, failed = [], [], []

    # convenience indexers for covs
    pc_list = _as_list(past_covs_scaled_list)
    fc_list = _as_list(future_covs_scaled_list)

    for i in indices:
        s = series_scaled_list[i]
        pc = None if pc_list is None else pc_list[i]
        fc = None if fc_list is None else fc_list[i]

        wid = idx_to_well[i]
        fname_stub = f"well_{wid}" if (save_by == "well" and wid is not None) else f"{i:03d}"
        log_label  = f"well {wid}" if (save_by == "well" and wid is not None) else f"idx {i}"

        try:
            expl_res = _explain_one(explainer, s, pc, fc)
        except Exception as e:
            print(f"⚠️ Skipping {log_label}: explain() failed with: {e}")
            failed.append(i)
            continue

        # accumulate numeric importances
        rows += _fi_rows_from_result(expl_res, series_idx=i, well_id=wid)
        succeeded.append(i)

        # Local VI plot (robust to Axes/Figure/None returns)
        try:
            # This call may return an Axes, a Figure, or None (and do a plt.show()).
            _ = explainer.plot_variable_selection(expl_res, fig_size=None)

            # Always recover the active figure safely
            fig = None
            if isinstance(_, plt.Axes):
                fig = _.get_figure()
            elif isinstance(_, plt.Figure):
                fig = _
            else:
                fig = plt.gcf()

            if fig is None:
                raise RuntimeError("No figure produced by plot_variable_selection().")

            # ---------- NEW: make the figure tall enough for all bars ----------
            # Count how many labels we’re trying to show across all subplots
            try:
                n_labels = sum(len(ax.get_yticklabels()) for ax in fig.axes)
            except Exception:
                n_labels = 20  # sensible fallback

            # Base width; height grows with number of labels (≈0.32" per label)
            width_in  = 11.0
            height_in = max(7.0, 0.32 * n_labels + 2.5)

            # Smaller fonts + a larger left margin for long feature names
            for ax in fig.axes:
                ax.tick_params(axis="y", labelsize=8)   # ↓ from default
                ax.tick_params(axis="x", labelsize=8)
                if ax.get_xlabel():
                    ax.set_xlabel(ax.get_xlabel(), fontsize=9)
                if ax.get_ylabel():
                    ax.set_ylabel(ax.get_ylabel(), fontsize=9)
                if ax.get_title():
                    ax.set_title(ax.get_title(), fontsize=11, pad=6)

                # More room on the left for long y tick labels
                # (we also keep a bit of top/bottom space for titles)
                fig.subplots_adjust(left=0.32, right=0.97, top=0.93, bottom=0.08)

                # Slight vertical padding so bar tops/bottoms aren’t cramped
                try:
                    ax.margins(y=0.05)
                except Exception:
                    pass

            fig.set_size_inches(width_in, height_in)
            fig.tight_layout()

            # Save and close
            fig.savefig(out_local / f"local_variable_importance_{fname_stub}.png",
                        dpi=180, bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            print(f"⚠️ Could not plot local VI for {log_label}: {e}")


        # Attention plots (all / heatmap)
        if "all" in attention_types:
            try:
                ax_or_fig = explainer.plot_attention(
                    expl_res, plot_type="all", show_index_as="relative", max_nr_series=5
                )
                _save_mpl(ax_or_fig, out_attn / f"attention_{fname_stub}.png", figsize=(11, 5))
            except Exception as e:
                print(f"⚠️ Could not plot attention (all) for {log_label}: {e}")

        if "heatmap" in attention_types:
            try:
                ax_or_fig = explainer.plot_attention(
                    expl_res, plot_type="heatmap", show_index_as="relative", max_nr_series=5
                )
                _save_mpl(ax_or_fig, out_attn / f"attention_heatmap_{fname_stub}.png", figsize=(8, 6))
            except Exception as e:
                print(f"⚠️ Could not plot attention (heatmap) for {log_label}: {e}")

    # write merged CSV
    df_local = pd.DataFrame(rows).sort_values(
        ["well_id", "series_idx", "block", "importance"], ascending=[True, True, True, False]
    )
    if not df_local.empty:
        df_local.to_csv(out_local / "local_variable_importance.csv", index=False)

    print(f"✅ Local VI/attention done. Succeeded: {len(succeeded)} | Failed: {len(failed)}")
    return df_local


def export_pseudo_global_vi(
    explainer: TFTExplainer,
    series_scaled_list: List[TimeSeries],
    past_covs_scaled_list: Optional[List[TimeSeries]],
    future_covs_scaled_list: Optional[List[TimeSeries]],
    sample_indices: Optional[list[int]],
    out_dir: Path,
):
    """
    Compute 'global' VI by averaging local VI over a sample of wells.
    More robust than explainer.explain() in global mode for some Darts versions.
    """
    ensure_dir(out_dir)

    series_scaled_list = _as_list(series_scaled_list)
    pc_list = _as_list(past_covs_scaled_list)
    fc_list = _as_list(future_covs_scaled_list)

    if sample_indices is None:
        n = len(series_scaled_list)
        k = min(20, n)
        sample_indices = sorted(set(np.linspace(0, n - 1, num=k, dtype=int).tolist()))
    else:
        sample_indices = list(sample_indices)

    idx_to_well = _series_idx_to_well_id(series_scaled_list)

    rows, ok = [], 0
    for i in sample_indices:
        s = series_scaled_list[i]
        pc = None if pc_list is None else pc_list[i]
        fc = None if fc_list is None else fc_list[i]
        wid = idx_to_well[i]
        try:
            expl_res = _explain_one(explainer, s, pc, fc)
            rows += _fi_rows_from_result(expl_res, series_idx=i, well_id=wid)
            ok += 1
        except Exception as e:
            print(f"⚠️ Skipping idx={i} for pseudo-global VI: {e}")

    if ok == 0:
        print("⚠️ No series explained; pseudo-global VI unavailable.")
        return pd.DataFrame(columns=["block", "feature", "importance_mean", "importance_sd", "n_series"])

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["block", "feature"], as_index=False)["importance"]
        .agg(importance_mean="mean", importance_sd="std", n_series="count")
        .sort_values(["block", "importance_mean"], ascending=[True, False])
    )
    agg.to_csv(out_dir / "global_variable_importance_pseudo.csv", index=False)
    print(f"✅ Pseudo-global VI computed over {ok} wells → {out_dir/'global_variable_importance_pseudo.csv'}")
    return agg


# =========================
# Main
# =========================

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TFT Explainability (local + pseudo-global)")
    parser.add_argument("--model-path", type=str, default=str(artifact_path("runs") / "search_bo/2025-09-12_10-02-41__bo_baseline_scaler/model_best.pt"))
    parser.add_argument("--eval-set", type=str, choices=["test", "train"], default="test",
                        help="use TEST (default) or TRAIN for foreground explanations")
    parser.add_argument("--local-wells", type=str, default="0,5,18",
                        help="comma-separated WELL IDs or 'all' for all wells")
    parser.add_argument("--save-by", type=str, choices=["well", "index"], default="well",
                        help="use well id or series index in filenames")
    parser.add_argument("--attention-types", type=str, default="all",
                        help="comma-separated attention types from {all,heatmap}")
    parser.add_argument("--global-sample", type=str, default="",
                        help="comma-separated indices for pseudo-global VI (empty => auto sample)")
    parser.add_argument("--matmul-precision", type=str, choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", type=str, default=str(Path(ARTIFACTS) / "explain/"))
    args = parser.parse_args(argv)

    torch.set_float32_matmul_precision(args.matmul_precision)
    set_seeds(args.seed)

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    # -------- Load data --------
    df_all = pd.read_csv(ALL_FILE, parse_dates=[TIME_COL])
    df_test = pd.read_csv(TEST_FILE, parse_dates=[TIME_COL])

    # statics transformer (fit on ALL, apply to both)
    static_tr = _build_statics_transformer()

    series_all = TimeSeries.from_group_dataframe(
        df=df_all[[TIME_COL, GROUP_COL, TARGET] + STATIC_CATS + STATIC_REALS].copy(),
        time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=TARGET, static_cols=STATIC_CATS + STATIC_REALS, freq=FREQ,
    )
    series_all = static_tr.fit_transform(series_all)

    # EVAL set selection
    df_eval = df_test if args.eval_set == "test" else df_all

    series_eval = TimeSeries.from_group_dataframe(
        df=df_eval[[TIME_COL, GROUP_COL, TARGET] + STATIC_CATS + STATIC_REALS].copy(),
        time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=TARGET, static_cols=STATIC_CATS + STATIC_REALS, freq=FREQ,
    )
    series_eval = static_tr.transform(series_eval)

    # covariates
    past_all  = _maybe_ts_from_group_df(df_all, PAST_REALS)
    past_eval = _maybe_ts_from_group_df(df_eval, PAST_REALS)

    # future covs (ensure coverage)
    
    fut_age_all = build_future_covariates_well_age_full(
        df=df_all, time_col=TIME_COL, group_col=GROUP_COL,
        completion_col="completion_date", horizon=12, freq=FREQ
    )
    
    fut_dca_all, dca_pars_all = build_future_covariates_dca_full(
        df=df_all, time_col=TIME_COL, group_col=GROUP_COL,
        target_col=TARGET, horizon=12, freq=FREQ,
        fit_k=INPUT_CHUNK, params_out=True
    )
    
    # after building fut_dca with 'dca_bpd'
    fut_dca_all["dca_bpd_log1p"] = np.log1p(fut_dca_all["dca_bpd"])
    
    # merge the FUTURE covariates on [group, time]
    fut_df_all = fut_age_all.merge(fut_dca_all[[GROUP_COL, TIME_COL, "dca_bpd_log1p"]], on=[GROUP_COL, TIME_COL], how="left")
    
    fut_age_eval = build_future_covariates_well_age_full(
        df=df_eval, time_col=TIME_COL, group_col=GROUP_COL,
        completion_col="completion_date", horizon=12, freq=FREQ
    )
    
    fut_dca_eval, dca_pars_eval = build_future_covariates_dca_full(
        df=df_eval, time_col=TIME_COL, group_col=GROUP_COL,
        target_col=TARGET, horizon=12, freq=FREQ,
        fit_k=INPUT_CHUNK, params_out=True
    )
    
    # after building fut_dca with 'dca_bpd'
    fut_dca_eval["dca_bpd_log1p"] = np.log1p(fut_dca_eval["dca_bpd"])
    
        # merge the FUTURE covariates on [group, time]
    fut_df_eval = fut_age_eval.merge(fut_dca_eval[[GROUP_COL, TIME_COL, "dca_bpd_log1p"]], on=[GROUP_COL, TIME_COL], how="left")
    
    
    # fut_all_df = build_future_covariates_well_age_full(
    #     df=df_all, time_col=TIME_COL, group_col=GROUP_COL,
    #     completion_col="completion_date", horizon=12, freq=FREQ,
    # )
    # fut_eval_df = build_future_covariates_well_age_full(
    #     df=df_eval, time_col=TIME_COL, group_col=GROUP_COL,
    #     completion_col="completion_date", horizon=12, freq=FREQ,
    # )
    future_all = TimeSeries.from_group_dataframe(
        df=fut_df_all, time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=FUTURE_REALS, freq=FREQ,
    )
    future_eval = TimeSeries.from_group_dataframe(
        df=fut_df_eval, time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=FUTURE_REALS, freq=FREQ,
    )

    # scalers (fit on ALL, apply to EVAL)
    y_scaler = Scaler(); y_scaler.fit(series_all)
    series_eval_scaled = y_scaler.transform(series_eval)

    x_scaler_past = Scaler() if past_all is not None else None
    x_scaler_future = Scaler() if future_all is not None else None

    if x_scaler_past:
        past_all_ts = TimeSeries.from_group_dataframe(
            df=df_all[[TIME_COL, GROUP_COL] + PAST_REALS].copy(),
            time_col=TIME_COL, group_cols=GROUP_COL,
            value_cols=PAST_REALS, freq=FREQ,
        )
        _ = x_scaler_past.fit_transform(past_all_ts)
        past_eval_scaled = x_scaler_past.transform(past_eval) if past_eval is not None else None
    else:
        past_eval_scaled = None

    if x_scaler_future:
        _ = x_scaler_future.fit_transform(future_all)
        future_eval_scaled = x_scaler_future.transform(future_eval)
    else:
        future_eval_scaled = None

    # -------- Load model --------
    from darts.models import TFTModel
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = TFTModel.load(str(model_path))

    # -------- Build explainer (background uses ALL shapes) --------
    explainer = TFTExplainer(
        model=model,
        background_series=series_all,
        background_past_covariates=past_all if past_all is not None else None,
        background_future_covariates=future_all if future_all is not None else None,
    )

    # -------- Resolve which series to explain (by well IDs) --------
    all_eval_well_ids = _series_idx_to_well_id(series_eval)  # in display order
    if args.local_wells.strip().lower() == "all":
        indices = list(range(len(series_eval_scaled)))
    else:
        # parse as list of well IDs
        req_wells = [int(x) for x in args.local_wells.split(",") if x.strip().isdigit()]
        indices = _well_ids_to_indices(series_eval, req_wells)
        if not indices:
            print("⚠️ No valid wells resolved from --local-wells; exiting.")
            return

    # -------- Attention types --------
    att_types = set([x.strip().lower() for x in args.attention_types.split(",") if x.strip()])
    att_types = {t for t in att_types if t in {"all", "heatmap"}}
    if not att_types:
        att_types = {"all"}

    # -------- Export local VI + attention --------
    df_local = export_local_vi_and_attention(
        explainer=explainer,
        series_scaled_list=series_eval_scaled,
        past_covs_scaled_list=past_eval_scaled,
        future_covs_scaled_list=future_eval_scaled,
        indices=indices,
        save_by=args.save_by,
        out_dir=Path(args.out_root),
        attention_types=att_types,
    )

    # -------- Pseudo-global VI (sample defaults to auto) --------
    if args.global_sample.strip():
        sample_idxs = [int(x) for x in args.global_sample.split(",") if x.strip().isdigit()]
    else:
        sample_idxs = None
    export_pseudo_global_vi(
        explainer=explainer,
        series_scaled_list=series_eval_scaled,
        past_covs_scaled_list=past_eval_scaled,
        future_covs_scaled_list=future_eval_scaled,
        sample_indices=sample_idxs,
        out_dir=Path(args.out_root) / "global",
    )

    print(f"✅ Explainability done. Outputs in: {args.out_root}")


if __name__ == "__main__":
    main()
