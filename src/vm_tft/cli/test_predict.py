# src/vm_tft/cli/test_predict.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import (
    Scaler,
    StaticCovariatesTransformer,
    InvertibleMapper,
)
from darts.dataprocessing.pipeline import Pipeline as DartsPipeline

from vm_tft.cfg import (
    ALL_FILE, TEST_FILE, FREQ,
    TIME_COL, GROUP_COL, TARGET,
    STATIC_CATS, STATIC_REALS,
    PAST_REALS, FUTURE_REALS,
    INPUT_CHUNK, FORECAST_H,
)
from vm_tft.features import build_future_covariates_well_age_full, build_future_covariates_dca_full
from vm_tft.io_utils import (
    ensure_dir, artifact_path, set_seeds, new_run_dir, copy_as_pointer, write_json
)
from vm_tft.evaluate import metrics_from_results


# -----------------------------
# scalers / helpers
# -----------------------------
def _make_scalers(
    use_log1p: bool,
    log_future: bool,
    has_past: bool,
    has_future: bool,
):
    """
    Build Darts transformers:
      y:      (log1p -> MinMax) or (MinMax)
      past:   same as y when present
      future: MinMax; add log1p only if log_future=True
    """
    if use_log1p:
        log_mapper = InvertibleMapper(fn=np.log1p, inverse_fn=np.expm1, name="log1p")
        y_scaler = DartsPipeline([log_mapper, Scaler()])
        x_past   = DartsPipeline([log_mapper, Scaler()]) if has_past else None
        if has_future:
            x_fut = DartsPipeline([log_mapper, Scaler()]) if log_future else Scaler()
        else:
            x_fut = None
    else:
        y_scaler = Scaler()
        x_past   = Scaler() if has_past else None
        x_fut    = Scaler() if has_future else None
    return y_scaler, x_past, x_fut


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


def _slice_warm(ts: TimeSeries, warm_months: int, input_chunk_length: int) -> TimeSeries:
    # Use the FIRST months to mimic early production warm-start
    need = max(warm_months, input_chunk_length)
    return ts if len(ts) <= need else ts[:need]


def _step_of(ts: TimeSeries):
    # robustly infer a pandas offset
    if getattr(ts, "freq", None) is not None:
        return to_offset(ts.freq)
    fs = getattr(ts, "freq_str", None) or pd.infer_freq(ts.time_index)
    return to_offset(fs)


def _align_future_covs_for_warm(
    warm_ts: TimeSeries,
    fut_cov: Optional[TimeSeries],
    horizon: int,
    input_chunk_length: int,
) -> Optional[TimeSeries]:
    if fut_cov is None:
        return None
    step = _step_of(warm_ts)
    enc_end = warm_ts.end_time()
    enc_start = warm_ts.start_time()  # warm_ts already has min required encoder length
    dec_end = enc_end + horizon * step if horizon is not None else fut_cov.end_time()
    # Make sure we cover encoder_start..decoder_end
    return fut_cov.slice(enc_start, dec_end)


def _ts_to_df(ts: TimeSeries, name: str) -> pd.DataFrame:
    vals = ts.values(copy=False)
    if vals.ndim == 3:
        vals = vals[:, 0, 0]
    elif vals.ndim == 2:
        vals = vals[:, 0]
    return pd.DataFrame({name: vals}, index=ts.time_index)

def _target_index_for_window(warm_ts: TimeSeries, horizon: int) -> pd.DatetimeIndex:
    step = _step_of(warm_ts)
    start = warm_ts.start_time()
    end   = warm_ts.end_time() + horizon * step
    return pd.date_range(start=start, end=end, freq=step)


def _ensure_past_cov_coverage_scaled(
    warm_ts: TimeSeries,
    pc_scaled: Optional[TimeSeries],
    horizon: int,
) -> Optional[TimeSeries]:
    """
    Ensure past_covariates (already scaled) span [warm.start .. warm.end + horizon].
    Forward-fill beyond their last timestamp using last available values.
    """
    if pc_scaled is None:
        return None

    tgt_idx = _target_index_for_window(warm_ts, horizon)
    # to pandas
    vals = pc_scaled.values(copy=False)
    if vals.ndim == 3:  # (time, comp, sample) -> (time, comp)
        vals = vals[:, :, 0]
    if vals.ndim == 1:
        vals = vals[:, None]

    df = pd.DataFrame(vals, index=pc_scaled.time_index)
    # slice/reindex to target window and ffill
    df = df.reindex(tgt_idx).ffill()

    # rebuild TimeSeries; preserve static covariates & component names if any
    comp_names = getattr(pc_scaled, "component_names", None)
    out = TimeSeries.from_times_and_values(
        times=tgt_idx,
        values=df.values,
        columns=comp_names,
        static_covariates=pc_scaled.static_covariates,
        freq=pc_scaled.freq if hasattr(pc_scaled, "freq") else None,
    )
    return out

def _debug_check_pc_coverage(warm_ts, pc, horizon, label=""):
    freq_str = getattr(warm_ts, "freq_str", None) or pd.infer_freq(warm_ts.time_index)
    step = to_offset(freq_str)
    enc_start = warm_ts.start_time()
    enc_end   = warm_ts.end_time()
    dec_end   = enc_end + horizon * step
    ok_start = pc.start_time() <= enc_start
    ok_end   = pc.end_time()   >= dec_end
    if not (ok_start and ok_end):
        print(f"[PC COVERAGE {label}] start={pc.start_time()} end={pc.end_time()} "
              f"| need start≤{enc_start}, end≥{dec_end}")

# =========================
# Inference core
# =========================

def predict_with_truth(
    model,
    series_list: List[TimeSeries],
    y_scaler: Scaler,
    warm_months: int,
    horizon: Optional[int],
    past_covs_list: Optional[List[TimeSeries]] = None,      # pass if the model was trained with them
    future_covs_list: Optional[List[TimeSeries]] = None,
    num_samples: int = 200,
    eval_until_end: bool = False,
    input_chunk_length: int = INPUT_CHUNK,
):
    """
    Warm-start predictions on unseen wells.
    Truth is aligned by TIME to the prediction window to avoid off-by-one issues.
    """
    out = []
    for i, ts in enumerate(series_list):
        if len(ts) <= warm_months:
            continue

        warm_ts = _slice_warm(ts, warm_months, input_chunk_length)

        if eval_until_end:
            horizon_i = len(ts) - len(warm_ts)
        else:
            if horizon is None:
                raise ValueError("horizon must be provided when eval_until_end=False")
            horizon_i = int(horizon)

        pc = None if past_covs_list is None else past_covs_list[i]
        fc = None
        if future_covs_list is not None:
            fc = _align_future_covs_for_warm(warm_ts, future_covs_list[i], horizon_i, input_chunk_length)

        # Predict in scaled space:
        pred_scaled = model.predict(
            n=horizon_i,
            series=warm_ts,
            past_covariates=pc,
            future_covariates=fc,
            num_samples=num_samples,
            show_warnings=False,
            verbose=False,
        )

        # Inverse-transform to original units:
        warm_ts_inv = y_scaler.inverse_transform(warm_ts)
        pred        = y_scaler.inverse_transform(pred_scaled)

        # Align truth by TIME to the pred window, then invert:
        truth_slice = ts.slice(pred_scaled.start_time(), pred_scaled.end_time())
        truth       = y_scaler.inverse_transform(truth_slice)

        out.append({
            "series_idx": i,
            "warm_ts": warm_ts_inv,
            "pred": pred,
            "truth": truth,
        })
    return out


def predict_future(
    model,
    series_list: List[TimeSeries],
    y_scaler: Scaler,
    warm_months: int,
    *,
    total_months: int | None = None,   # e.g., 60 for 5 years total from start
    future_months: int | None = None,  # alternative: fixed horizon beyond warm slice
    past_covs_list: Optional[List[TimeSeries]] = None,      # pass if used at train time
    future_covs_list: Optional[List[TimeSeries]] = None,
    num_samples: int = 200,
    input_chunk_length: int = INPUT_CHUNK,
):
    """
    Case C: forecast beyond the warm slice (no truth returned).

    - If `total_months` is provided, horizon_i = max(0, total_months - len(warm_ts)).
    - Else if `future_months` is provided, horizon_i = future_months.
    - Predictions start right after the warm slice and proceed sequentially.
    - Uses the same covariate alignment pattern as `predict_with_truth()`.
    """
    if (total_months is None) and (future_months is None):
        raise ValueError("Provide either `total_months` or `future_months`.")

    outputs = []
    for i, ts in enumerate(series_list):
        if len(ts) <= warm_months:
            continue

        warm_ts = _slice_warm(ts, warm_months, input_chunk_length)

        # per-series horizon
        if total_months is not None:
            horizon_i = max(0, int(total_months) - len(warm_ts))
            if horizon_i == 0:
                continue
        else:
            horizon_i = int(future_months)

        pc = None if past_covs_list is None else past_covs_list[i]
        fc = None
        if future_covs_list is not None:
            fc = _align_future_covs_for_warm(
                warm_ts=warm_ts,
                fut_cov=future_covs_list[i],
                horizon=horizon_i,
                input_chunk_length=input_chunk_length,
            )
            
        # NEW: extend past covs to cover the full prediction window
        pc_ext = _ensure_past_cov_coverage_scaled(warm_ts, pc, horizon_i)

        # predict in scaled space
        pred_scaled = model.predict(
            n=horizon_i,
            series=warm_ts,
            past_covariates=pc_ext,   # <-- use the extended past covs
            future_covariates=fc,
            num_samples=num_samples,
            show_warnings=False,
            verbose=False,
        )

        outputs.append({
            "series_idx": i,
            "warm_ts": y_scaler.inverse_transform(warm_ts),
            "pred": y_scaler.inverse_transform(pred_scaled),
        })

    return outputs


# =========================
# Plotting
# =========================

def _save_plot(fig: plt.Figure, path: Path, dpi: int = 180):
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_prediction_vs_truth(res, title: str) -> plt.Figure:
    warm_ts, pred, truth = res["warm_ts"], res["pred"], res["truth"]
    fig, ax = plt.subplots(figsize=(10, 6))
    warm_ts.plot(ax=ax, label="Warm start (history)", lw=2)
    if len(truth) > 0:
        truth.plot(ax=ax, label="Actual (truth)", lw=2)
    try:
        if getattr(pred, "n_samples", 1) > 1:
            pred.plot(ax=ax, low_quantile=0.10, high_quantile=0.90, label="Pred 10–90%", alpha=0.3)
            pred.quantile(0.50).plot(ax=ax, label="Pred median (q0.50)", lw=2)
        else:
            pred.plot(ax=ax, label="Pred (det)", lw=2)
    except Exception:
        pred.plot(ax=ax, label="Pred", lw=2)
    ax.set_title(title)
    ax.set_xlabel("Date"); ax.set_ylabel(TARGET)
    ax.grid(True); ax.legend()
    fig.tight_layout()
    return fig


def _plot_prediction_future(res, title: str) -> plt.Figure:
    warm_ts, pred = res["warm_ts"], res["pred"]
    fig, ax = plt.subplots(figsize=(10, 6))
    warm_ts.plot(ax=ax, label="Warm start (history)", lw=2)
    try:
        if getattr(pred, "n_samples", 1) > 1:
            pred.plot(ax=ax, low_quantile=0.10, high_quantile=0.90, label="Pred 10–90%", alpha=0.3)
            pred.quantile(0.50).plot(ax=ax, label="Pred median (q0.50)", lw=2)
        else:
            pred.plot(ax=ax, label="Pred (det)", lw=2)
    except Exception:
        pred.plot(ax=ax, label="Pred", lw=2)
    ax.set_title(title)
    ax.set_xlabel("Date"); ax.set_ylabel(TARGET)
    ax.grid(True); ax.legend()
    fig.tight_layout()
    return fig


# =========================
# CLI
# =========================

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Warm-start evaluation on TEST set (Case B) and Future Forecasts (Case C)")
    parser.add_argument("--model-path", type=str, default=str(artifact_path("runs") / "search/2025-09-11_03-48-54__baseline_again/model_best.pt"))
    parser.add_argument("--warm-months", type=int, default=36, help="months of initial history provided")

    # Which cases to run (A removed)
    parser.add_argument("--case", type=str, choices=["B", "C", "both"], default="both",
                        help="B=until end of history; C=forecast to a total of N months from start")

    # Case C controls
    parser.add_argument("--total-months", type=int, default=60,
                        help="Case C: total production months from start (e.g., 60 for 5 years). "
                             "Horizon per well = max(0, total_months - len(warm_slice)).")

    parser.add_argument("--samples", type=int, default=200, help="num_samples for predict()")
    parser.add_argument("--matmul-precision", type=str, choices=["highest","high","medium"], default="high")
    parser.add_argument("--seed", type=int, default=42)

    # export / plotting
    parser.add_argument("--save-plots", action="store_true", help="save PNGs to artifacts/predictions/plots")
    parser.add_argument("--export-csv", action="store_true", help="export per-well CSVs (always all wells)")
    parser.add_argument("--select-by", type=str, choices=["index","well"], default="well",
                        help="interpret --plot-indices as series indices or well IDs")
    parser.add_argument("--plot-indices", type=str, default="",
                        help="comma-separated indices or well IDs to plot (e.g. 0,5,18 or 161720,161952)")
    parser.add_argument("--plot-all", action="store_true", help="plot PNGs for ALL wells (ignores the default 12-cap)")
    parser.add_argument("--run-tag", type=str, default="test", help="Suffix tag for the run folder name.")

    # transforms
    parser.add_argument("--no-log1p", action="store_true", help="Disable log1p (default: enabled).")
    parser.add_argument("--log-future", action="store_true",
                        help="Also apply log1p to future covariates (default: off).")

    args = parser.parse_args(argv)
    torch.set_float32_matmul_precision(args.matmul_precision)
    set_seeds(args.seed)

    # ---------------- load data ----------------
    df_all  = pd.read_csv(ALL_FILE,  parse_dates=[TIME_COL])
    df_test = pd.read_csv(TEST_FILE, parse_dates=[TIME_COL])

    # target series with statics (fit static transformer on ALL, apply to TEST)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scale", MinMaxScaler())])
    cat_enc  = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    static_tr = StaticCovariatesTransformer(
        transformer_num=num_pipe,
        transformer_cat=cat_enc,
        cols_num=STATIC_REALS,
        cols_cat=STATIC_CATS,
    )

    series_all = TimeSeries.from_group_dataframe(
        df=df_all[[TIME_COL, GROUP_COL, TARGET] + STATIC_CATS + STATIC_REALS].copy(),
        time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=TARGET, static_cols=STATIC_CATS + STATIC_REALS, freq=FREQ,
    )
    series_all = static_tr.fit_transform(series_all)

    series_test = TimeSeries.from_group_dataframe(
        df=df_test[[TIME_COL, GROUP_COL, TARGET] + STATIC_CATS + STATIC_REALS].copy(),
        time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=TARGET, static_cols=STATIC_CATS + STATIC_REALS, freq=FREQ,
    )
    series_test = static_tr.transform(series_test)

    # ---------- covariates ----------
    past_covs_all  = _maybe_ts_from_group_df(df_all,  PAST_REALS)
    past_covs_test = _maybe_ts_from_group_df(df_test, PAST_REALS)
    
    cov_horizon_test = max(
        FORECAST_H,                                   # keep B happy (it only needs to reach end_hist)
        max(0, int(args.total_months) - max(args.warm_months, INPUT_CHUNK))  # Case C need
    )
    
    # fut_age_all = build_future_covariates_well_age_full(
    #     df=df_all, time_col=TIME_COL, group_col=GROUP_COL,
    #     completion_col="completion_date", horizon=FORECAST_H, freq=FREQ
    # )
    
    # fut_dca_all, dca_pars_all = build_future_covariates_dca_full(
    #     df=df_all, time_col=TIME_COL, group_col=GROUP_COL,
    #     target_col=TARGET, horizon=FORECAST_H, freq=FREQ,
    #     fit_k=INPUT_CHUNK, params_out=True
    # )
    
    # # after building fut_dca with 'dca_bpd'
    # fut_dca_all["dca_bpd_log1p"] = np.log1p(fut_dca_all["dca_bpd"])
    
    # # merge the FUTURE covariates on [group, time]
    # fut_df_all = fut_age_all.merge(fut_dca_all[[GROUP_COL, TIME_COL, "dca_bpd_log1p"]], on=[GROUP_COL, TIME_COL], how="left")
    
    fut_df_all = build_future_covariates_well_age_full(
        df=df_all, time_col=TIME_COL, group_col=GROUP_COL,
        completion_col="completion_date",
        horizon=FORECAST_H, freq=FREQ,
    )
    
    future_covs_all = TimeSeries.from_group_dataframe(
        df=fut_df_all, time_col=TIME_COL, group_cols=GROUP_COL, value_cols=FUTURE_REALS, freq=FREQ,
    )
    
    # Test
    
    # fut_age_test = build_future_covariates_well_age_full(
    #     df=df_test, time_col=TIME_COL, group_col=GROUP_COL,
    #     completion_col="completion_date", horizon=cov_horizon_test, freq=FREQ
    # )
    
    # fut_dca_test, dca_pars_test = build_future_covariates_dca_full(
    #     df=df_test, time_col=TIME_COL, group_col=GROUP_COL,
    #     target_col=TARGET, horizon=cov_horizon_test, freq=FREQ,
    #     fit_k=INPUT_CHUNK, params_out=True
    # )
    
    # # after building fut_dca with 'dca_bpd'
    # fut_dca_test["dca_bpd_log1p"] = np.log1p(fut_dca_test["dca_bpd"])
    
    # # merge the FUTURE covariates on [group, time]
    # fut_df_test = fut_age_test.merge(fut_dca_test[[GROUP_COL, TIME_COL, "dca_bpd_log1p"]], on=[GROUP_COL, TIME_COL], how="left")    
    
    
    fut_df_test = build_future_covariates_well_age_full(
        df=df_test, time_col=TIME_COL, group_col=GROUP_COL,
        completion_col="completion_date",
        horizon=cov_horizon_test, freq=FREQ,
    )
    
    future_covs_test = TimeSeries.from_group_dataframe(
        df=fut_df_test, time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=FUTURE_REALS, freq=FREQ,
    )

    # scalers (fit on ALL → transform TEST)
    use_log1p = not args.no_log1p
    y_scaler, x_scaler_past, x_scaler_future = _make_scalers(
        use_log1p=use_log1p, log_future=args.log_future,
        has_past=(past_covs_test is not None), has_future=(future_covs_test is not None),
    )
    y_scaler.fit(series_all)
    series_test_scaled = y_scaler.transform(series_test)

    if x_scaler_past and past_covs_test is not None:
        x_scaler_past.fit(past_covs_all)
        past_covs_test_scaled = x_scaler_past.transform(past_covs_test)
    else:
        past_covs_test_scaled = None

    if x_scaler_future and future_covs_test is not None:
        x_scaler_future.fit(future_covs_all)
        future_covs_test_scaled = x_scaler_future.transform(future_covs_test)
    else:
        future_covs_test_scaled = None

    # ---------------- load model ----------------
    from darts.models import TFTModel
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = TFTModel.load(str(model_path))

    # ---------------- run predictions ----------------
    run_dir = new_run_dir(kind="test", tag=args.run_tag); ensure_dir(run_dir)
    fixed_preds_dir = artifact_path("predictions") / "plots"; ensure_dir(fixed_preds_dir)

    do_B = (args.case in ("B", "both"))
    do_C = (args.case in ("C", "both"))

    # map well_id(s) to series_idx if plotting/exporting by well id
    well_ids = df_test[GROUP_COL].dropna().unique().tolist()
    id_to_idx = {wid: i for i, wid in enumerate(sorted(well_ids))}

    results_B = results_C = None
    if do_B:
        results_B = predict_with_truth(
            model=model,
            series_list=list(series_test_scaled),
            y_scaler=y_scaler,
            warm_months=args.warm_months,
            horizon=None,  # ignored (eval_until_end=True)
            past_covs_list=None if past_covs_test_scaled is None else list(past_covs_test_scaled),
            future_covs_list=list(future_covs_test_scaled),
            num_samples=args.samples,
            eval_until_end=True,
            input_chunk_length=INPUT_CHUNK,
        )
        metrics_B = metrics_from_results(results_B)
        (run_dir / "caseB_overall_mean.json").write_text(metrics_B["overall_mean"].to_json(indent=2))
        metrics_B["per_series"].to_csv(run_dir / "caseB_per_series.csv", index=False)
        copy_as_pointer(run_dir / "caseB_overall_mean.json", artifact_path("predictions") / "caseB_overall_mean.json")

    if do_C:
        results_C = predict_future(
            model=model,
            series_list=list(series_test_scaled),
            y_scaler=y_scaler,
            warm_months=args.warm_months,
            total_months=int(args.total_months),   # e.g., 60 → 5 years total
            past_covs_list=None if past_covs_test_scaled is None else list(past_covs_test_scaled),
            future_covs_list=list(future_covs_test_scaled),
            num_samples=args.samples,
            input_chunk_length=INPUT_CHUNK,
        )
        # No metrics for Case C

    # ---------------- export per-well CSVs & PNGs ----------------
    def _parse_tokens(s: str) -> list[int]:
        return [int(t) for t in s.split(",") if t.strip().isdigit()]

    # choose indices to PLOT (CSV is always all)
    to_plot: list[int] = []
    if args.plot_indices.strip():
        tokens = _parse_tokens(args.plot_indices)
        if args.select_by == "index":
            to_plot = tokens
        else:
            for wid in tokens:
                if wid in id_to_idx:
                    to_plot.append(id_to_idx[wid])
    elif args.plot_all:
        to_plot = list(range(len(series_test_scaled)))
    else:
        to_plot = list(range(min(12, len(series_test_scaled))))  # default cap

    # resolve well_id from series_idx using unscaled TEST series (same order)
    series_test_unscaled = list(TimeSeries.from_group_dataframe(
        df=df_test[[TIME_COL, GROUP_COL, TARGET] + STATIC_CATS + STATIC_REALS].copy(),
        time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=TARGET, static_cols=STATIC_CATS + STATIC_REALS, freq=FREQ,
    ))
    def _well_id_for_series_idx(idx: int) -> int:
        try:
            return int(series_test_unscaled[idx].static_covariates[GROUP_COL].values[0])
        except Exception:
            return int(series_test_unscaled[idx].static_covariates["well_id"].values[0])

    # ---- Case B export (truth available) ----
    if do_B and results_B:
        # CSVs → ALWAYS for ALL results
        if args.export_csv:
            for r in results_B:
                sid  = r["series_idx"]
                wid  = _well_id_for_series_idx(sid)
                pred = r["pred"]
                truth= r["truth"]

                df_pred  = _ts_to_df(pred.quantile(0.10), "p10").join(
                            _ts_to_df(pred.quantile(0.50), "p50"), how="outer"
                          ).join(
                            _ts_to_df(pred.quantile(0.90), "p90"), how="outer"
                          )
                df_truth = _ts_to_df(truth, TARGET)
                df_out   = df_pred.join(df_truth, how="outer").reset_index().rename(columns={"index": TIME_COL})

                # out_csv = fixed_preds_dir / f"pred_well_{wid}_caseB.csv"
                out_csv = run_dir / f"pred_well_{wid}_caseB.csv"
                ensure_dir(out_csv.parent)
                df_out.to_csv(out_csv, index=False)

        # PNGs → subset
        if args.save_plots:
            selected = set(to_plot) if to_plot else set()
            for r in results_B:
                sid = r["series_idx"]
                if selected and sid not in selected:
                    continue
                wid = _well_id_for_series_idx(sid)
                title = f"Well {wid} – Forecast (caseB)"
                fig = _plot_prediction_vs_truth(r, title=title)
                # out_png = fixed_preds_dir / f"caseB_series_{sid:03d}.png"
                out_png = run_dir / f"caseB_series_{sid:03d}.png"
                _save_plot(fig, out_png)

    # ---- Case C export (no truth) ----
    if do_C and results_C:
        if args.export_csv:
            for r in results_C:
                sid  = r["series_idx"]
                wid  = _well_id_for_series_idx(sid)
                pred = r["pred"]

                df_pred = _ts_to_df(pred.quantile(0.10), "p10").join(
                            _ts_to_df(pred.quantile(0.50), "p50"), how="outer"
                          ).join(
                            _ts_to_df(pred.quantile(0.90), "p90"), how="outer"
                          ).reset_index().rename(columns={"index": TIME_COL})

                # out_csv = fixed_preds_dir / f"pred_well_{wid}.csv"
                out_csv = run_dir / f"pred_well_{wid}.csv"
                ensure_dir(out_csv.parent)
                df_pred.to_csv(out_csv, index=False)

        if args.save_plots:
            selected = set(to_plot) if to_plot else set()
            for r in results_C:
                sid = r["series_idx"]
                if selected and sid not in selected:
                    continue
                wid = _well_id_for_series_idx(sid)
                title = f"Well {wid} – Forecast (caseC)"
                fig = _plot_prediction_future(r, title=title)
                # out_png = fixed_preds_dir / f"caseC_series_{sid:03d}.png"
                out_png = run_dir / f"caseC_series_{sid:03d}.png"
                _save_plot(fig, out_png)

    # manifest
    write_json(run_dir / "predict_config.json", {
        "model_path": str(model_path),
        "warm_months": args.warm_months,
        "case": args.case,
        "total_months": args.total_months,
        "samples": args.samples,
        "use_log1p": use_log1p,
        "log_future": args.log_future,
        "plot_all": args.plot_all,
    })

    print(f"\n✅ Done. Run outputs in: {run_dir}")
    print(f"Fixed predictions dir: {artifact_path('predictions') / 'plots'}")


if __name__ == "__main__":
    main()
