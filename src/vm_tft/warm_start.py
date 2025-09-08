# Warm-start forecasting & metrics
import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset
import matplotlib.pyplot as plt
from darts.metrics import mape, rmse, mae, smape
from darts.metrics import mql, mic
from darts.timeseries import TimeSeries
from typing import List, Optional

# ---- helpers (same shapes as your CV code) ----

# Helper: check_future_cov_coverage
def check_future_cov_coverage(
    test_series_scaled: "list[TimeSeries]",
    test_future_covs_scaled: "list[TimeSeries]",
    warm_months: int,
    input_chunk_length: int,
    horizon: int,
    freq: str = "MS",
) -> None:
    """
    For each well, verify that future_covariates start early enough to cover the encoder window
    and extend through the decoder horizon.
    """
    step = to_offset(freq)
    n_series = len(test_series_scaled)
    problems = 0

    for i in range(n_series):
        ts = test_series_scaled[i]
        fc = test_future_covs_scaled[i]

        # encoder end is last time of warm slice; encoder start depends on input_chunk_length
        need_hist = max(warm_months, input_chunk_length)
        enc_end = ts.end_time()
        enc_start = enc_end - (need_hist - 1) * step

        dec_start = enc_end + step
        dec_end   = enc_end + horizon * step

        ok_past  = fc.start_time() <= enc_start
        ok_fut   = fc.end_time()   >= dec_end

        if not (ok_past and ok_fut):
            problems += 1
            print(f"[{i}] FUTURE_COV start={fc.start_time()} end={fc.end_time()} | "
                  f"required start≤{enc_start}, end≥{dec_end}")

    if problems == 0:
        print("✅ All future_covariates cover encoder past and decoder future.")
    else:
        print(f"⚠️ {problems} series with insufficient future_covariates coverage.")

def _get_step(ts: TimeSeries):
    if getattr(ts, "freq", None) is not None:
        return ts.freq
    freq_str = getattr(ts, "freq_str", None) or pd.infer_freq(ts.time_index)
    return to_offset(freq_str)

def _slice_warm_start(ts: TimeSeries, warm_months: int, input_chunk_length: int) -> TimeSeries:
    need = max(input_chunk_length, warm_months)
    return ts if len(ts) <= need else ts[-need:]

def _align_future_covs(
    warm_series: TimeSeries,
    future_cov: TimeSeries | None,
    horizon: int,
    input_chunk_length: int = 24,
    warm_months: int = 24,
):
    """
    Make future_cov cover [encoder_start .. decoder_end].
    encoder_start = enc_end - (max(input_chunk_length, warm_months) - 1) * step
    decoder_end   = enc_end + horizon * step
    """
    if future_cov is None:
        return None

    step = _get_step(warm_series)
    enc_end = warm_series.end_time()
    need_hist = max(input_chunk_length, warm_months)
    enc_start = enc_end - (need_hist - 1) * step
    dec_end   = enc_end + horizon * step

    fc = future_cov.slice(enc_start, dec_end)
    return fc if len(fc) > 0 else future_cov


def _ts_to_df(ts: TimeSeries, name: str) -> pd.DataFrame:
    vals = ts.values(copy=False)
    if vals.ndim == 3:  # (time, comp, sample)
        vals = vals[:, 0, 0]
    elif vals.ndim == 2:
        vals = vals[:, 0]
    return pd.DataFrame({name: vals}, index=ts.time_index)

def _safe_mape(y_true, y_pred, eps=1e-1):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)

def _smape(y_true, y_pred, eps=1e-1):
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

# ---- main: metrics for single warm-start/horizon predictions ----


def predict_with_truth(model, series_list: List[TimeSeries], y_scaler,
                       warm_months: int, horizon: Optional[int],
                       past_covs_list=None, future_covs_list=None,
                       num_samples=200, eval_until_end=False):
    results = []
    for i, ts in enumerate(series_list):
        if len(ts) <= warm_months: 
            continue
        warm_ts = ts[:warm_months]
        pc = None if past_covs_list is None else past_covs_list[i]
        fc = None if future_covs_list is None else future_covs_list[i]
        horizon_i = (len(ts) - warm_months) if eval_until_end else horizon
        pred_s = model.predict(n=horizon_i, series=warm_ts,
                               past_covariates=pc, future_covariates=fc,
                               num_samples=num_samples, show_warnings=False,
                               verbose=False)
        pred = y_scaler.inverse_transform(pred_s)
        truth = y_scaler.inverse_transform(ts[warm_months:warm_months+horizon_i])
        results.append({"series_idx": i, "warm_ts": y_scaler.inverse_transform(warm_ts),
                        "pred": pred, "truth": truth})
    return results

def predict_future(model, series_list: List[TimeSeries], y_scaler,
                   warm_months: int, horizon: int,
                   past_covs_list=None, future_covs_list=None, num_samples=200):
    """Case C: forecast beyond history; no truth."""
    outputs = []
    for i, ts in enumerate(series_list):
        if len(ts) <= warm_months: 
            continue
        warm_ts = ts[:warm_months]
        pc = None if past_covs_list is None else past_covs_list[i]
        fc = None if future_covs_list is None else future_covs_list[i]
        pred_s = model.predict(n=horizon, series=warm_ts, past_covariates=pc,
                               future_covariates=fc, num_samples=num_samples,
                               show_warnings=False, verbose=False)
        pred = y_scaler.inverse_transform(pred_s)
        outputs.append({"series_idx": i, "warm_ts": y_scaler.inverse_transform(warm_ts), "pred": pred})
    return outputs


def metrics_from_results(results, q_interval=(0.10, 0.90)):
    """
    Compute metrics for predictions vs truth.

    Parameters
    ----------
    results : list of dicts
        Output from predict_with_truth()
    q_interval : tuple
        Interval for MIC (default 10%-90%)

    Returns
    -------
    dict with:
        - per_series : DataFrame of metrics per well
        - overall_mean : simple average of metrics
        - overall_weighted : weighted average by #points
    """
    rows = []

    for r in results:
        i     = r["series_idx"]
        pred  = r["pred"]
        truth = r["truth"]

        # Align truth and pred
        pred_df  = _ts_to_df(pred.quantile(0.50), "y_hat")
        truth_df = _ts_to_df(truth, "y_true").reindex(pred_df.index).dropna()
        if truth_df.empty:
            continue
        pred_df = pred_df.loc[truth_df.index]

        y_hat = pred_df["y_hat"].to_numpy()
        y_true = truth_df["y_true"].to_numpy()
        n_pts = len(y_true)

        # deterministic metrics
        rmse   = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))
        mae    = float(np.mean(np.abs(y_hat - y_true)))
        mape_  = _safe_mape(y_true, y_hat)
        smape_ = _smape(y_true, y_hat)

        # probabilistic metrics
        pred_overlap = pred.slice(truth.start_time(), truth.end_time())
        truth_overlap = truth.slice(truth.start_time(), truth.end_time())

        mql_050   = float(mql(truth_overlap, pred_overlap, 0.50))
        mic_10_90 = float(mic(truth_overlap, pred_overlap, q_interval=q_interval))

        rows.append({
            "series_idx": i,
            "n_points": n_pts,
            "rmse": rmse,
            "mae": mae,
            "mape": mape_,
            "smape": smape_,
            "mql_0_50": mql_050,
            "mic_10_90": mic_10_90,
        })

    per_series = pd.DataFrame(rows).sort_values("series_idx").reset_index(drop=True)

    out = {"per_series": per_series}
    if not per_series.empty:
        # simple mean
        out["overall_mean"] = per_series[["rmse","mae","mape","smape","mql_0_50","mic_10_90"]].mean()

        # weighted mean
        w = per_series["n_points"].to_numpy()
        w = w / w.sum()
        out["overall_weighted"] = pd.Series({
            "rmse_w": np.sum(w * per_series["rmse"].to_numpy()),
            "mae_w": np.sum(w * per_series["mae"].to_numpy()),
            "mape_w": np.sum(w * per_series["mape"].to_numpy()),
            "smape_w": np.sum(w * per_series["smape"].to_numpy()),
            "mql_0_50_w": np.sum(w * per_series["mql_0_50"].to_numpy()),
            "mic_10_90_w": np.sum(w * per_series["mic_10_90"].to_numpy()),
        })
    else:
        out["overall_mean"] = pd.Series(dtype=float)
        out["overall_weighted"] = pd.Series(dtype=float)

    return out


# ---- helper: plotting test results ----

def plot_prediction_vs_truth(results, well_idx, quantiles=(0.1, 0.5, 0.9), figsize=(10,6)):
    """
    Plot prediction vs truth for a given well from results.
    """
    res = next((r for r in results if r["series_idx"] == well_idx), None)
    if res is None:
        print(f"⚠️ Well {well_idx} not found in results")
        return

    warm_ts = res["warm_ts"]
    pred    = res["pred"]
    truth   = res["truth"]

    plt.figure(figsize=figsize)

    # Warm start
    warm_ts.plot(label="Warm start (history)", lw=2, c="tab:blue")

    # Truth continuation
    if len(truth) > 0:
        truth.plot(label="Actual (truth)", lw=2, c="tab:green")

    # Predictions
    try:
        if pred.n_samples > 1:  # probabilistic
            low_q, med_q, high_q = quantiles

            # Plot interval
            pred.plot(low_quantile=low_q, high_quantile=high_q,
                      label=f"Pred {int(low_q*100)}–{int(high_q*100)}%", alpha=0.3)

            # Plot median
            pred.quantile(med_q).plot(label=f"Pred median (q{med_q:.2f})", lw=2, c="tab:red")
        else:
            pred.plot(label="Pred (deterministic)", lw=2, c="tab:red")

    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")
        pred.plot(label="Prediction", lw=2, c="tab:red")

    plt.title(f"Well {well_idx} – Forecast vs Truth")
    plt.xlabel("Date")
    plt.ylabel("Oil Rate (bpd)")
    plt.legend()
    plt.grid(True)
    plt.show()






from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
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
from vm_tft.features import build_future_covariates_well_age_full
from vm_tft.io_utils import ensure_dir, artifact_path, set_seeds, new_run_dir, copy_as_pointer, write_json
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
    if use_log1p:
        log_mapper = InvertibleMapper(fn=np.log1p, inverse_fn=np.expm1, name="log1p")
        y_scaler = DartsPipeline([log_mapper, Scaler(MinMaxScaler())])
        x_past   = DartsPipeline([log_mapper, Scaler(MinMaxScaler())]) if has_past else None
        if has_future:
            x_fut = DartsPipeline([log_mapper, Scaler(MinMaxScaler())]) if log_future else Scaler(MinMaxScaler())
        else:
            x_fut = None
    else:
        y_scaler = Scaler(MinMaxScaler())
        x_past   = Scaler(MinMaxScaler()) if has_past else None
        x_fut    = Scaler(MinMaxScaler()) if has_future else None
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
    # ensure at least model encoder length
    need = max(warm_months, input_chunk_length)
    return ts if len(ts) <= need else ts[-need:]


def _ts_to_df(ts: TimeSeries, name: str) -> pd.DataFrame:
    vals = ts.values(copy=False)
    if vals.ndim == 3:
        vals = vals[:, 0, 0]
    elif vals.ndim == 2:
        vals = vals[:, 0]
    return pd.DataFrame({name: vals}, index=ts.time_index)


def predict_with_truth(
    model,
    series_list: List[TimeSeries],
    y_scaler: Scaler,
    warm_months: int,
    horizon: Optional[int],
    past_covs_list: Optional[List[TimeSeries]] = None,
    future_covs_list: Optional[List[TimeSeries]] = None,
    num_samples: int = 200,
    eval_until_end: bool = False,
    input_chunk_length: int = INPUT_CHUNK,
):
    out = []
    for i, ts in enumerate(series_list):
        if len(ts) <= warm_months:
            continue
        warm_ts = _slice_warm(ts, warm_months, input_chunk_length)
        pc = None if past_covs_list is None else past_covs_list[i]
        fc = None if future_covs_list is None else future_covs_list[i]
        if eval_until_end:
            horizon_i = len(ts) - len(warm_ts)
        else:
            if horizon is None:
                raise ValueError("horizon must be provided when eval_until_end=False")
            horizon_i = int(horizon)

        pred_scaled = model.predict(
            n=horizon_i,
            series=warm_ts,
            past_covariates=pc,
            future_covariates=fc,
            num_samples=num_samples,
            show_warnings=False,
            verbose=False,
        )
        pred  = y_scaler.inverse_transform(pred_scaled)
        truth = y_scaler.inverse_transform(ts[len(warm_ts) : len(warm_ts) + horizon_i])
        out.append({
            "series_idx": i,
            "warm_ts": y_scaler.inverse_transform(warm_ts),
            "pred": pred,
            "truth": truth,
        })
    return out


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


# -----------------------------
# main
# -----------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Warm-start evaluation on TEST set (Case A/B)")
    parser.add_argument("--model-path", type=str, default=str(artifact_path("models") / "tft_best_from_search.pt"))
    parser.add_argument("--warm-months", type=int, default=36, help="months of initial history provided")
    parser.add_argument("--horizon", type=int, default=6, help="Case A horizon (ignored in Case B)")
    parser.add_argument("--case", type=str, choices=["A", "B", "both"], default="both",
                        help="A=fixed horizon; B=until end of history")
    parser.add_argument("--samples", type=int, default=200, help="num_samples for predict()")
    parser.add_argument("--matmul-precision", type=str, choices=["highest","high","medium"], default="high")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-plots", action="store_true", help="save per-well plots")
    parser.add_argument("--export-csv", action="store_true", help="export per-well CSVs for Streamlit")
    parser.add_argument("--select-by", type=str, choices=["index","well"], default="well",
                        help="interpret --plot-indices as series indices or well IDs")
    parser.add_argument("--plot-indices", type=str, default="",
                        help="comma-separated indices or well IDs to plot (e.g. 0,5,18 or 161720,161952)")
    parser.add_argument("--run-tag", type=str, default="test", help="Suffix tag for the run folder name.")
    # transforms
    parser.add_argument("--no-log1p", action="store_true", help="Disable log1p (default: enabled).")
    parser.add_argument("--log-future", action="store_true",
                        help="Also apply log1p to future covariates (default: off).")

    args = parser.parse_args(argv)
    torch.set_float32_matmul_precision(args.matmul_precision)
    set_seeds(args.seed)

    # -------- load data --------
    df_all  = pd.read_csv(ALL_FILE,  parse_dates=[TIME_COL])
    df_test = pd.read_csv(TEST_FILE, parse_dates=[TIME_COL])

    # statics transformer (fit on ALL, apply to TEST)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())])
    cat_enc  = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    static_tr = StaticCovariatesTransformer(
        transformer_num=num_pipe, transformer_cat=cat_enc,
        cols_num=STATIC_REALS, cols_cat=STATIC_CATS,
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

    # covariates
    past_covs_test = _maybe_ts_from_group_df(df_test, PAST_REALS)
    fut_df_test = build_future_covariates_well_age_full(
        df=df_test, time_col=TIME_COL, group_col=GROUP_COL,
        completion_col="completion_date", horizon=max(args.horizon, FORECAST_H), freq=FREQ,
    )
    future_covs_test = TimeSeries.from_group_dataframe(
        df=fut_df_test, time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=FUTURE_REALS, freq=FREQ,
    )

    # fit scalers on ALL, apply to TEST
    use_log1p = not args.no_log1p
    # build future over ALL for stable scaling
    fut_df_all = build_future_covariates_well_age_full(
        df=df_all, time_col=TIME_COL, group_col=GROUP_COL,
        completion_col="completion_date", horizon=FORECAST_H, freq=FREQ,
    )
    future_covs_all = TimeSeries.from_group_dataframe(
        df=fut_df_all, time_col=TIME_COL, group_cols=GROUP_COL, value_cols=FUTURE_REALS, freq=FREQ,
    )

    y_scaler, x_scaler_past, x_scaler_future = _make_scalers(
        use_log1p=use_log1p, log_future=args.log_future,
        has_past=(past_covs_test is not None), has_future=(future_covs_test is not None),
    )
    y_scaler.fit(series_all)
    series_test_scaled = y_scaler.transform(series_test)

    if x_scaler_past and past_covs_test is not None:
        past_all = _maybe_ts_from_group_df(df_all, PAST_REALS)
        x_scaler_past.fit(past_all)
        past_covs_test_scaled = x_scaler_past.transform(past_covs_test)
    else:
        past_covs_test_scaled = None

    if x_scaler_future and future_covs_test is not None:
        x_scaler_future.fit(future_covs_all)
        future_covs_test_scaled = x_scaler_future.transform(future_covs_test)
    else:
        future_covs_test_scaled = None

    # -------- load model --------
    from darts.models import TFTModel
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = TFTModel.load(str(model_path))

    # -------- run predictions --------
    run_dir = new_run_dir(kind="test", tag=args.run_tag)
    ensure_dir(run_dir)
    fixed_preds_dir = artifact_path("predictions") / "plots"
    ensure_dir(fixed_preds_dir)

    do_A = (args.case in ("A","both"))
    do_B = (args.case in ("B","both"))

    # map well_id(s) to series_idx if plotting/exporting specific wells by well id
    well_ids = df_test[GROUP_COL].dropna().unique().tolist()
    id_to_idx = {wid: i for i, wid in enumerate(sorted(well_ids))}

    results_A = results_B = None
    if do_A:
        results_A = predict_with_truth(
            model=model,
            series_list=list(series_test_scaled),
            y_scaler=y_scaler,
            warm_months=args.warm_months,
            horizon=args.horizon,
            past_covs_list=None if past_covs_test_scaled is None else list(past_covs_test_scaled),
            future_covs_list=list(future_covs_test_scaled),
            num_samples=args.samples,
            eval_until_end=False,
            input_chunk_length=INPUT_CHUNK,
        )
        metrics_A = metrics_from_results(results_A)
        (run_dir / "caseA_overall_mean.json").write_text(metrics_A["overall_mean"].to_json(indent=2))
        metrics_A["per_series"].to_csv(run_dir / "caseA_per_series.csv", index=False)
        # fixed pointer
        copy_as_pointer(run_dir / "caseA_overall_mean.json", artifact_path("predictions") / "caseA_overall_mean.json")

    if do_B:
        results_B = predict_with_truth(
            model=model,
            series_list=list(series_test_scaled),
            y_scaler=y_scaler,
            warm_months=args.warm_months,
            horizon=None,
            past_covs_list=None if past_covs_test_scaled is None else list(past_covs_test_scaled),
            future_covs_list=list(future_covs_test_scaled),
            num_samples=args.samples,
            eval_until_end=True,
            input_chunk_length=INPUT_CHUNK,
        )
        metrics_B = metrics_from_results(results_B)
        (run_dir / "caseB_overall_mean.json").write_text(metrics_B["overall_mean"].to_json(indent=2))
        metrics_B["per_series"].to_csv(run_dir / "caseB_per_series.csv", index=False)
        # fixed pointer
        copy_as_pointer(run_dir / "caseB_overall_mean.json", artifact_path("predictions") / "caseB_overall_mean.json")

    # -------- export per-well CSVs & plots (for Streamlit) --------
    if args.save_plots or args.export_csv:
        to_plot = []
        if args.plot_indices.strip():
            tokens = [t.strip() for t in args.plot_indices.split(",") if t.strip()]
            if args.select_by == "index":
                to_plot = [int(t) for t in tokens if t.isdigit()]
            else:
                # well ids → indices
                for t in tokens:
                    try:
                        wid = int(t)
                        if wid in id_to_idx:
                            to_plot.append(id_to_idx[wid])
                    except Exception:
                        pass
        else:
            # nothing specified; plot/export first few for convenience
            to_plot = list(range(min(12, len(series_test_scaled))))

        def _export_case(results, case_tag: str):
            if not results:
                return
            for r in results:
                i = r["series_idx"]
                if i not in to_plot:
                    continue
                wid = sorted(well_ids)[i] if i < len(well_ids) else i

                # CSV for app
                if args.export_csv:
                    pred = r["pred"]
                    truth = r["truth"]
                    # build frame with p10,p50,p90 and truth
                    df_pred = _ts_to_df(pred.quantile(0.10), "p10").join(
                        _ts_to_df(pred.quantile(0.50), "p50"), how="outer"
                    ).join(
                        _ts_to_df(pred.quantile(0.90), "p90"), how="outer"
                    )
                    df_truth = _ts_to_df(truth, TARGET)
                    df_out = df_pred.join(df_truth, how="outer").reset_index().rename(columns={"index": TIME_COL})
                    out_csv = fixed_preds_dir / f"pred_well_{wid}.csv"
                    ensure_dir(out_csv.parent)
                    df_out.to_csv(out_csv, index=False)

                # PNG plot
                if args.save_plots:
                    title = f"Well {wid} – Forecast ({case_tag})"
                    fig = _plot_prediction_vs_truth(r, title=title)
                    out_png = fixed_preds_dir / f"{case_tag}_series_{i:03d}.png"
                    _save_plot(fig, out_png)
        if do_A: _export_case(results_A, "caseA")
        if do_B: _export_case(results_B, "caseB")

    # dump simple manifest for the app if needed
    write_json(run_dir / "predict_config.json", {
        "model_path": str(model_path),
        "warm_months": args.warm_months,
        "horizon": args.horizon,
        "case": args.case,
        "samples": args.samples,
        "use_log1p": use_log1p,
        "log_future": args.log_future,
    })

    print(f"\n✅ Done. Run outputs in: {run_dir}")
    print(f"Fixed predictions dir: {fixed_preds_dir}")


if __name__ == "__main__":
    main()


###### OLD VERSION WORKING

# src/vm_tft/cli/test_predict.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer

from vm_tft.cfg import (
    ALL_FILE, TEST_FILE, FREQ,
    TIME_COL, GROUP_COL, TARGET,
    STATIC_CATS, STATIC_REALS,
    PAST_REALS, FUTURE_REALS,
    INPUT_CHUNK, FORECAST_H,
)
from vm_tft.features import build_future_covariates_well_age_full
from vm_tft.io_utils import ensure_dir, artifact_path, set_seeds
from vm_tft.evaluate import metrics_from_results  # expects the "empty-safe" version


# -----------------------------
# helpers local to this script
# -----------------------------

def _series_well_id(s: TimeSeries) -> int:
    try:
        return int(s.static_covariates[GROUP_COL].values[0])
    except Exception:
        return int(s.static_covariates["well_id"].values[0])

def _pred_dict_to_csv_row(pred_ts: TimeSeries, truth_ts: TimeSeries) -> pd.DataFrame:
    """Build a DF with date, y_true, p10, p50, p90 (aligned to pred dates)."""
    # align truth to pred dates
    truth_al = truth_ts.slice(pred_ts.start_time(), pred_ts.end_time())
    # extract quantiles (works for deterministic too: p10==p50==p90)
    p10 = pred_ts.quantile(0.10)
    p50 = pred_ts.quantile(0.50)
    p90 = pred_ts.quantile(0.90)
    df = pd.DataFrame({
        TIME_COL: pred_ts.time_index,
        "y_true": truth_al.values(copy=False).squeeze(),
        "p10":    p10.values(copy=False).squeeze(),
        "p50":    p50.values(copy=False).squeeze(),
        "p90":    p90.values(copy=False).squeeze(),
    })
    return df

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
    """
    For a warm-start evaluation that mimics 'early production', we want the FIRST
    `max(warm_months, input_chunk_length)` months from the start of the well.
    """
    need = max(warm_months, input_chunk_length)
    return ts if len(ts) <= need else ts[:need]

def _ts_to_df(ts: TimeSeries, name: str) -> pd.DataFrame:
    vals = ts.values(copy=False)
    if vals.ndim == 3:
        vals = vals[:, 0, 0]
    elif vals.ndim == 2:
        vals = vals[:, 0]
    return pd.DataFrame({name: vals}, index=ts.time_index)

def build_wellid_to_index(series_list: list[TimeSeries]) -> dict[int, int]:
    """
    Map well_id (=group_id in your statics) -> series_idx
    """
    mapping: dict[int, int] = {}
    for i, s in enumerate(series_list):
        try:
            gid = int(s.static_covariates[GROUP_COL].values[0])
        except Exception:
            # fallback if the statics column name differs
            gid = int(s.static_covariates["well_id"].values[0])
        mapping[gid] = i
    return mapping

def predict_with_truth(
    model,
    series_list: List[TimeSeries],
    y_scaler: Scaler,
    warm_months: int,
    horizon: Optional[int],
    past_covs_list: Optional[List[TimeSeries]] = None,
    future_covs_list: Optional[List[TimeSeries]] = None,
    num_samples: int = 200,
    eval_until_end: bool = False,
    input_chunk_length: int = INPUT_CHUNK,
):
    """
    Warm-start predictions on unseen wells.

    Returns list of dicts: {series_idx, warm_ts, pred, truth}
    """
    out = []
    for i, ts in enumerate(series_list):
        if len(ts) <= warm_months:
            continue

        # FIRST months (not the last ones)
        warm_ts = _slice_warm(ts, warm_months, input_chunk_length)

        pc = None if past_covs_list is None else past_covs_list[i]
        fc = None if future_covs_list is None else future_covs_list[i]

        if eval_until_end:
            horizon_i = len(ts) - len(warm_ts)
        else:
            if horizon is None:
                raise ValueError("horizon must be provided when eval_until_end=False")
            horizon_i = int(horizon)

        pred_scaled = model.predict(
            n=horizon_i,
            series=warm_ts,
            past_covariates=pc,
            future_covariates=fc,
            num_samples=num_samples,
            show_warnings=False,
            verbose=False,
        )

        pred  = y_scaler.inverse_transform(pred_scaled)
        # truth is the next horizon_i months *after* warm_ts
        truth = y_scaler.inverse_transform(ts[len(warm_ts): len(warm_ts) + horizon_i])

        out.append({
            "series_idx": i,
            "warm_ts": y_scaler.inverse_transform(warm_ts),
            "pred": pred,
            "truth": truth,
        })
    return out

def plot_prediction_vs_truth(
    results,
    series_idx: int,
    quantiles=(0.1, 0.5, 0.9),
    figsize=(10, 6),
    title_suffix="",
    show: bool = False,
):
    """
    Plot prediction vs truth for a given well from results.
    Returns: (fig, ax)
    """
    res = next((r for r in results if r["series_idx"] == series_idx), None)
    if res is None:
        print(f"⚠️ series_idx={series_idx} not found in results")
        return None, None

    warm_ts, pred, truth = res["warm_ts"], res["pred"], res["truth"]

    fig, ax = plt.subplots(figsize=figsize)
    warm_ts.plot(label="Warm start (history)", lw=2, ax=ax)
    if len(truth) > 0:
        truth.plot(label="Actual (truth)", lw=2, ax=ax)

    try:
        if getattr(pred, "n_samples", 1) > 1:
            low_q, med_q, high_q = quantiles
            pred.plot(low_quantile=low_q, high_quantile=high_q,
                      label=f"Pred {int(low_q*100)}–{int(high_q*100)}%", alpha=0.3, ax=ax)
            pred.quantile(med_q).plot(label=f"Pred median (q{med_q:.2f})", lw=2, ax=ax)
        else:
            pred.plot(label="Pred (det)", lw=2, ax=ax)
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")
        pred.plot(label="Pred", lw=2, ax=ax)

    ax.set_title(f"Series {series_idx} – Forecast vs Truth {title_suffix}")
    ax.set_xlabel("Date"); ax.set_ylabel(TARGET)
    ax.grid(True); ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax

def _parse_csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip().isdigit()]


# -----------------------------
# main
# -----------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Warm-start evaluation on TEST set (Case A/B)")
    parser.add_argument("--model-path", type=str, default=str(artifact_path("models") / "tft_best_from_search.pt"))
    parser.add_argument("--warm-months", type=int, default=36, help="months of initial history provided")
    parser.add_argument("--horizon", type=int, default=6, help="Case A horizon (ignored in Case B)")
    parser.add_argument("--case", type=str, choices=["A", "B", "both"], default="both",
                        help="A=fixed horizon; B=until end of history")
    parser.add_argument("--samples", type=int, default=200, help="num_samples for predict()")
    parser.add_argument("--matmul-precision", type=str, choices=["highest","high","medium"], default="high")
    parser.add_argument("--seed", type=int, default=42)

    # plotting I/O
    parser.add_argument("--save-plots", action="store_true", help="save PNGs to artifacts/predictions/plots")
    parser.add_argument("--plot-indices", type=str, default="",
                        help="comma-separated series indices to plot (e.g. 0,5,18)")
    parser.add_argument("--plot-well-ids", type=str, default="",
                        help="comma-separated WELL IDs to plot (e.g. 159281,160333). Overrides --plot-indices")
    parser.add_argument("--no-show", action="store_true", help="do not open GUI windows (batch mode)")
    parser.add_argument("--export-per-well", action="store_true",
                    help="write per-well CSVs (date,y_true,p10,p50,p90) for Streamlit")

    args = parser.parse_args(argv)
    torch.set_float32_matmul_precision(args.matmul_precision)
    set_seeds(args.seed)

    # Use a non-interactive backend if we're saving or explicitly disabling show
    if args.save_plots or args.no_show:
        matplotlib.use("Agg")
        plt.ioff()

    # ---------------- load data ----------------
    df_all  = pd.read_csv(ALL_FILE,  parse_dates=[TIME_COL])
    df_test = pd.read_csv(TEST_FILE, parse_dates=[TIME_COL])

    # target series with statics (fit static transformer on TRAIN/ALL, apply to TEST)
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
    series_all = static_tr.fit_transform(series_all)  # fit statics on ALL

    series_test = TimeSeries.from_group_dataframe(
        df=df_test[[TIME_COL, GROUP_COL, TARGET] + STATIC_CATS + STATIC_REALS].copy(),
        time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=TARGET, static_cols=STATIC_CATS + STATIC_REALS, freq=FREQ,
    )
    series_test = static_tr.transform(series_test)    # apply to TEST

    # covariates
    past_covs_test = _maybe_ts_from_group_df(df_test, PAST_REALS)

    fut_df_test = build_future_covariates_well_age_full(
        df=df_test,
        time_col=TIME_COL, group_col=GROUP_COL,
        completion_col="completion_date",
        horizon=max(args.horizon, FORECAST_H),  # cover decoder + safety
        freq=FREQ,
    )
    future_covs_test = TimeSeries.from_group_dataframe(
        df=fut_df_test, time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=FUTURE_REALS, freq=FREQ,
    )

    # scalers (fit on ALL, transform TEST to avoid leakage from TEST)
    y_scaler = Scaler()
    x_scaler_past   = Scaler() if past_covs_test is not None else None
    x_scaler_future = Scaler() if future_covs_test is not None else None

    y_scaler.fit(series_all)
    series_test_scaled = y_scaler.transform(series_test)

    if x_scaler_past:
        past_covs_all = _maybe_ts_from_group_df(df_all, PAST_REALS)
        x_scaler_past.fit(past_covs_all)
        past_covs_test_scaled = x_scaler_past.transform(past_covs_test)
    else:
        past_covs_test_scaled = None

    if x_scaler_future:
        fut_df_all = build_future_covariates_well_age_full(
            df=df_all, time_col=TIME_COL, group_col=GROUP_COL,
            completion_col="completion_date", horizon=FORECAST_H, freq=FREQ,
        )
        future_covs_all = TimeSeries.from_group_dataframe(
            df=fut_df_all, time_col=TIME_COL, group_cols=GROUP_COL,
            value_cols=FUTURE_REALS, freq=FREQ,
        )
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
    results_dir = artifact_path("predictions"); ensure_dir(results_dir)
    plots_dir   = results_dir / "plots";        ensure_dir(plots_dir)

    do_A = (args.case in ("A","both"))
    do_B = (args.case in ("B","both"))

    # Precompute mapping for well_id selection
    wellid_to_idx = build_wellid_to_index(list(series_test))

    def pick_plot_indices() -> list[int]:
        if args.plot_well_ids.strip():
            ids = _parse_csv_ints(args.plot_well_ids)
            return [wellid_to_idx[i] for i in ids if i in wellid_to_idx]
        return _parse_csv_ints(args.plot_indices)

    if do_A:
        res_A = predict_with_truth(
            model=model,
            series_list=list(series_test_scaled),
            y_scaler=y_scaler,
            warm_months=args.warm_months,
            horizon=args.horizon,
            past_covs_list=None if past_covs_test_scaled is None else list(past_covs_test_scaled),
            future_covs_list=list(future_covs_test_scaled),
            num_samples=args.samples,
            eval_until_end=False,
            input_chunk_length=INPUT_CHUNK,
        )
        metrics_A = metrics_from_results(res_A)
        # save metrics if available
        (results_dir / "caseA_overall_mean.json").write_text(
            metrics_A["overall_mean"].to_json(indent=2) if not metrics_A["overall_mean"].empty else "{}"
        )
        if not metrics_A["per_series"].empty:
            metrics_A["per_series"].to_csv(results_dir / "caseA_per_series.csv", index=False)
        print("Case A overall mean:\n", metrics_A["overall_mean"])

        if args.export_per_well or args.save_plots:
            # map series_idx -> well_id from *unscaled* test series (same order)
            series_test_unscaled = list(TimeSeries.from_group_dataframe(
                df=df_test[[TIME_COL, GROUP_COL, TARGET] + STATIC_CATS + STATIC_REALS].copy(),
                time_col=TIME_COL, group_cols=GROUP_COL,
                value_cols=TARGET, static_cols=STATIC_CATS + STATIC_REALS, freq=FREQ,
            ))
            for r in res_A:
                sid = r["series_idx"]
                wid = _series_well_id(series_test_unscaled[sid])
                df_out = _pred_dict_to_csv_row(r["pred"], r["truth"])
                out_path = plots_dir / f"pred_well_{wid}.csv"
                df_out.to_csv(out_path, index=False)
        
        
        # plots
        if args.save_plots:
            for i in pick_plot_indices()[:12]:
                fig, ax = plot_prediction_vs_truth(res_A, i, title_suffix="(Case A)", show=not args.no_show)
                if fig is not None:
                    fig.savefig(plots_dir / f"caseA_series_{i:03d}.png", dpi=180, bbox_inches="tight")
                    plt.close(fig)

    if do_B:
        res_B = predict_with_truth(
            model=model,
            series_list=list(series_test_scaled),
            y_scaler=y_scaler,
            warm_months=args.warm_months,
            horizon=None,  # ignored
            past_covs_list=None if past_covs_test_scaled is None else list(past_covs_test_scaled),
            future_covs_list=list(future_covs_test_scaled),
            num_samples=args.samples,
            eval_until_end=True,
            input_chunk_length=INPUT_CHUNK,
        )
        metrics_B = metrics_from_results(res_B)
        (results_dir / "caseB_overall_mean.json").write_text(
            metrics_B["overall_mean"].to_json(indent=2) if not metrics_B["overall_mean"].empty else "{}"
        )
        if not metrics_B["per_series"].empty:
            metrics_B["per_series"].to_csv(results_dir / "caseB_per_series.csv", index=False)
        print("Case B overall mean:\n", metrics_B["overall_mean"])

        if args.export_per_well or args.save_plots:
            series_test_unscaled = list(TimeSeries.from_group_dataframe(
                df=df_test[[TIME_COL, GROUP_COL, TARGET] + STATIC_CATS + STATIC_REALS].copy(),
                time_col=TIME_COL, group_cols=GROUP_COL,
                value_cols=TARGET, static_cols=STATIC_CATS + STATIC_REALS, freq=FREQ,
            ))
            for r in res_B:
                sid = r["series_idx"]
                wid = _series_well_id(series_test_unscaled[sid])
                df_out = _pred_dict_to_csv_row(r["pred"], r["truth"])
                out_path = plots_dir / f"pred_well_{wid}.csv"  # optional different name to keep A and B
                df_out.to_csv(out_path, index=False)
        
        if args.save_plots:
            for i in pick_plot_indices()[:12]:
                fig, ax = plot_prediction_vs_truth(res_B, i, title_suffix="(Case B)", show=not args.no_show)
                if fig is not None:
                    fig.savefig(plots_dir / f"caseB_series_{i:03d}.png", dpi=180, bbox_inches="tight")
                    plt.close(fig)

    print(f"✅ Done. Outputs in: {results_dir}")


if __name__ == "__main__":
    main()


        # truth       = y_scaler.inverse_transform(ts[len(warm_ts): len(warm_ts) + horizon_i])
        
    # param_grid = {
    #     "hidden_size":            [64, 96, 128, 192, 256],
    #     "num_attention_heads":    [1, 2, 4, 8],
    #     "lstm_layers":            [1, 2, 3],
    #     "hidden_continuous_size": [8, 16, 32],
    #     "dropout":                [0.05, 0.10, 0.20, 0.30],
    #     "batch_size":             [32, 64, 128],
    #     "n_epochs":               [20, 30, 40],
    #     "lr":                     _logspace_lrs(),
    #     "weight_decay":           [0.0, 1e-6, 1e-5, 1e-4],
    # }
    
