# src/vm_tft/cli/fit_cv.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer, InvertibleMapper
from darts.dataprocessing.pipeline import Pipeline as DartsPipeline

from vm_tft.cfg import (
    ALL_FILE, FREQ, TIME_COL, GROUP_COL, TARGET,
    STATIC_CATS, STATIC_REALS,
    PAST_REALS, FUTURE_REALS,
    INPUT_CHUNK, FORECAST_H, N_WINDOWS,
)
from vm_tft.io_utils import (
    set_seeds, ensure_dir, artifact_path,
    new_run_dir, write_json, copy_as_pointer, update_best_if_better,
)
from vm_tft.features import build_future_covariates_well_age_full
from vm_tft.features import build_future_covariates_dca_full
from vm_tft.modeling import build_tft, default_add_encoders
from vm_tft.evaluate import metrics_from_cv

# use Tensor Cores with safe precision/perf tradeoff
torch.set_float32_matmul_precision("high")


# -----------------------
# helpers
# -----------------------
def encode_year(idx: pd.DatetimeIndex) -> float:
    return (idx.year - 2000) / 50.0


def _maybe_ts_from_group_df(df: pd.DataFrame, cols: list[str] | None):
    if not cols:
        return None
    return TimeSeries.from_group_dataframe(
        df=df[[TIME_COL, GROUP_COL] + cols].copy(),
        time_col=TIME_COL,
        group_cols=GROUP_COL,
        value_cols=cols,
        freq=FREQ,
    )


def _make_scalers(
    use_log1p: bool,
    log_future: bool,
    has_past: bool,
    has_future: bool,
):
    """
    Build (target, past, future) Darts transformers.
    If use_log1p=True:
      - Target: log1p -> MinMax
      - Past:   log1p -> MinMax (if present)
      - Future: log1p -> MinMax only if log_future=True; else MinMax only
    If use_log1p=False:
      - All: MinMax only (when present).
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


def _lower_is_better(metric_key: str) -> bool:
    # most are minimized; MIC is maximized
    return metric_key.lower() not in {"mic_10_90", "mic", "coverage"}


# -----------------------
# main
# -----------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit baseline TFT and run CV (with optional log1p scaling)")
    # training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--hidden-continuous-size", type=int, default=8)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--quantiles", type=str, default="0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=100, help="num_samples for probabilistic preds during CV")
    parser.add_argument("--stride", type=int, default=FORECAST_H, help="stride for historical_forecasts")
    # logging / pointers
    parser.add_argument("--run-tag", type=str, default="baseline", help="Suffix tag for the run folder name.")
    parser.add_argument("--only-if-better", action="store_true",
                        help="Only refresh fixed pointers (metrics/model) if metric improves.")
    parser.add_argument("--opt-metric", type=str, default="mql_0_50",
                        choices=["mql_0_50", "rmse", "mae", "smape", "mic_10_90"])
    # transforms
    parser.add_argument("--no-log1p", action="store_true",
                        help="Disable log1p transform (default: enabled).")
    parser.add_argument("--log-future", action="store_true",
                        help="Also apply log1p to future covariates (default: off). Be careful if futures include negatives.")

    args = parser.parse_args(argv)
    set_seeds(args.seed)

    # ---------- load data ----------
    df = pd.read_csv(ALL_FILE, parse_dates=[TIME_COL])
    print(f"Loaded: {ALL_FILE}  ({df.shape[0]:,} rows)")

    # ---------- target series + statics ----------
    series = TimeSeries.from_group_dataframe(
        df=df[[TIME_COL, GROUP_COL, TARGET] + STATIC_CATS + STATIC_REALS].copy(),
        time_col=TIME_COL,
        group_cols=GROUP_COL,
        value_cols=TARGET,
        static_cols=STATIC_CATS + STATIC_REALS,
        freq=FREQ,
    )

    # static encoders (numeric + categorical)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())])
    cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    static_tr = StaticCovariatesTransformer(
        transformer_num=num_pipe,
        transformer_cat=cat_enc,
        cols_num=STATIC_REALS,
        cols_cat=STATIC_CATS,
    )
    series = static_tr.fit_transform(series)
    categorical_embedding_sizes = {col: int(df[col].nunique(dropna=True)) for col in STATIC_CATS}

    # ---------- covariates ----------
    past_covariates = _maybe_ts_from_group_df(df, PAST_REALS)
    
    # fut_age = build_future_covariates_well_age_full(
    #     df=df, time_col=TIME_COL, group_col=GROUP_COL,
    #     completion_col="completion_date", 
    #     horizon=FORECAST_H, freq=FREQ
    # )
    
    # fut_dca, dca_pars = build_future_covariates_dca_full(
    #     df=df, time_col=TIME_COL, group_col=GROUP_COL,
    #     target_col=TARGET, horizon=FORECAST_H, freq=FREQ,
    #     fit_k=INPUT_CHUNK, params_out=True
    # )
    
    # # after building fut_dca with 'dca_bpd'
    # fut_dca["dca_bpd_log1p"] = np.log1p(fut_dca["dca_bpd"])
    
    # # merge the FUTURE covariates on [group, time]
    # fut_df = fut_age.merge(fut_dca[[GROUP_COL, TIME_COL, "dca_bpd_log1p"]], on=[GROUP_COL, TIME_COL], how="left")

    fut_df = build_future_covariates_well_age_full(
        df=df, time_col=TIME_COL, group_col=GROUP_COL,
        completion_col="completion_date",
        horizon=FORECAST_H, freq=FREQ,
    )
    future_covariates = TimeSeries.from_group_dataframe(
        df=fut_df, time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=FUTURE_REALS, freq=FREQ,
    )

    # ---------- scalers (log1p+scale OR scale only) ----------
    use_log1p = not args.no_log1p
    y_scaler, x_scaler_past, x_scaler_future = _make_scalers(
        use_log1p=use_log1p,
        log_future=args.log_future,
        has_past=(past_covariates is not None),
        has_future=(future_covariates is not None),
    )

    series_scaled = y_scaler.fit_transform(series)
    past_covs_scaled = x_scaler_past.fit_transform(past_covariates) if x_scaler_past else None
    future_covs_scaled = x_scaler_future.fit_transform(future_covariates) if x_scaler_future else None

    # ---------- model ----------
    add_encoders = default_add_encoders(encode_year)
    quantiles = [float(q) for q in args.quantiles.split(",")]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [0] if accelerator == "gpu" else []

    model = build_tft(
        categorical_embedding_sizes=categorical_embedding_sizes,
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=FORECAST_H,
        add_encoders=add_encoders,
        lr=args.lr,
        hidden_size=args.hidden_size,
        hidden_continuous_size=args.hidden_continuous_size,
        lstm_layers=args.lstm_layers,
        n_heads=args.heads,
        dropout=args.dropout,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        quantiles=quantiles,
        seed=args.seed,
        accelerator=accelerator,
        devices=devices,
    )

    # ---------- fit ----------
    print("Fitting TFT on full training set …")
    model.fit(
        series=series_scaled,
        past_covariates=past_covs_scaled,
        future_covariates=future_covs_scaled,
        verbose=True,
    )
    print("Fit complete.")

    # ---------- rolling CV ----------
    print("Running rolling CV (historical_forecasts) …")
    cv = model.historical_forecasts(
        series=series_scaled,
        past_covariates=past_covs_scaled,
        future_covariates=future_covs_scaled,
        start=-(N_WINDOWS * FORECAST_H),
        start_format="position",
        forecast_horizon=FORECAST_H,
        stride=args.stride,
        retrain=True,
        last_points_only=False,
        num_samples=args.samples,
        verbose=True,
    )

    # ---------- metrics ----------
    m = metrics_from_cv(cv, series_scaled, y_scaler, compute_global=True)
    per_sw = m["per_series_window"]
    overall = m["overall"]

    # ---------- save: run dir + fixed pointers ----------
    run_dir = new_run_dir(kind="cv", tag=args.run_tag)
    ensure_dir(run_dir)

    per_sw_path = run_dir / "metrics_per_series_window.csv"
    overall_path = run_dir / "metrics_overall.json"
    model_path = run_dir / "model.pt"
    cfg_path = run_dir / "fit_cv_config.json"

    per_sw.to_csv(per_sw_path, index=False)
    overall.to_json(overall_path, indent=2)
    model.save(str(model_path))

    # record run config
    write_json(cfg_path, {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "hidden_size": args.hidden_size, "hidden_continuous_size": args.hidden_continuous_size,
        "lstm_layers": args.lstm_layers,
        "heads": args.heads, "dropout": args.dropout, "lr": args.lr,
        "quantiles": quantiles, "seed": args.seed,
        "input_chunk_length": INPUT_CHUNK, "output_chunk_length": FORECAST_H,
        "n_windows": N_WINDOWS, "stride": args.stride, "samples": args.samples,
        "accelerator": accelerator, "use_log1p": use_log1p, "log_future": args.log_future,
    })

    # update fixed pointers (optionally only-if-better)
    fixed_metrics_csv  = artifact_path("metrics") / "cv_metrics_per_series_window.csv"
    fixed_overall_json = artifact_path("metrics") / "cv_overall.json"
    fixed_model        = artifact_path("models")  / "base_model.pt"

    ensure_dir(fixed_metrics_csv.parent); ensure_dir(fixed_model.parent)

    if args.only_if_better:
        # Compare chosen metric; fall back to lower-is-better except for MIC.
        opt_key = args.opt_metric
        cand_val = float(overall.get(opt_key, np.inf if _lower_is_better(opt_key) else -np.inf))

        best_json_path = artifact_path("metrics") / f"cv_best_{opt_key}.json"
        payload = {
            "metric_name": opt_key,
            "run_dir": str(run_dir),
            "overall_json": str(overall_path),
            "metrics_csv": str(per_sw_path),
            "model_path": str(model_path),
        }
        improved = update_best_if_better(
            candidate_metric=cand_val,
            best_json_path=best_json_path,
            candidate_payload=payload,
            lower_is_better=_lower_is_better(opt_key),
        )
        if improved:
            copy_as_pointer(per_sw_path,  fixed_metrics_csv)
            copy_as_pointer(overall_path, fixed_overall_json)
            copy_as_pointer(model_path,   fixed_model)
            print(f"↑ Improved {opt_key}; fixed pointers refreshed.")
        else:
            print(f"↳ Not improved on {opt_key}; fixed pointers left unchanged.")
    else:
        copy_as_pointer(per_sw_path,  fixed_metrics_csv)
        copy_as_pointer(overall_path, fixed_overall_json)
        copy_as_pointer(model_path,   fixed_model)

    print("\n✅ CV finished.")
    print(f"- Run dir                → {run_dir}")
    print(f"- Per-series/window CSV  → {per_sw_path}")
    print(f"- Overall JSON           → {overall_path}")
    print(f"- Model                  → {model_path}")
    print(f"- Fixed pointers updated → {fixed_overall_json}, {fixed_metrics_csv}, {fixed_model}")


if __name__ == "__main__":
    main()
