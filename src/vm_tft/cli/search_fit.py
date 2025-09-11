from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import ParameterGrid
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
from darts.utils.callbacks import TFMProgressBar
from darts.utils.likelihood_models import QuantileRegression

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
from vm_tft.modeling import default_add_encoders
from vm_tft.evaluate import metrics_from_cv

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
    return metric_key.lower() not in {"mic_10_90", "mic", "coverage"}


def _logspace_lrs() -> List[float]:
    return [1e-3, 1.5e-3]


def _build_param_list(n_iter: int, seed: int) -> list[dict]:
    param_grid = {
        "hidden_size":            [128],
        "num_attention_heads":    [8],
        "lstm_layers":            [3, 4],
        "hidden_continuous_size": [16],
        "dropout":                [0.05, 0.1],
        "batch_size":             [128],
        "n_epochs":               [20, 30],
        "lr":                     _logspace_lrs(),
        "weight_decay":           [0.0],
    }
    # Build all combos once
    all_params = [p for p in ParameterGrid(param_grid)
                  if p["hidden_size"] % p["num_attention_heads"] == 0]

    rng = np.random.default_rng(seed)
    n_pick = min(n_iter, len(all_params))
    idx = rng.choice(len(all_params), size=n_pick, replace=False)
    return [all_params[i] for i in idx]


# -----------------------
# main
# -----------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Randomized hyperparameter search for TFT + CV selection")
    # search controls
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--stride", type=int, default=FORECAST_H)
    # logging / pointers
    parser.add_argument("--run-tag", type=str, default="search")
    parser.add_argument("--only-if-better", action="store_true")
    parser.add_argument("--opt-metric", type=str, default="mql_0_50",
                        choices=["mql_0_50", "rmse", "mae", "smape", "mic_10_90"])
    # precision / quantiles
    parser.add_argument("--matmul-precision", type=str, choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--quantiles", type=str, default="0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99")
    # transforms
    parser.add_argument("--no-log1p", action="store_true", help="Disable log1p (default: enabled).")
    parser.add_argument("--log-future", action="store_true", help="Also apply log1p to future covariates.")

    args = parser.parse_args(argv)
    torch.set_float32_matmul_precision(args.matmul_precision)
    set_seeds(args.seed)

    # run dir
    run_dir = new_run_dir(kind="search", tag=args.run_tag)
    ensure_dir(run_dir)

    # --------------------------------
    # load processed data
    # --------------------------------
    df = pd.read_csv(ALL_FILE, parse_dates=[TIME_COL])
    print(f"Loaded: {ALL_FILE}  ({df.shape[0]:,} rows)")

    # target series with statics
    series = TimeSeries.from_group_dataframe(
        df=df[[TIME_COL, GROUP_COL, TARGET] + STATIC_CATS + STATIC_REALS].copy(),
        time_col=TIME_COL,
        group_cols=GROUP_COL,
        value_cols=TARGET,
        static_cols=STATIC_CATS + STATIC_REALS,
        freq=FREQ,
    )

    # static transformers
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())])
    cat_enc  = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    static_tr = StaticCovariatesTransformer(
        transformer_num=num_pipe,
        transformer_cat=cat_enc,
        cols_num=STATIC_REALS,
        cols_cat=STATIC_CATS,
    )
    series = static_tr.fit_transform(series)

    # covariates
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

    # scalers (consistent with fit_cv & test_predict)
    use_log1p = not args.no_log1p
    y_scaler, x_scaler_past, x_scaler_future = _make_scalers(
        use_log1p=use_log1p,
        log_future=args.log_future,
        has_past=(past_covariates is not None),
        has_future=(future_covariates is not None),
    )
    series_scaled = y_scaler.fit_transform(series)
    past_covs_scaled   = x_scaler_past.fit_transform(past_covariates) if x_scaler_past else None
    future_covs_scaled = x_scaler_future.fit_transform(future_covariates) if x_scaler_future else None

    categorical_embedding_sizes = {col: int(df[col].nunique(dropna=True)) for col in STATIC_CATS}
    add_encoders = default_add_encoders(encode_year)
    quantiles = [float(q) for q in args.quantiles.split(",")]
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [0] if accelerator == "gpu" else []

    # --------------------------------
    # randomized search
    # --------------------------------
    param_list = _build_param_list(args.n_iter, args.seed)

    trials: List[Dict] = []
    best_trial: Optional[Dict] = None
    opt_key = args.opt_metric
    lower_is_better = _lower_is_better(opt_key)
    best_score = np.inf if lower_is_better else -np.inf

    print(f"Starting randomized search: {len(param_list)} trials")
    for t, p in enumerate(param_list, start=1):
        cfg = {
            "input_chunk_length":     INPUT_CHUNK,
            "output_chunk_length":    FORECAST_H,
            "hidden_size":            int(p["hidden_size"]),
            "num_attention_heads":    int(p["num_attention_heads"]),
            "lstm_layers":            int(p["lstm_layers"]),
            "hidden_continuous_size": int(p["hidden_continuous_size"]),
            "dropout":                float(p["dropout"]),
            "batch_size":             int(p["batch_size"]),
            "n_epochs":               int(p["n_epochs"]),
            "lr":                     float(p["lr"]),
            "weight_decay":           float(p["weight_decay"]),
        }

        print(f"\n[{t}/{len(param_list)}] Trial config: {cfg}")

        from darts.models import TFTModel
        model = TFTModel(
            input_chunk_length=cfg["input_chunk_length"],
            output_chunk_length=cfg["output_chunk_length"],
            hidden_size=cfg["hidden_size"],
            lstm_layers=cfg["lstm_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            hidden_continuous_size=cfg["hidden_continuous_size"],
            dropout=cfg["dropout"],
            batch_size=cfg["batch_size"],
            n_epochs=cfg["n_epochs"],
            likelihood=QuantileRegression(quantiles=quantiles),
            categorical_embedding_sizes=categorical_embedding_sizes,
            use_static_covariates=True,
            add_encoders=add_encoders,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
            pl_trainer_kwargs={
                "accelerator": accelerator,
                "devices": devices,
                "gradient_clip_val": 1.0,
                "callbacks": [TFMProgressBar()],
            },
            random_state=args.seed,
            show_warnings=False,
        )

        try:
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
                verbose=False,
            )

            m = metrics_from_cv(cv, series_scaled, y_scaler, compute_global=True)
            overall = m["overall"]
            row = {
                **cfg,
                "mql_0_50": float(overall.get("mql_0_50", np.nan)),
                "rmse": float(overall.get("rmse", np.nan)),
                "mae": float(overall.get("mae", np.nan)),
                "smape": float(overall.get("smape", np.nan)),
                "mic_10_90": float(overall.get("mic_10_90", np.nan)),
            }
            trials.append(row)
            print(f" → MQL@0.50={row['mql_0_50']:.4f} | RMSE={row['rmse']:.2f} | sMAPE={row['smape']:.2f}")

            cand = row[opt_key]
            if (lower_is_better and cand < best_score) or ((not lower_is_better) and cand > best_score):
                best_score = cand
                best_trial = {"cfg": cfg, "metrics": row}

        except Exception as e:
            print(f" !! Trial failed: {e}")
            trials.append({**cfg, "error": str(e)})

    # --------------------------------
    # save search results
    # --------------------------------
    trials_df = pd.DataFrame(trials)
    trials_csv = run_dir / "trials.csv"
    trials_df.to_csv(trials_csv, index=False)

    write_json(run_dir / "search_config.json", {
        "n_iter": args.n_iter, "seed": args.seed, "samples": args.samples,
        "stride": args.stride, "quantiles": quantiles,
        "n_windows": N_WINDOWS, "accelerator": accelerator,
        "matmul_precision": args.matmul_precision,
        "use_log1p": use_log1p, "log_future": args.log_future,
    })

    if best_trial is None:
        print("\n⚠️ No successful trials; nothing to refit.")
        print(f"Trials CSV → {trials_csv}")
        return

    best_json = run_dir / "search_best.json"
    Path(best_json).write_text(json.dumps(best_trial, indent=2))

    print(f"\n✅ Search finished. Best {opt_key} = {best_score:.4f}")
    print(f"All trials → {trials_csv}")
    print(f"Best trial → {best_json}")

    # --------------------------------
    # refit best model on full data
    # --------------------------------
    cfg = best_trial["cfg"]
    from darts.models import TFTModel
    model_best = TFTModel(
        input_chunk_length=cfg["input_chunk_length"],
        output_chunk_length=cfg["output_chunk_length"],
        hidden_size=cfg["hidden_size"],
        lstm_layers=cfg["lstm_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        hidden_continuous_size=cfg["hidden_continuous_size"],
        dropout=cfg["dropout"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        likelihood=QuantileRegression(quantiles=quantiles),
        categorical_embedding_sizes={col: int(df[col].nunique(dropna=True)) for col in STATIC_CATS},
        use_static_covariates=True,
        add_encoders=add_encoders,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
        pl_trainer_kwargs={
            "accelerator": accelerator,
            "devices": devices,
            "gradient_clip_val": 1.0,
            "callbacks": [TFMProgressBar()],
        },
        random_state=args.seed,
        show_warnings=False,
    )

    model_best.fit(
        series=series_scaled,
        past_covariates=past_covs_scaled,
        future_covariates=future_covs_scaled,
        verbose=True,
    )

    # Optionally compute CV for the refit (useful to log consistent metrics)
    cv_best = model_best.historical_forecasts(
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
        verbose=False,
    )
    m_best = metrics_from_cv(cv_best, series_scaled, y_scaler, compute_global=True)
    best_overall = m_best["overall"]

    # save best artifacts into run dir
    best_overall_path = run_dir / "best_overall.json"
    best_overall.to_json(best_overall_path, indent=2)
    model_path = run_dir / "model_best.pt"
    model_best.save(str(model_path))

    # fixed pointers
    fixed_trials_csv     = artifact_path("metrics") / "search_trials.csv"
    fixed_best_overall   = artifact_path("metrics") / "search_best_overall.json"
    fixed_best_json      = artifact_path("runs")    / "search_best.json"
    fixed_best_model     = artifact_path("models")  / "tft_best_from_search.pt"

    ensure_dir(fixed_trials_csv.parent); ensure_dir(fixed_best_model.parent)

    # always refresh trials
    copy_as_pointer(trials_csv, fixed_trials_csv)

    # update best pointers, maybe only if better
    cand_val = float(best_overall.get(opt_key, np.inf if _lower_is_better(opt_key) else -np.inf))
    best_gate_json = artifact_path("metrics") / f"search_best_{opt_key}.json"
    payload = {
        "metric_name": opt_key,
        "run_dir": str(run_dir),
        "overall_json": str(best_overall_path),
        "model_path": str(model_path),
    }
    if args.only_if_better:
        improved = update_best_if_better(
            candidate_metric=cand_val,
            best_json_path=best_gate_json,
            candidate_payload=payload,
            lower_is_better=_lower_is_better(opt_key),
        )
        if improved:
            copy_as_pointer(best_overall_path, fixed_best_overall)
            copy_as_pointer(best_json,        fixed_best_json)
            copy_as_pointer(model_path,       fixed_best_model)
            print(f"↑ Improved {opt_key}; best pointers refreshed.")
        else:
            print(f"↳ Not improved on {opt_key}; best pointers left unchanged.")
    else:
        copy_as_pointer(best_overall_path, fixed_best_overall)
        copy_as_pointer(best_json,        fixed_best_json)
        copy_as_pointer(model_path,       fixed_best_model)

    print("\n✅ Refit complete.")
    print(f"- Best overall metrics → {best_overall_path}")
    print(f"- Best model saved     → {model_path}")
    print(f"- Fixed pointers       → {fixed_best_overall}, {fixed_best_json}, {fixed_best_model}")


if __name__ == "__main__":
    main()
