from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
import optuna

# tqdm is optional; we degrade gracefully
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# optional skopt sampler
def _make_sampler(name: str, seed: int):
    if name == "skopt":
        try:
            from optuna.integration import SkoptSampler
        except Exception as e:
            raise ImportError(
                "Sampler 'skopt' requires scikit-optimize. Install with: pip install scikit-optimize"
            ) from e
        return SkoptSampler(seed=seed)
    else:
        return optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer, InvertibleMapper
from darts.dataprocessing.pipeline import Pipeline as DartsPipeline
from darts.utils.callbacks import TFMProgressBar
from darts.utils.likelihood_models import QuantileRegression

from vm_tft.cfg import (
    ALL_FILE, FREQ, TIME_COL, GROUP_COL, TARGET,
    STATIC_CATS, STATIC_REALS, PAST_REALS, FUTURE_REALS,
    INPUT_CHUNK, FORECAST_H, N_WINDOWS,
)
from vm_tft.io_utils import (
    set_seeds, ensure_dir, artifact_path,
    new_run_dir, write_json, copy_as_pointer, update_best_if_better,
)
from vm_tft.features import build_future_covariates_well_age_full, build_future_covariates_dca_full
from vm_tft.modeling import default_add_encoders
from vm_tft.evaluate import metrics_from_cv

torch.set_float32_matmul_precision("high")


# ----------------------- utils -----------------------
def _lower_is_better(metric_key: str) -> bool:
    return metric_key.lower() not in {"mic_10_90", "mic", "coverage"}

def encode_year(idx: pd.DatetimeIndex) -> float:
    return (idx.year - 2000) / 50.0

def _maybe_ts_from_group_df(df: pd.DataFrame, cols: list[str] | None):
    if not cols:
        return None
    return TimeSeries.from_group_dataframe(
        df=df[[TIME_COL, GROUP_COL] + cols].copy(),
        time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=cols, freq=FREQ,
    )

def _make_scalers(use_log1p: bool, log_future: bool, has_past: bool, has_future: bool):
    """
    Target: log1p->MinMax if use_log1p else MinMax
    Past:   same as target when present
    Future: MinMax; add log1p only if log_future=True
    """
    if use_log1p:
        log_mapper = InvertibleMapper(fn=np.log1p, inverse_fn=np.expm1, name="log1p")
        y_scaler = DartsPipeline([log_mapper, Scaler()])
        x_past   = DartsPipeline([log_mapper, Scaler()]) if has_past else None
        x_fut    = (DartsPipeline([log_mapper, Scaler()]) if log_future else Scaler()) if has_future else None
    else:
        y_scaler = Scaler()
        x_past   = Scaler() if has_past else None
        x_fut    = Scaler() if has_future else None
    return y_scaler, x_past, x_fut

def _build_future_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build FUTURE covariates by cfg. If FUTURE_REALS includes 'dca_bpd_log1p',
    we fit DCA and merge; otherwise just well_age.
    """
    fut_age = build_future_covariates_well_age_full(
        df=df, time_col=TIME_COL, group_col=GROUP_COL,
        completion_col="completion_date", horizon=FORECAST_H, freq=FREQ
    )
    if "dca_bpd_log1p" in FUTURE_REALS:
        fut_dca, _ = build_future_covariates_dca_full(
            df=df, time_col=TIME_COL, group_col=GROUP_COL,
            target_col=TARGET, horizon=FORECAST_H, freq=FREQ,
            fit_k=INPUT_CHUNK, params_out=True
        )
        fut_dca["dca_bpd_log1p"] = np.log1p(fut_dca["dca_bpd"].astype(float))
        fut = fut_age.merge(
            fut_dca[[GROUP_COL, TIME_COL, "dca_bpd_log1p"]],
            on=[GROUP_COL, TIME_COL], how="left"
        )
    else:
        fut = fut_age
    return fut


def _suggest_params(trial: optuna.trial.Trial) -> dict:
    heads = trial.suggest_categorical("num_attention_heads", [4, 8])
    hidden_size = trial.suggest_categorical(
        "hidden_size", [h for h in [64, 96, 128] if h % heads == 0]
    )
    return {
        "input_chunk_length":     INPUT_CHUNK,
        "output_chunk_length":    FORECAST_H,
        "hidden_size":            int(hidden_size),
        "num_attention_heads":    int(heads),
        "lstm_layers":            trial.suggest_int("lstm_layers", 1, 3),
        "hidden_continuous_size": trial.suggest_categorical("hidden_continuous_size", [16, 24, 32]),
        "dropout":                trial.suggest_float("dropout", 0.05, 0.25, step=0.05),
        "batch_size":             trial.suggest_categorical("batch_size", [64, 128]),
        "n_epochs":               trial.suggest_int("n_epochs", 30, 50, step=10),
        "lr":                     trial.suggest_float("lr", 5e-4, 3e-3, log=True),
        "weight_decay":           trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    }


# ----------------------- main -----------------------
def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Bayesian Optimization (Optuna) for TFT + CV selection")
    # BO controls
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--timeout", type=int, default=None, help="Seconds budget; overrides n-trials if set.")
    p.add_argument("--sampler", type=str, choices=["tpe","skopt"], default="tpe")
    p.add_argument("--pruner", type=str, choices=["none","median"], default="none")
    # CV controls
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--stride", type=int, default=FORECAST_H)
    # transforms
    p.add_argument("--no-log1p", action="store_true", help="Disable log1p for target/past (default: enabled).")
    p.add_argument("--log-future", action="store_true", help="Also apply log1p to future covariates.")
    p.add_argument("--quantiles", type=str, default="0.1,0.5,0.9")
    # objective metric
    p.add_argument("--opt-metric", type=str, default="mql_0_50",
                   choices=["mql_0_50","rmse","mae","smape","mic_10_90"])
    p.add_argument("--mic-target", type=float, default=None, help="If set (e.g., 0.80), minimize |MIC−target|.")
    # run & reproducibility
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-tag", type=str, default="bo")
    p.add_argument("--only-if-better", action="store_true",
                   help="When updating fixed pointers at the very end, only overwrite if strictly better.")
    p.add_argument("--save-on-improve", action="store_true",
                   help="Refit and save the best model immediately when the objective improves (expensive).")
    p.add_argument("--progress", dest="progress", action="store_true", default=True,
                   help="Show tqdm progress bar (default on if tqdm is installed).")
    p.add_argument("--no-progress", dest="progress", action="store_false")
    p.add_argument("--matmul-precision", type=str, choices=["highest","high","medium"], default="high")
    args = p.parse_args(argv)

    torch.set_float32_matmul_precision(args.matmul_precision)
    set_seeds(args.seed)

    # run dir
    run_dir = new_run_dir(kind="search_bo", tag=args.run_tag); ensure_dir(run_dir)

    # persist config early (so partial runs are traceable)
    write_json(run_dir / "search_config.json", {
        "n_trials": args.n_trials, "timeout": args.timeout, "seed": args.seed,
        "sampler": args.sampler, "pruner": args.pruner, "quantiles": args.quantiles,
        "stride": args.stride, "opt_metric": args.opt_metric, "mic_target": args.mic_target,
        "save_on_improve": args.save_on_improve, "use_log1p": (not args.no_log1p),
        "log_future": args.log_future, "matmul_precision": args.matmul_precision,
    })

    # ------------- load & prep data -------------
    df = pd.read_csv(ALL_FILE, parse_dates=[TIME_COL])
    print(f"Loaded: {ALL_FILE}  ({df.shape[0]:,} rows)")

    series = TimeSeries.from_group_dataframe(
        df=df[[TIME_COL, GROUP_COL, TARGET] + STATIC_CATS + STATIC_REALS].copy(),
        time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=TARGET, static_cols=STATIC_CATS + STATIC_REALS, freq=FREQ,
    )

    # static transformers (fit on ALL)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())])
    cat_enc  = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    static_tr = StaticCovariatesTransformer(
        transformer_num=num_pipe, transformer_cat=cat_enc,
        cols_num=STATIC_REALS, cols_cat=STATIC_CATS,
    )
    series = static_tr.fit_transform(series)

    past_covariates = _maybe_ts_from_group_df(df, PAST_REALS)
    fut_df = _build_future_covariates(df)
    future_covariates = TimeSeries.from_group_dataframe(
        df=fut_df, time_col=TIME_COL, group_cols=GROUP_COL,
        value_cols=FUTURE_REALS, freq=FREQ,
    )

    use_log1p = not args.no_log1p
    y_scaler, x_past_s, x_fut_s = _make_scalers(
        use_log1p=use_log1p, log_future=args.log_future,
        has_past=(past_covariates is not None), has_future=(future_covariates is not None),
    )
    series_scaled = y_scaler.fit_transform(series)
    past_covs_scaled   = x_past_s.fit_transform(past_covariates) if x_past_s else None
    future_covs_scaled = x_fut_s.fit_transform(future_covariates) if x_fut_s else None

    categorical_embedding_sizes = {col: int(df[col].nunique(dropna=True)) for col in STATIC_CATS}
    add_encoders = default_add_encoders(encode_year)
    quantiles = [float(q) for q in args.quantiles.split(",")]
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [0] if accelerator == "gpu" else []

    # ------------- Optuna study -------------
    opt_key = args.opt_metric
    lower_is_better = _lower_is_better(opt_key)
    if opt_key == "mic_10_90" and (args.mic_target is not None):
        direction = "minimize"           # minimize |MIC - target|
    else:
        direction = "minimize" if lower_is_better else "maximize"

    sampler = _make_sampler(args.sampler, args.seed)
    pruner = (optuna.pruners.MedianPruner(n_warmup_steps=3) if args.pruner == "median"
              else optuna.pruners.NopPruner())
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner,
                                study_name=f"tft_bo_{args.run_tag}")

    # ---------- helpers to persist after each trial ----------
    trials_csv = run_dir / "trials.csv"
    best_json  = run_dir / "search_best.json"
    fixed_trials_csv   = artifact_path("metrics") / "search_trials_bo.csv"
    fixed_best_overall = artifact_path("metrics") / "search_best_overall_bo.json"
    fixed_best_json    = artifact_path("runs")    / "search_best_bo.json"
    fixed_best_model   = artifact_path("models")  / "tft_best_from_search_bo.pt"
    ensure_dir(fixed_trials_csv.parent); ensure_dir(fixed_best_model.parent)

    def _dump_trials_csv(study: optuna.Study):
        rows = []
        for tr in study.trials:
            rec = {**tr.params}
            rec["value"] = tr.value
            m = tr.user_attrs.get("metrics", {})
            for k, v in m.items():
                rec[k] = v
            cfg_u = tr.user_attrs.get("cfg", {})
            for k, v in cfg_u.items():
                rec[k] = v
            rows.append(rec)
        pd.DataFrame(rows).to_csv(trials_csv, index=False)
        # refresh pointer for dashboards
        copy_as_pointer(trials_csv, fixed_trials_csv)

    def _write_best_payload(study: optuna.Study):
        if not study.best_trial:
            return None, None
        bt = study.best_trial
        best_cfg = bt.user_attrs.get("cfg", {})
        best_m   = bt.user_attrs.get("metrics", {})
        Path(best_json).write_text(json.dumps({"cfg": best_cfg, "metrics": best_m}, indent=2))
        copy_as_pointer(best_json, fixed_best_json)
        return best_cfg, best_m

    def _refit_and_save_best(best_cfg: Dict):
        """Optionally refit the current best on full data and save model + CV summary."""
        from darts.models import TFTModel
        model_best = TFTModel(
            input_chunk_length=best_cfg["input_chunk_length"],
            output_chunk_length=best_cfg["output_chunk_length"],
            hidden_size=best_cfg["hidden_size"],
            lstm_layers=best_cfg["lstm_layers"],
            num_attention_heads=best_cfg["num_attention_heads"],
            hidden_continuous_size=best_cfg["hidden_continuous_size"],
            dropout=best_cfg["dropout"],
            batch_size=best_cfg["batch_size"],
            n_epochs=best_cfg["n_epochs"],
            likelihood=QuantileRegression(quantiles=quantiles),
            categorical_embedding_sizes=categorical_embedding_sizes,
            use_static_covariates=True,
            add_encoders=add_encoders,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": best_cfg["lr"], "weight_decay": best_cfg["weight_decay"]},
            pl_trainer_kwargs={"accelerator": accelerator, "devices": devices,
                               "gradient_clip_val": 1.0, "callbacks": [TFMProgressBar()]},
            random_state=args.seed, show_warnings=False,
        )
        model_best.fit(series=series_scaled, past_covariates=past_covs_scaled,
                       future_covariates=future_covs_scaled, verbose=False)

        # (optional) log CV on refit to keep comparability with other runs
        cv_best = model_best.historical_forecasts(
            series=series_scaled, past_covariates=past_covs_scaled, future_covariates=future_covs_scaled,
            start=-(N_WINDOWS * FORECAST_H), start_format="position",
            forecast_horizon=FORECAST_H, stride=args.stride, retrain=True,
            last_points_only=False, num_samples=args.samples, verbose=False,
        )
        m_best = metrics_from_cv(cv_best, series_scaled, y_scaler, compute_global=True)
        best_overall = m_best["overall"]
        best_overall_path = run_dir / "best_overall.json"
        best_overall.to_json(best_overall_path, indent=2)

        model_path = run_dir / "model_best.pt"
        model_best.save(str(model_path))

        # keep pointers current
        copy_as_pointer(best_overall_path, fixed_best_overall)
        copy_as_pointer(model_path,       fixed_best_model)

        # optionally gate by --only-if-better against a saved “gate” json
        cand_val = float(best_overall.get(args.opt_metric, np.inf if _lower_is_better(args.opt_metric) else -np.inf))
        best_gate_json = artifact_path("metrics") / f"search_best_{args.opt_metric}_bo.json"
        payload = {"metric_name": args.opt_metric, "run_dir": str(run_dir),
                   "overall_json": str(best_overall_path), "model_path": str(model_path)}
        if args.only_if_better:
            update_best_if_better(
                candidate_metric=cand_val, best_json_path=best_gate_json,
                candidate_payload=payload, lower_is_better=_lower_is_better(args.opt_metric),
            )

    # ---------- objective ----------
    def objective(trial: optuna.trial.Trial) -> float:
        cfg = _suggest_params(trial)
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
                "accelerator": accelerator, "devices": devices,
                "gradient_clip_val": 1.0, "callbacks": [TFMProgressBar()],
            },
            random_state=args.seed,
            show_warnings=False,
        )

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
        trial.set_user_attr("metrics", {
            "mql_0_50": float(overall.get("mql_0_50", np.nan)),
            "rmse":     float(overall.get("rmse", np.nan)),
            "mae":      float(overall.get("mae", np.nan)),
            "smape":    float(overall.get("smape", np.nan)),
            "mic_10_90":float(overall.get("mic_10_90", np.nan)),
        })
        trial.set_user_attr("cfg", cfg)

        score = float(overall.get(opt_key, np.nan))
        if np.isnan(score):
            raise RuntimeError(f"Opt metric {opt_key} is NaN")

        if opt_key == "mic_10_90" and (args.mic_target is not None):
            return abs(score - float(args.mic_target))  # bring coverage close to target
        return score

    # ---------- progress + per-trial persistence callback ----------
    total_trials = None if args.timeout else int(args.n_trials)
    pbar = tqdm(total=total_trials, desc="BO trials", leave=True) if (_HAS_TQDM and args.progress) else None

    def _on_trial_end(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        # progress
        if pbar is not None:
            pbar.update(1)

        # always refresh trials.csv (so you can stop anytime)
        _dump_trials_csv(study)

        # if this trial is the new best → save best json and optionally refit+save model
        if study.best_trial and trial.number == study.best_trial.number:
            best_cfg, _ = _write_best_payload(study)
            if args.save_on_improve and best_cfg:
                print(f"[improve] Refit+save best @ trial {trial.number}: {best_cfg}")
                _refit_and_save_best(best_cfg)

    # ---------- run optimization ----------
    study.optimize(
        objective,
        n_trials=None if args.timeout else args.n_trials,
        timeout=args.timeout,
        callbacks=[_on_trial_end],
    )
    if pbar is not None:
        pbar.close()

    # ------------- final saves (in case last trial wasn't an improvement) -------------
    _dump_trials_csv(study)
    best_cfg, best_m = _write_best_payload(study)

    print(f"\n✅ BO finished. Best {args.opt_metric} = {study.best_value:.4f}")
    print(f"All trials → {trials_csv}")
    print(f"Best trial → {best_json}")

    # ------------- (optional) final refit if not already done -------------
    if best_cfg and not args.save_on_improve:
        # save a final refit so you always leave with a model
        _refit_and_save_best(best_cfg)

    print("\n✅ Done. Fixed pointers updated:")
    print(f"- Trials pointer       → {artifact_path('metrics') / 'search_trials_bo.csv'}")
    print(f"- Best json pointer    → {artifact_path('runs') / 'search_best_bo.json'}")
    print(f"- Best metrics pointer → {artifact_path('metrics') / 'search_best_overall_bo.json'}")
    print(f"- Best model pointer   → {artifact_path('models') / 'tft_best_from_search_bo.pt'}")


if __name__ == "__main__":
    main()
