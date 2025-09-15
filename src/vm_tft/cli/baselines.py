from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict
import warnings

import numpy as np
import pandas as pd

from statsmodels.tools.sm_exceptions import ConvergenceWarning

from darts.timeseries import TimeSeries, concatenate
from darts.models import NaiveDrift
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing as ETS
from darts.models.forecasting.theta import Theta
from darts.metrics import rmse as darts_rmse, mae as darts_mae, smape as darts_smape

# Darts >= 0.30 uses an enum for seasonality mode; older versions accept strings.
try:
    from darts.utils.statistics import SeasonalityMode as SM
except Exception:
    SM = None

from vm_tft.cfg import (
    ALL_FILE, FREQ, TIME_COL, GROUP_COL, TARGET,
    FORECAST_H, N_WINDOWS, ARTIFACTS
)

# Silence common Holt-Winters convergence chatter so logs stay readable
warnings.simplefilter("ignore", ConvergenceWarning)


# ─────────────────────────────
# Helpers
# ─────────────────────────────
def _series_from_df(df: pd.DataFrame) -> List[TimeSeries]:
    """
    Build one TimeSeries per well from a long dataframe with [TIME_COL, GROUP_COL, TARGET].
    """
    return TimeSeries.from_group_dataframe(
        df=df[[TIME_COL, GROUP_COL, TARGET]].copy(),
        time_col=TIME_COL,
        group_cols=GROUP_COL,
        value_cols=TARGET,
        freq=FREQ,
    )


def _coerce_cv_to_per_series_list(cv) -> List[TimeSeries]:
    """
    Normalize historical_forecasts() output to List[TimeSeries]:
      - TimeSeries                      -> [TimeSeries]
      - List[TimeSeries]                -> as-is
      - List[List[TimeSeries]] (windows)-> concatenate windows along the time axis
    """
    if isinstance(cv, TimeSeries):
        return [cv]
    if not cv:
        return []
    if isinstance(cv[0], TimeSeries):
        return list(cv)

    out: List[TimeSeries] = []
    for win_list in cv:
        # Ensure time order, then merge windows for that series
        win_list = sorted(win_list, key=lambda s: s.start_time())
        merged = concatenate(win_list, axis="time", ignore_time_axis=True)
        out.append(merged)
    return out


def _iso_date(ts) -> str:
    """
    Return an ISO date string from an (optionally tz-aware) timestamp.
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.date().isoformat()


def _flatten_cv_and_score(cv, series, model_name: str, series_ids=None) -> List[Dict]:
    """
    Align prediction with truth on the overlap per series and compute basic metrics.
    Returns a list of dicts (one row per evaluated series).
    """
    preds_per_series = _coerce_cv_to_per_series_list(cv)
    trues_per_series = list(series) if isinstance(series, (list, tuple)) else [series]
    if series_ids is None:
        series_ids = list(range(len(preds_per_series)))

    rows: List[Dict] = []
    for sid, ts_true, ts_pred in zip(series_ids, trues_per_series, preds_per_series):
        # Overlap guard
        start = max(ts_true.start_time(), ts_pred.start_time())
        end   = min(ts_true.end_time(),   ts_pred.end_time())
        if end < start:
            continue

        y    = ts_true.slice(start, end)
        yhat = ts_pred.slice(start, end)

        rows.append({
            "model":        model_name,
            "well_id":      sid,
            "window_start": _iso_date(yhat.start_time()),
            "window_end":   _iso_date(yhat.end_time()),
            "n_points":     int(len(yhat)),
            "rmse":         float(darts_rmse(y, yhat)),
            "mae":          float(darts_mae(y, yhat)),
            "smape":        float(darts_smape(y, yhat)),
        })
    return rows


def _cv(model, series: List[TimeSeries], horizon: int, n_windows: int, stride: int):
    """
    Run rolling CV with a TFT-like policy:
      • start at -(n_windows * horizon)
      • retrain=True
      • last_points_only=False
      • hide 'start shifted' warnings for cleaner logs
    Returns Darts' historical_forecasts output (can be nested).
    """
    return model.historical_forecasts(
        series=series,
        start=-(n_windows * horizon),
        start_format="position",
        forecast_horizon=horizon,
        stride=stride,
        retrain=True,
        last_points_only=False,
        verbose=False,
        show_warnings=False,
    )


def make_ets_trend_only() -> ETS:
    """
    Robust ETS (Holt-Winters) baseline: no trend/seasonality components forced,
    which is stable for short, noisy monthly oil series.
    """
    return ETS(trend=None, seasonal=None)


def make_theta_baseline(series: List[TimeSeries],
                        additive_seasonality: bool = True,
                        monthly_period: int = 12) -> Theta:
    """
    Theta baseline with safe seasonality:
      • If series are too short (< 2*period), disable seasonality.
      • Use additive seasonality (works with zeros).
      • Use enum when available (prevents '.value' attribute crashes).
    """
    min_len = min(len(s) for s in series)
    seasonality_period = monthly_period if min_len >= 2 * monthly_period else None

    if SM is not None:  # newer Darts
        season_mode = SM.ADDITIVE if additive_seasonality else SM.MULTIPLICATIVE
    else:               # older Darts still accepts strings
        season_mode = "additive" if additive_seasonality else "multiplicative"

    return Theta(season_mode=season_mode, seasonality_period=seasonality_period)


# ─────────────────────────────
# Main
# ─────────────────────────────
def main():
    ap = argparse.ArgumentParser("Run simple baselines with TFT-like CV and save metrics")
    ap.add_argument("--csv", default=ALL_FILE, help="Prepared ALL_FILE path")
    ap.add_argument("--h", type=int, default=FORECAST_H, help="Forecast horizon per window")
    ap.add_argument("--n-windows", type=int, default=N_WINDOWS, help="Number of rolling windows")
    ap.add_argument("--stride", type=int, default=FORECAST_H, help="Step between windows")
    ap.add_argument("--outdir", default=str(Path(ARTIFACTS) / "metrics"), help="Output dir for metrics")
    args = ap.parse_args()

    # Load & build per-well series
    df = pd.read_csv(args.csv, parse_dates=[TIME_COL])
    ids = sorted(df[GROUP_COL].dropna().unique().tolist())
    series = _series_from_df(df)

    # Define baselines
    baselines = {
        "NaiveDrift":           NaiveDrift(),
        "ExponentialSmoothing": make_ets_trend_only(),                 # “HW” (trend-only / robust)
        "Theta":                make_theta_baseline(series, True, 12),  # additive; disables seasonality if too short
    }

    # Run CV per model (robust) and collect metrics
    all_rows: List[Dict] = []
    failures: List[tuple[str, str]] = []

    for name, model in baselines.items():
        try:
            cv = _cv(model, series, horizon=args.h, n_windows=args.n_windows, stride=args.stride)
            rows = _flatten_cv_and_score(cv, series, name, ids)
            if not rows:
                failures.append((name, "No overlapping forecast/actual points (empty rows)."))
            else:
                all_rows.extend(rows)
        except Exception as e:
            failures.append((name, repr(e)))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save per-series/window rows (even if empty so you can inspect)
    per_series_csv = outdir / "baselines_per_series.csv"
    pd.DataFrame(all_rows).to_csv(per_series_csv, index=False)
    print(f"✔ Saved per-series rows → {per_series_csv} (n={len(all_rows)})")

    # Safe aggregation
    dfm = pd.DataFrame(all_rows)
    if dfm.empty or "model" not in dfm.columns:
        print("⚠️ No successful baseline rows → skipping overall aggregation.")
    else:
        metric_cols = [c for c in ["rmse", "mae", "smape"] if c in dfm.columns]
        if not metric_cols:
            print("⚠️ No metric columns found in rows → skipping overall aggregation.")
        else:
            summary = (
                dfm.groupby("model")[metric_cols]
                   .agg(["mean", "median"])
                   .round(4)
            )
            # Flatten MultiIndex columns for clean JSON
            summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
            overall_json = outdir / "baselines_overall.json"
            overall_json.write_text(json.dumps(summary.to_dict(orient="index"), indent=2))
            print(f"✔ Saved overall summary → {overall_json}")

    # Log failures at the end without crashing the run
    if failures:
        print("—— Baseline failures ——————————")
        for name, msg in failures:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
