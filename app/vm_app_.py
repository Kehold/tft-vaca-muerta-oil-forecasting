from __future__ import annotations

# --- make local src/ importable on Streamlit Cloud ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root (app/ is one level below)
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# -----------------------------------------------------

import plotly.io as pio
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.stattools import pacf as sm_pacf
from statsmodels.tsa.seasonal import STL
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Darts objects are optional – only needed if you later add “live” model actions
try:
    from darts.models import TFTModel  
    from darts.timeseries import TimeSeries  
except Exception:
    TFTModel = None
    TimeSeries = None

# ---- internal modules (your package) ----
from vm_tft.cfg import (
    ALL_FILE, TEST_FILE, ARTIFACTS,
    TIME_COL, GROUP_COL, TARGET, FREQ,
)
from vm_tft.io_utils import ensure_dir  

from PIL import Image, UnidentifiedImageError

def show_image_safe(rel_path: str, caption: str = "") -> None:
    img_path = ROOT / rel_path
    if not img_path.exists():
        st.warning(f"Image not found: {img_path} (check name & case).")
        return
    try:
        img = Image.open(img_path)
    except (UnidentifiedImageError, OSError) as e:
        st.error(f"Could not open image {img_path.name}: {e}")
        return
    # ✅ For images, use use_column_width
    st.image(img, use_column_width=True, caption=caption)

# =========================
# Page Settings
# =========================
st.set_page_config(
    page_title="Vaca Muerta – TFT Dashboard",
    layout="wide",
    page_icon="🛢️",
)

# =========================
# Global Plot Style (uniform across app)
# =========================
def _set_plot_style():
    # Base template for a clean look
    px.defaults.template = "simple_white"

    # Consistent colorway for all figures
    px.defaults.color_discrete_sequence = [
        "#2E86AB",  # blue
        "#F18F01",  # orange
        "#C73E1D",  # red
        "#4ECDC4",  # teal
        "#6C757D",  # gray
    ]

    # Create a small custom template to set font and margins
    vm_tpl = go.layout.Template()
    vm_tpl.layout.font = dict(
        family="Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
        size=14,
    )
    vm_tpl.layout.margin = dict(l=10, r=10, t=40, b=10)

    # Register and activate combined template
    pio.templates["vm_white"] = vm_tpl
    pio.templates.default = "simple_white+vm_white"

_set_plot_style()

# =========================
# Utilities & Cache
# =========================
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=[TIME_COL])
    return df

@st.cache_data(show_spinner=False)
def read_metrics_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def read_metrics_csv(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)

# @st.cache_data(show_spinner=False)
# def compute_acf_for_well(df: pd.DataFrame, well_id: int, y_col: str, max_lags: int, do_diff: bool):
#     """Returns lags, acf_values, and CI half-width (±1.96/sqrt(N))."""
#     col = GROUP_COL if GROUP_COL in df.columns else "well_id"
#     s = (
#         df.loc[df[col] == well_id, [TIME_COL, y_col]]
#           .dropna()
#           .sort_values(TIME_COL)[y_col]
#           .to_numpy(dtype=float)
#     )
#     if s.size == 0:
#         return np.arange(0), np.array([]), 0.0

#     if do_diff and s.size > 1:
#         s = np.diff(s)

#     # statsmodels ACF (we’ll use the simple normal CI)
#     nlags = int(max(1, min(max_lags, max(1, s.size - 1))))
#     acf_vals = sm_acf(s, nlags=nlags, fft=True)
#     ci_half = 1.96 / np.sqrt(s.size)
#     lags = np.arange(len(acf_vals))
#     return lags, acf_vals, float(ci_half)

@st.cache_data(show_spinner=False)
def compute_acf_for_well(df, well_id, target_col, nlags=36, difference=False):
    s = (
        df.loc[df[GROUP_COL] == well_id, target_col]
        .dropna().astype(float).to_numpy()
    )
    if difference:
        s = np.diff(s)
    if len(s) < 3:
        return np.array([]), np.array([]), 0.0
    nlags = int(min(nlags, len(s) - 1))
    acf_vals = sm_acf(s, nlags=nlags, fft=True)
    lags = np.arange(len(acf_vals))
    ci_half = 1.96 / np.sqrt(len(s))
    return lags, acf_vals, ci_half

def plot_acf_plotly(lags: np.ndarray, acf_vals: np.ndarray, ci_half: float, title: str = "ACF"):
    import plotly.graph_objects as go
    if lags.size == 0:
        return go.Figure()

    fig = go.Figure()

    # Confidence band
    x_band = np.concatenate([lags, lags[::-1]])
    y_band = np.concatenate([np.full_like(lags, ci_half, dtype=float),
                             np.full_like(lags, -ci_half, dtype=float)[::-1]])
    fig.add_trace(go.Scatter(
        x=x_band, y=y_band, fill="toself", name="95% CI",
        line=dict(width=0), opacity=0.20
    ))

    # Zero line
    fig.add_hline(y=0.0, line_width=1, line_color="black", opacity=0.6)

    # ACF bars
    fig.add_trace(go.Bar(x=lags, y=acf_vals, name="ACF", marker_line_color="black", marker_line_width=1))

    fig.update_layout(
        title=title,
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        bargap=0.15,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    return fig

@st.cache_data(show_spinner=False)
def compute_pacf_for_well(df, well_id, target_col, nlags=36, difference=False):
    s = (
        df.loc[df[GROUP_COL] == well_id, target_col]
        .dropna().astype(float).to_numpy()
    )
    if difference:
        s = np.diff(s)
    # need at least 3 points
    if len(s) < 3:
        return np.array([]), np.array([]), 0.0

    # PACF constraint: nlags < len(s) / 2
    # we also guard against tiny results so we don't request nlags < 1
    max_allowed = max(1, (len(s) // 2) - 1)
    nlags = int(min(nlags, max_allowed))

    if nlags < 1:
        return np.array([]), np.array([]), 0.0

    pacf_vals = sm_pacf(s, nlags=nlags, method="ywadjusted")  # valid/stable method
    lags = np.arange(len(pacf_vals))
    ci_half = 1.96 / np.sqrt(len(s))
    return lags, pacf_vals, ci_half

def plot_pacf_plotly(lags: np.ndarray, pacf_vals: np.ndarray, ci_half: float, title: str = "PACF"):
    fig = go.Figure()
    if lags.size == 0:
        return fig

    x_band = np.concatenate([lags, lags[::-1]])
    y_band = np.concatenate([np.full_like(lags, ci_half, dtype=float),
                             np.full_like(lags, -ci_half, dtype=float)[::-1]])
    fig.add_trace(go.Scatter(
        x=x_band, y=y_band, fill="toself", name="95% CI",
        line=dict(width=0), opacity=0.20
    ))
    fig.add_hline(y=0.0, line_width=1, line_color="black", opacity=0.6)
    fig.add_trace(go.Bar(x=lags, y=pacf_vals, name="PACF", marker_line_color="black", marker_line_width=1))
    fig.update_layout(
        title=title, xaxis_title="Lag", yaxis_title="Partial autocorr",
        bargap=0.15, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
    )
    return fig

@st.cache_data(show_spinner=False)
def compute_stl_for_well(df: pd.DataFrame, well_id: int, y_col: str, period: int):
    col = GROUP_COL if GROUP_COL in df.columns else "well_id"
    df_w = (df.loc[df[col] == well_id, [TIME_COL, y_col]]
              .dropna()
              .sort_values(TIME_COL)
              .set_index(TIME_COL))
    if df_w.empty or df_w.shape[0] < 2 * period:
        return None  # not enough data

    s = df_w[y_col].astype(float)
    stl = STL(s, period=period, robust=True)
    res = stl.fit()
    return {
        "index": s.index,
        "observed": s.values,
        "trend": res.trend.values,
        "seasonal": res.seasonal.values,
        "resid": res.resid.values,
    }
    
def plot_stl_small_multiples(stl_dict, title_prefix="STL Decomposition"):
    if stl_dict is None:
        return go.Figure()

    idx = stl_dict["index"]
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
    )
    fig.add_trace(go.Scatter(x=idx, y=stl_dict["observed"], mode="lines", name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=stl_dict["trend"], mode="lines", name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=idx, y=stl_dict["seasonal"], mode="lines", name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=idx, y=stl_dict["resid"], mode="lines", name="Residual"), row=4, col=1)
    fig.add_hline(y=0.0, line=dict(width=1, color="black", dash="dot"), row=4, col=1)

    fig.update_layout(
        title=title_prefix, height=700,
        margin=dict(l=10, r=10, t=40, b=10), showlegend=False
    )
    return fig

def plot_series_with_overlays(df_w: pd.DataFrame, y_col: str, title: str,
                              roll_win: int, show_mean: bool, show_median: bool):
    fig = go.Figure()
    if df_w.empty:
        return fig

    fig.add_trace(go.Scatter(
        x=df_w[TIME_COL], y=df_w[y_col], mode="lines", name=y_col
    ))

    if show_mean and roll_win > 1:
        rm = df_w[y_col].rolling(roll_win, min_periods=max(1, roll_win // 2)).mean()
        fig.add_trace(go.Scatter(
            x=df_w[TIME_COL], y=rm, mode="lines", name=f"Rolling mean ({roll_win})", line=dict(width=2)
        ))

    if show_median and roll_win > 1:
        rmed = df_w[y_col].rolling(roll_win, min_periods=max(1, roll_win // 2)).median()
        fig.add_trace(go.Scatter(
            x=df_w[TIME_COL], y=rmed, mode="lines", name=f"Rolling median ({roll_win})", line=dict(width=2, dash="dot")
        ))

    fig.update_layout(title=title, margin=dict(l=0, r=0, t=50, b=0))
    return fig  

def _series_index_for_well(df_test: pd.DataFrame, well_id: int) -> Optional[int]:
    """Mirror vm-test ordering: sorted unique well IDs → index."""
    if df_test.empty or GROUP_COL not in df_test.columns:
        return None
    uniq = sorted(df_test[GROUP_COL].dropna().unique().tolist())
    try:
        return uniq.index(well_id)
    except ValueError:
        return None

def _find_prediction_image(preds_dir: Path, well_id: int, series_idx: Optional[int], case_tag: str) -> Optional[Path]:
    """
    Look for Darts-style PNGs saved by CLI, preferring well-named if you add them later.
    Supported:
      - caseB_well_<id>.png / caseC_well_<id>.png (optional, future-friendly)
      - caseB_series_###.png / caseC_series_###.png (current)
    """
    # 1) by well id (optional nicety)
    by_well = preds_dir / f"{case_tag}_well_{well_id}.png"
    if by_well.exists():
        return by_well

    # 2) by series index (current)
    if series_idx is not None:
        by_series = preds_dir / f"{case_tag}_series_{series_idx:03d}.png"
        if by_series.exists():
            return by_series

    return None

def _cum_forecast_df(
    hist_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    time_col: str,
    hist_cum_col: str = "cum_oil_bbl",
    p10_col: str = "p10",
    p50_col: str = "p50",
    p90_col: str = "p90",
) -> Optional[pd.DataFrame]:
    """
    Build a DataFrame with cumulative forecast curves (cum_p10, cum_p50, cum_p90)
    starting from the last available historical cumulative value.
    Assumes prediction columns are *rates per day* (e.g., bpd). We convert to
    monthly barrels via days_in_month.
    """
    if pred_df.empty or time_col not in pred_df.columns:
        return None

    # last historical cumulative value
    cum_hist = None
    if not hist_df.empty and hist_cum_col in hist_df.columns:
        _h = hist_df[[time_col, hist_cum_col]].dropna(subset=[hist_cum_col]).sort_values(time_col)
        if not _h.empty:
            cum_hist = float(_h[hist_cum_col].iloc[-1])
    if cum_hist is None:
        cum_hist = 0.0  # fallback if missing historical cumulative

    dfp = pred_df.copy()
    dfp = dfp.sort_values(time_col)
    # days in each monthly step
    if not np.issubdtype(dfp[time_col].dtype, np.datetime64):
        return None
    days = dfp[time_col].dt.days_in_month.astype(float)

    # If any of the quantile cols are missing, derive safe fallbacks
    if p50_col not in dfp.columns and "y_hat" in dfp.columns:
        dfp[p50_col] = dfp["y_hat"]
    for c in (p10_col, p90_col):
        if c not in dfp.columns:
            dfp[c] = dfp.get(p50_col, np.nan)

    # clip negative rates (just in case) then convert rate→monthly volume
    for c in (p10_col, p50_col, p90_col):
        if c in dfp.columns:
            dfp[c] = np.clip(dfp[c].astype(float), a_min=0, a_max=None) * days

    out = pd.DataFrame({time_col: dfp[time_col].values})
    out["cum_p10"] = cum_hist + dfp[p10_col].cumsum()
    out["cum_p50"] = cum_hist + dfp[p50_col].cumsum()
    out["cum_p90"] = cum_hist + dfp[p90_col].cumsum()
    return out

def plot_cumulative_oil(
    hist_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    time_col: str,
    hist_cum_col: str = "cum_oil_bbl",
    title: str = "5-year cumulative oil",
):
    """
    Plot historical cumulative + forecast cumulative band (P10–P90) and median.
    """
    cum_fore = _cum_forecast_df(hist_df, pred_df, time_col, hist_cum_col)
    if cum_fore is None:
        return None

    fig = go.Figure()

    # Historical cumulative (if present)
    if not hist_df.empty and hist_cum_col in hist_df.columns:
        h = hist_df[[time_col, hist_cum_col]].dropna().sort_values(time_col)
        if not h.empty:
            fig.add_trace(go.Scatter(
                x=h[time_col], y=h[hist_cum_col],
                mode="lines", name="Historical cum",
                line=dict(width=2)
            ))

    # Forecast cumulative band and median
    fig.add_trace(go.Scatter(
        x=pd.concat([cum_fore[time_col], cum_fore[time_col][::-1]]),
        y=pd.concat([cum_fore["cum_p90"], cum_fore["cum_p10"][::-1]]),
        fill="toself", name="P10–P90 (cum)",
        line=dict(width=0), opacity=0.25
    ))
    fig.add_trace(go.Scatter(
        x=cum_fore[time_col], y=cum_fore["cum_p50"],
        mode="lines", name="P50 (cum)",
        line=dict(width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Oil (bbl)",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig

def list_artifacts() -> dict:
    base = Path(ARTIFACTS)
    return {
        "models": {
            "best_from_search": base / "models" / "model_best_bo_baseline_scaler.pt", # has MinMax scaler
            # "best_manual": base / "models" / "model_log_baseline_repeat.pt", # has MinMax scaler
        },
        "cv": {
            "overall_metrics": base / "metrics" / "metrics_overall_baseline_again.json",
            "per_series_window": base / "metrics" / "metrics_per_series_window_baseline_again.csv",
        },
        "search": {
            "trials_csv": base / "metrics" / "search_trials_bo_baseline_scaler.csv",
            "best_config": base / "metrics" / "search_best_bo_baseline_scaler.json",
        },
        "test": {
            # Keep B (has metrics), add C (no metrics by design)
            "caseB_metrics": base / "predictions" / "caseB_overall_mean_bo_baseline.json",
            "predictions_dir": base / "predictions" / "plots",
        },
        "explain": {
            "local_dir": base / "explain" / "local",
            "attention_dir": base / "explain" / "attention",
            "global_csv": base / "explain" / "global" / "global_variable_importance_pseudo.csv",
        }
    }

def get_well_ids(df: pd.DataFrame) -> List[int]:
    if df.empty:
        return []
    col = GROUP_COL if GROUP_COL in df.columns else "well_id"
    return sorted(df[col].dropna().unique().tolist())

def filter_df_by_well(df: pd.DataFrame, well_id: int) -> pd.DataFrame:
    col = GROUP_COL if GROUP_COL in df.columns else "well_id"
    return df[df[col] == well_id].sort_values(TIME_COL)

def metric_card(label: str, value: float | str, help_text: str = ""):
    st.metric(label, value)
    if help_text:
        st.caption(help_text)

def plot_series_px(df: pd.DataFrame, y: str, title: str):
    fig = px.line(df, x=TIME_COL, y=y, title=title)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
def _series_index_for_well(df: pd.DataFrame, well_id: int) -> Optional[int]:
    """Map well_id -> series index using the same ordering used in test_predict."""
    if df.empty or well_id is None:
        return None
    ids = sorted(df[GROUP_COL].dropna().unique().tolist())
    try:
        return ids.index(well_id)
    except ValueError:
        return None

def _find_prediction_image(preds_dir: Path, well_id: int, series_idx: Optional[int], case_tag: str) -> Optional[Path]:
    """
    Try several filename patterns to locate a pre-rendered Darts PNG:
      - <case>_well_<WELLID>.png          (future-friendly)
      - pred_well_<WELLID>.png            (if you decide to save one generic per well)
      - <case>_series_<IDX:03d>.png       (your current naming)
    """
    candidates = []
    if well_id is not None:
        candidates += [
            preds_dir / f"{case_tag}_well_{well_id}.png",
            preds_dir / f"pred_well_{well_id}.png",
        ]
    if series_idx is not None:
        candidates.append(preds_dir / f"{case_tag}_series_{series_idx:03d}.png")
    for p in candidates:
        if p.exists():
            return p
    return None

@st.cache_data(show_spinner=False)
def hw_forecast(df_well: pd.DataFrame, target_col: str, horizon: int = 12):
    """
    Holt-Winters baseline: trend-only (additive), no seasonality (common for monthly oil rates).
    Returns DataFrame with [TIME_COL, actual, hw_forecast].
    """
    ts = df_well[[TIME_COL, target_col]].dropna().sort_values(TIME_COL)
    y = ts[target_col].astype(float).values

    # Small guardrail for tiny series
    if len(y) < 6:
        # simple naive/fallback
        last = y[-1] if len(y) else 0.0
        yhat_in = np.full_like(y, last, dtype=float)
        yhat_out = np.full(horizon, last, dtype=float)
    else:
        model = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        yhat_in = fit.fittedvalues
        yhat_out = fit.forecast(horizon)

    # future dates
    freq = pd.infer_freq(ts[TIME_COL]) or "MS"
    last_date = ts[TIME_COL].iloc[-1]
    future_idx = pd.date_range(last_date, periods=horizon+1, freq=freq)[1:]

    df = pd.DataFrame({
        TIME_COL: list(ts[TIME_COL]) + list(future_idx),
        "actual": list(ts[target_col]) + [np.nan]*horizon,
        "baseline_forecast": list(yhat_in) + list(yhat_out)
    })
    return df


# --- Matplotlib forecast plot to match test_predict.py style ---
def plot_forecast_mpl(hist_df: pd.DataFrame, pred_df: pd.DataFrame, y_col: str, title: str):
    import matplotlib.pyplot as plt

    # keep history strictly before the prediction window (so it looks like the saved PNGs)
    if not pred_df.empty:
        pred_start = pred_df[TIME_COL].min()
        hist_df = hist_df.loc[hist_df[TIME_COL] < pred_start].copy()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Warm start (history)
    if not hist_df.empty:
        ax.plot(hist_df[TIME_COL], hist_df[y_col], label="Warm start (history)", lw=2, color="black")

    # Truth continuation
    if "y_true" in pred_df.columns:
        ax.plot(pred_df[TIME_COL], pred_df["y_true"], label="Actual (truth)", lw=2)

    # Prediction interval and median
    has_p10p90 = {"p10", "p90"}.issubset(pred_df.columns)
    if has_p10p90:
        ax.fill_between(pred_df[TIME_COL], pred_df["p10"], pred_df["p90"], alpha=0.30, label="Pred 10–90%")
    if "p50" in pred_df.columns:
        ax.plot(pred_df[TIME_COL], pred_df["p50"], label="Pred median (q0.50)", lw=2)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_col)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_forecast_band_p10_p50_p90(
    hist: pd.DataFrame,
    pred: pd.DataFrame,
    y_col: str,
    title: str = "Forecast",
):
    # Colors consistent with defaults: history (blue), p50 (orange), band (orange fill)
    band_color = "rgba(241, 143, 1, 0.22)"  # #F18F01 w/ alpha
    line_color_p50 = "#F18F01"

    fig = go.Figure()

    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist[TIME_COL], y=hist[y_col],
            mode="lines", name="Actual (history)"
        ))

    if not pred.empty and {"p10", "p50", "p90"}.issubset(pred.columns):
        # Prediction interval band: p10–p90
        fig.add_trace(go.Scatter(
            x=pd.concat([pred[TIME_COL], pred[TIME_COL][::-1]]),
            y=pd.concat([pred["p90"], pred["p10"][::-1]]),
            fill="toself",
            name="P10–P90",
            line=dict(width=0),
            fillcolor=band_color,
            hoverinfo="skip",
        ))
        # Median
        fig.add_trace(go.Scatter(
            x=pred[TIME_COL], y=pred["p50"],
            mode="lines", name="P50",
            line=dict(width=2, color=line_color_p50)
        ))
    elif not pred.empty and "y_hat" in pred.columns:
        fig.add_trace(go.Scatter(
            x=pred[TIME_COL], y=pred["y_hat"],
            mode="lines", name="Prediction", line=dict(width=2)
        ))

    fig.update_layout(title=title, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("Vaca Muerta – TFT")
section = st.sidebar.radio(
    "Navigate",
    ["Home", "Data Explorer", "Model CV", "Randomized Search", "Test Predictions", "Explainability"],
    index=0,
)

if st.sidebar.button("↻ Refresh data & metrics cache"):
    load_csv.clear()
    read_metrics_json.clear()
    read_metrics_csv.clear()
    st.experimental_rerun()
st.sidebar.markdown("---")

# global controls
art = list_artifacts()
df_all  = load_csv(ALL_FILE)
df_test = load_csv(TEST_FILE)

st.sidebar.subheader("Global controls")
default_set = st.sidebar.selectbox("Active dataset", ["Test", "Train/All"], index=0)
active_df = df_test if default_set == "Test" else df_all

well_ids = get_well_ids(active_df)
well_id = st.sidebar.selectbox("Well", well_ids, index=0 if well_ids else None)

st.sidebar.markdown("---")

# # Define the footer HTML and CSS
# footer_html = """
# <div style='position: absolute; bottom: 0; width: 100%; background-color: #f1f1f1; text-align: center; padding: 10px 0;'>
#     <p style='margin: 0; font-size: 12px;'>© 2025 Alexis Ortega | <a href="https://github.com/alexort74/ttf-vaca-muerta-oil-forecasting"></a></p>
# </div>
# """

# # Add the footer to the sidebar
# st.sidebar.markdown(footer_html, unsafe_allow_html=True)
# st.sidebar.caption(f"By Alexis Ortega")

# =========================
# Sections
# =========================

# ---- 0) Home ----
if section == "Home":
    st.header("🏠 Forecasting Oil Production with Temporal Fusion Transformers")

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Project Description")
        st.markdown(
            """
**Goal.** Build robust, explainable monthly production forecasts for Vaca Muerta wells using the
Temporal Fusion Transformer (TFT), with warm-start predictions for unseen wells.

**Scope.**
- Oil rate (bpd) as target; gas/water and engineering features as covariates.
- Static covariates: operator, area, completion campaign, lateral length, stages, fluids/proppants, etc.
- Past/future covariates include well age and calendar encoders.

**Methodology.**
- **Data prep** → clean & split (train/test) and generate well-age future covariates.
- **Scaling** → optional `log1p` + Std. Scaler for targets & past covariates.
- **Modeling** → TFT with quantile regression (**P10/P50/P90**).
- **Validation** → rolling windows via `historical_forecasts` (retrain each step).
- **Selection** → randomized hyperparameter search (minimize MQL@0.50).
- **Explainability** → local/global variable importance and attention maps.

**Tooling.**
- Python, PyTorch Lightning, Darts, scikit-learn, Plotly, Streamlit.
- Reproducible artifacts in `artifacts/` with “fixed pointers” to latest/best results.

Use the left navigation to explore data, CV metrics, randomized search results, test predictions,
and model explainability.
            """
        )

    with right:
        # st.image("data/images/tft_paper.png", use_container_width=True, caption="TFT forecasting architecture")
        show_image_safe("data/images/tft_paper.png", "TFT forecasting architecture")


# ---- 1) Data Explorer ----
if section == "Data Explorer":
    st.header("📊 Data Explorer")

    if active_df.empty:
        st.warning("No data found. Run `vm-prepare-data` first.")
        st.stop()
    if well_id is None:
        st.info("Select a well in the sidebar.")
        st.stop()

    # ---------- Sidebar controls (specific to this page) ----------
    sb = st.sidebar.expander("Data Explorer Controls", expanded=False)

    with sb:
        # --- in the sidebar, under "Data Explorer Controls" ---
        st.markdown("**Plot Selection**")
        acf_kind = st.radio(
            "Correlation view", ("ACF", "PACF"),
            index=0, help="Show either autocorrelation (ACF) or partial autocorrelation (PACF).")
        max_lags = st.slider("Max lags", 6, 60, 36, step=6, key="acf_lags")
        do_diff  = st.checkbox("Difference series (Δy) before ACF/PACF", value=False,  key="acf_diff", help="Compute correlations on first differences.")
        
        st.markdown("---")
        st.markdown("**Overlays**")
        roll_win = st.slider("Rolling window (months)", 3, 24, 6, 1, key="roll_win")
        show_mean = st.checkbox("Show rolling mean", value=True, key="roll_mean")
        show_median = st.checkbox("Show rolling median", value=False, key="roll_median")

        # st.markdown("---")
        # st.markdown("**ACF / PACF**")
        # max_lags = st.slider("Max lags", min_value=12, max_value=72, value=36, step=6, key="acf_lags")
        # do_diff = st.checkbox("Difference series before ACF/PACF", value=False, key="acf_diff")
        # show_pacf = st.checkbox("Show PACF", value=True, key="acf_show_pacf")

        st.markdown("---")
        st.markdown("**STL decomposition**")
        stl_on = st.checkbox("Enable STL", value=False, key="stl_on",
                             help="Requires enough data; ~2×period recommended.")
        stl_period = st.slider("STL period (months)", 6, 24, 12, 1, key="stl_period")

    # ---------- Main layout ----------
    left, right = st.columns([2, 1], gap="large")

    # Main time series with overlays
    with left:
        st.subheader("Production time series")
        df_w = filter_df_by_well(active_df, well_id)
        if df_w.empty:
            st.info("Selected well has no rows.")
        else:
            fig_ts = plot_series_with_overlays(
                df_w, TARGET, f"Well {well_id} – {TARGET}",
                roll_win=roll_win, show_mean=show_mean, show_median=show_median
            )
            st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})

    # ACF (+ optional PACF)
    with right:  # whatever column you use beside the time-series
        # if acf_kind == "ACF":
        #     lags, vals, ci = compute_acf_for_well(active_df, well_id, TARGET, max_lags, do_diff)
        #     st.subheader("Autocorrelation (ACF)")
        if acf_kind == "PACF":
            lags, vals, ci = compute_pacf_for_well(active_df, well_id, TARGET, max_lags, do_diff)
            st.subheader("Partial Autocorrelation (PACF)")
            # If we had to cap nlags, let the user know
            actual_nlags = max(0, len(vals) - 1)
            if actual_nlags and actual_nlags < max_lags:
                st.caption(f"Note: PACF max lags capped to {actual_nlags} due to sample-size limits.")
        else:
            lags, vals, ci = compute_acf_for_well(active_df, well_id, TARGET, max_lags, do_diff)
            st.subheader("Autocorrelation (ACF)")

        if len(vals):
            fig = go.Figure()
            fig.add_vline(x=0, line_width=1)
            fig.add_hrect(y0=-ci, y1=ci, opacity=0.15, line_width=0)
            fig.add_bar(x=lags, y=vals)
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Lag",
                yaxis_title="Correlation",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data points to compute correlations for this well.")

        # Quick summary
        if not df_w.empty:
            n = len(df_w)
            start = df_w[TIME_COL].min()
            end = df_w[TIME_COL].max()
            mean_v = df_w[TARGET].mean()
            std_v = df_w[TARGET].std()
            st.caption(f"**Summary** — n={n:,} • span: {start.date()} → {end.date()} • "
                       f"mean={mean_v:,.1f} • std={std_v:,.1f}")

    # STL (full width) if enabled
    if stl_on:
        st.subheader("Seasonal decomposition (STL)")
        stl_res = compute_stl_for_well(active_df, well_id, TARGET, period=stl_period)
        if stl_res is None:
            st.info("Not enough data for STL (need at least ~2×period).")
        else:
            fig_stl = plot_stl_small_multiples(stl_res, title_prefix=f"STL (Well {well_id}, period={stl_period})")
            st.plotly_chart(fig_stl, use_container_width=True, config={"displayModeBar": False})

    # Raw table (filtered to the selected well)
    st.subheader("Raw data")

    df_w = filter_df_by_well(active_df, well_id)
    if df_w.empty:
        st.info("Selected well has no rows.")
    else:
        st.dataframe(
            df_w.sort_values(TIME_COL).head(200),
            use_container_width=True
        )
        
    st.subheader("Baseline Forecast (Holt-Winters)")
    if not df_w.empty:
        df_f = hw_forecast(df_w, TARGET, horizon=12)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_f[TIME_COL], y=df_f["actual"], mode="lines+markers", name="Actual"))
        fig.add_trace(go.Scatter(x=df_f[TIME_COL], y=df_f["baseline_forecast"], mode="lines+markers", name="HW baseline"))
        fig.update_layout(title=f"Baseline forecast – Well {well_id}", xaxis_title="Date", yaxis_title=TARGET)
        st.plotly_chart(fig, use_container_width=True)


# ---- 2) Model CV ----
elif section == "Model CV":
    st.header("🧪 Cross-Validation Results")
    
    # ---------- Sidebar controls (specific to this page) ----------
    with st.sidebar.expander("Model CV controls", expanded=True):
        # --- in the sidebar, under "Data Explorer Controls" ---
        cv_hist_metric = st.radio(
            "Histogram metric",
            ["RMSE", "MAE", "sMAPE"],
            index=0,
            help="Choose which metric to visualize in the histogram."
        )
        # Optional bins slider (uncomment if you want)
        # cv_bins = st.slider("Histogram bins", 10, 80, 40, 5)

    overall_path = art["cv"]["overall_metrics"]
    per_series_path = art["cv"]["per_series_window"]

    overall = read_metrics_json(overall_path)
    per_series = read_metrics_csv(per_series_path)

    # Map pretty label -> dataframe column
    metric_map = {"RMSE": "rmse", "MAE": "mae", "sMAPE": "smape"}
    # default if sidebar didn't render for any reason
    selected_label = locals().get("cv_hist_metric", "RMSE")
    hist_col = metric_map.get(selected_label, "rmse")

    # Top row: KPI cards (left) + histogram (right)
    left, right = st.columns([1.2, 1.8], gap="large")

    with left:
        st.subheader("Overall metrics")
        if overall:
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE",   f"{overall.get('rmse',  np.nan):.2f}")
            c2.metric("MAE",    f"{overall.get('mae',   np.nan):.2f}")
            c3.metric("sMAPE",  f"{overall.get('smape', np.nan):.2f}")

            c4, c5, c6 = st.columns(3)
            c4.metric("MQL@0.50",      f"{overall.get('mql_0_50',  np.nan):.2f}")
            c5.metric("MIC@10–90",     f"{overall.get('mic_10_90', np.nan):.3f}")
            c6.metric("MAPE",          f"{overall.get('mape',       np.nan):.2f}")
        else:
            st.info("No CV overall metrics found yet.")

    with right:
        st.subheader(f"{selected_label} distribution")
        if per_series.empty:
            st.info("No per-series/window CSV found.")
        else:
            fig = px.histogram(
                per_series,
                x=hist_col,
                nbins=40,  # or use `cv_bins` if you enable the slider
                title=f"{selected_label} distribution",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Full table below
    st.subheader("Per-series / window metrics")
    if per_series.empty:
        st.info("No per-series/window CSV found.")
    else:
        st.dataframe(per_series, use_container_width=True, height=420)

# ---- 3) Randomized Search ----
elif section == "Randomized Search":
    st.header("🔍 Randomized Search")

    trials_csv = art["search"]["trials_csv"]
    best_cfg   = art["search"]["best_config"]

    trials = read_metrics_csv(trials_csv)

    # Columns we never want as HP selectors
    exclude_cols = {
        "mql_0_50", "rmse", "mae", "smape", "mic_10_90",
        "input_chunk_length", "output_chunk_length",
    }
    hp_cols = [c for c in trials.columns if c not in exclude_cols] if not trials.empty else []

    # ---------- Sidebar controls (inside expander) ----------
    with st.sidebar.expander("Randomized Search controls", expanded=False):
        metric_opt = st.radio(
            "Optimize / show metric",
            ["mql_0_50", "rmse", "mae", "smape", "mic_10_90"],
            index=0,
            help="Y-axis metric for the scatter plot.",
            key="rs_metric",
        )

        if hp_cols:
            x_hp = st.selectbox("X hyperparameter", hp_cols, index=0, key="rs_x_hp")
            remaining = [c for c in hp_cols if c != x_hp] or [x_hp]
            color_hp = st.selectbox("Color by (hyperparameter)", remaining, index=0, key="rs_color_hp")
        else:
            st.caption("No trials loaded yet — controls disabled.")
            x_hp, color_hp = None, None

    if trials.empty:
        st.info("No trials summary yet. Run `vm-search-fit`.")
    else:
        st.subheader("Trials summary")
        st.dataframe(trials, use_container_width=True, height=420)

        if x_hp is not None and color_hp is not None:
            fig = px.scatter(
                trials,
                x=x_hp,
                y=metric_opt,
                color=color_hp,
                size=metric_opt,            # 🔹 scale point size by metric
                size_max=18,                # 🔹 cap bubble size
                hover_data=trials.columns,
                title=f"{metric_opt} vs {x_hp} (color={color_hp}, size={metric_opt})",
            )
            fig.update_traces(marker=dict(line=dict(width=0.5, color="DarkSlateGrey")))
            fig.update_layout(
                margin=dict(l=0, r=0, t=60, b=0),
                legend=dict(itemsizing="constant")
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---- Best configuration ----
    st.subheader("Best configuration")
    best = read_metrics_json(best_cfg)

    if isinstance(best, dict) and ("cfg" in best and "metrics" in best):
        # left table: hyperparameters (exactly what's under cfg)
        cfg_dict = best["cfg"]
        df_hparams = pd.DataFrame(sorted(cfg_dict.items()), columns=["Hyperparameter", "Value"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Hyperparameters**")
            st.table(df_hparams.set_index("Hyperparameter"))

        # right table: ONLY metrics (strip out any hp keys that leaked into 'metrics')
        metric_keys_whitelist = ["mql_0_50", "rmse", "mae", "smape", "mic_10_90"]
        metrics_raw = best["metrics"] or {}
        metrics_only = {k: metrics_raw[k] for k in metric_keys_whitelist if k in metrics_raw}

        # pretty order: put the sidebar-selected metric first if present
        order = [metric_opt] + [k for k in metric_keys_whitelist if k != metric_opt]
        metrics_ordered = [(k, metrics_only[k]) for k in order if k in metrics_only]

        df_metrics = pd.DataFrame(metrics_ordered, columns=["Metric", "Value"])
        # optional: nicer formatting
        df_metrics["Value"] = df_metrics["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

        with c2:
            st.markdown("**Metrics**")
            st.table(df_metrics.set_index("Metric"))
    else:
        st.caption("No best_config.json yet or file not in expected format.")

# ---- 4) Test Predictions ----
elif section == "Test Predictions":
    st.header("🧪 Test Predictions (Warm-start)")

    # Fixed directories / files (from list_artifacts())
    preds_dir     = art["test"]["predictions_dir"]
    caseB_metrics = read_metrics_json(art["test"]["caseB_metrics"])

    # ---------------- Sidebar note ----------------
    st.sidebar.caption("This page shows Case B metrics and a Case C (5-year) cumulative plot.")

    st.subheader("Per-well visualization (Darts PNGs preferred)")
    if active_df.empty or well_id is None:
        st.info("Select a well to view predictions.")
        st.stop()

    # Map selected well → TEST series index (matches vm-test export order)
    series_idx = _series_index_for_well(df_test, well_id)

    # Try to show Darts-rendered PNGs first (exact pred.plot style)
    colB, colC = st.columns(2)
    shown_any = False
    for case_tag, col in [("caseB", colB), ("caseC", colC)]:
        png_path = _find_prediction_image(Path(preds_dir or ""), well_id, series_idx, case_tag)
        if png_path is not None:
            col.image(str(png_path), caption=f"{case_tag.upper()} – Well {well_id}", use_container_width=True)
            shown_any = True

    # If no PNGs found, fall back to CSV-driven Matplotlib replica (full width)
    if not shown_any:
        per_well_csv = Path(preds_dir or "") / f"pred_well_{well_id}.csv"
        if per_well_csv.exists():
            df_pred = pd.read_csv(per_well_csv, parse_dates=[TIME_COL])
            df_w = filter_df_by_well(active_df, well_id)
            fig = plot_forecast_mpl(
                hist_df=df_w[[TIME_COL, TARGET]],
                pred_df=df_pred,
                y_col=TARGET,
                title=f"Well {well_id} – Forecast",
            )
            st.pyplot(fig, use_container_width=True)
            st.dataframe(df_pred.tail(100), use_container_width=True)
        else:
            st.info(
                "No Darts PNG or CSV found for this well. "
                "Run `vm-test --case B --save-plots --export-csv` and/or "
                "`vm-test --case C --save-plots --export-csv` to generate artifacts."
            )

    # ---------------- Two columns: Case B metrics (left) + Case C cumulative plot (right) ----------------
    c1, c2 = st.columns(2)

    # ===== Left: Case B metrics table =====
    if caseB_metrics:
        c1.subheader("Case B (until end of history)")
        metrics_order = ["mql_0_50", "rmse", "mae", "smape", "mic_10_90"]
        dfB = pd.DataFrame(
            [(k, caseB_metrics[k]) for k in metrics_order if k in caseB_metrics],
            columns=["Metric", "Value"],
        )
        if not dfB.empty:
            dfB["Value"] = dfB["Value"].map(lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else x)
            c1.table(dfB)
    else:
        c1.info("No Case B metrics yet. Run `vm-test --case B --export-csv` to produce them.")

    # ===== Right: Case C KPI + cumulative plot =====
    # Helper: build cumulative figure from historical cum and prediction quantiles
    def _plot_cumulative_oil_caseC(hist_df: pd.DataFrame,
                                   pred_df: pd.DataFrame,
                                   *,
                                   time_col: str,
                                   hist_cum_col: str,
                                   title: str):
        """
        hist_df: columns [time_col, hist_cum_col]
        pred_df: columns [time_col, p10, p50, p90]
        Assumes TARGET is a rate. If name contains '_bpd', multiplies predictions by days-in-month.
        Returns: (fig, final_p50_total)
        """
        if hist_df.empty or pred_df.empty:
            return None, None

        # Ensure sorted by time
        hist_df = hist_df.sort_values(time_col).copy()
        pred_df = pred_df.sort_values(time_col).copy()

        # Determine when predictions start
        pred_start = pred_df[time_col].min()

        # Historical cumulative up to prediction start
        hist_up_to_pred = hist_df[hist_df[time_col] <= pred_start].copy()
        start_cum = float(hist_up_to_pred[hist_cum_col].iloc[-1]) if not hist_up_to_pred.empty else 0.0
        hist_segment = hist_up_to_pred[[time_col, hist_cum_col]] if not hist_up_to_pred.empty else pd.DataFrame(columns=[time_col, hist_cum_col])

        # Convert predicted rate → monthly volume if TARGET looks like bpd
        if isinstance(TARGET, str) and "_bpd" in TARGET.lower():
            days = pred_df[time_col].dt.days_in_month.astype(float)
            vol_p10 = pred_df["p10"].astype(float) * days
            vol_p50 = pred_df["p50"].astype(float) * days
            vol_p90 = pred_df["p90"].astype(float) * days
        else:
            vol_p10 = pred_df["p10"].astype(float)
            vol_p50 = pred_df["p50"].astype(float)
            vol_p90 = pred_df["p90"].astype(float)

        # Build cumulative continuations
        cum_p10 = start_cum + vol_p10.cumsum()
        cum_p50 = start_cum + vol_p50.cumsum()
        cum_p90 = start_cum + vol_p90.cumsum()

        # Figure
        fig = go.Figure()

        # 1) Historical cumulative up to prediction start
        if not hist_segment.empty:
            fig.add_trace(go.Scatter(
                x=hist_segment[time_col],
                y=hist_segment[hist_cum_col],
                mode="lines",
                name="Actual cumulative",
                line=dict(width=2)
            ))

        # 2) Prediction envelope and median
        fig.add_trace(go.Scatter(
            x=pd.concat([pred_df[time_col], pred_df[time_col][::-1]]),
            y=pd.concat([cum_p90, cum_p10[::-1]]),
            fill="toself",
            name="P10–P90",
            line=dict(width=0),
            opacity=0.25,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=pred_df[time_col],
            y=cum_p50,
            mode="lines",
            name="P50 cumulative",
            line=dict(width=2)
        ))
        fig.update_layout(
            title=title,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis_title="Date",
            yaxis_title="Cumulative oil (bbl)",
        )

        final_p10_total = float(cum_p10.iloc[-1]) if len(cum_p10) else None
        final_p50_total = float(cum_p50.iloc[-1]) if len(cum_p50) else None
        final_p90_total = float(cum_p90.iloc[-1]) if len(cum_p90) else None
        return fig, final_p10_total, final_p50_total, final_p90_total

    # Build the cumulative plot + KPI for the selected well
    if active_df.empty or well_id is None:
        c2.info("Select a well to view the 5-year cumulative forecast.")
    else:
        per_well_csv = Path(preds_dir or "") / f"pred_well_{well_id}.csv"
        df_hist = filter_df_by_well(active_df, well_id)

        if not per_well_csv.exists():
            c2.info("No Case C CSV found. Run `vm-test --case C --export-csv` to enable the cumulative plot.")
        elif "cum_oil_bbl" not in df_hist.columns:
            c2.warning("Column 'cum_oil_bbl' not found in data; cannot build cumulative plot.")
        else:
            df_pred = pd.read_csv(per_well_csv, parse_dates=[TIME_COL])

            need_cols = [TIME_COL, "p10", "p50", "p90"]
            missing = [c for c in need_cols if c not in df_pred.columns]
            if missing:
                c2.warning(f"Prediction CSV missing required columns: {missing}")
            else:
                fig_cum, final_p10, final_p50, final_p90 = _plot_cumulative_oil_caseC(
                    hist_df=df_hist[[TIME_COL, "cum_oil_bbl"]],
                    pred_df=df_pred[need_cols].dropna(subset=["p50"]),
                    time_col=TIME_COL,
                    hist_cum_col="cum_oil_bbl",
                    title=f"Well {well_id} – 5-yr cumulative (P10/P50/P90)",
                )

                if (fig_cum is not None) and (final_p50 is not None):
                    # --- KPI BOX in LEFT column (under Case B table), height matched to plot ---
                    start_date = df_pred[TIME_COL].min()
                    hist_up_to = df_hist[df_hist[TIME_COL] <= start_date]
                    start_cum_val = float(hist_up_to["cum_oil_bbl"].iloc[-1]) if not hist_up_to.empty else 0.0

                    inc_p10 = final_p10 - start_cum_val if final_p10 is not None else None
                    inc_p50 = final_p50 - start_cum_val if final_p50 is not None else None
                    inc_p90 = final_p90 - start_cum_val if final_p90 is not None else None
                    c1.markdown(
                        f"""
                        <div style="
                            border:1px solid #ddd; border-radius:8px; padding:12px;
                            background-color:#f9f9f9; height:150px;
                            display:flex; align-items:center; justify-content:center;
                        ">
                        <div style="display:flex; gap:16px; width:100%; max-width:720px; justify-content:space-between;">
                            <div style="flex:1; text-align:center;">
                            <div style="font-size:13px; font-weight:600; color:#666; margin-bottom:4px;">5-yr P10 Cum</div>
                            <div style="font-size:20px; font-weight:700; color:#000;">{final_p10:,.0f} bbl</div>
                            <div style="font-size:12px; color:#888;">Δ {inc_p10:,.0f} bbl</div>
                            </div>
                            <div style="flex:1; text-align:center;">
                            <div style="font-size:13px; font-weight:600; color:#666; margin-bottom:4px;">5-yr P50 Cum</div>
                            <div style="font-size:20px; font-weight:700; color:#000;">{final_p50:,.0f} bbl</div>
                            <div style="font-size:12px; color:#888;">Δ {inc_p50:,.0f} bbl</div>
                            </div>
                            <div style="flex:1; text-align:center;">
                            <div style="font-size:13px; font-weight:600; color:#666; margin-bottom:4px;">5-yr P90 Cum</div>
                            <div style="font-size:20px; font-weight:700; color:#000;">{final_p90:,.0f} bbl</div>
                            <div style="font-size:12px; color:#888;">Δ {inc_p90:,.0f} bbl</div>
                            </div>
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Plot in RIGHT column with the same height (350px)
                    c2.plotly_chart(fig_cum, use_container_width=True, height=350)

                    # Optional: incremental P50 over forecast window (caption on right)
                    # start_date = df_pred[TIME_COL].min()
                    # hist_up_to = df_hist[df_hist[TIME_COL] <= start_date]
                    # start_cum_val = float(hist_up_to["cum_oil_bbl"].iloc[-1]) if not hist_up_to.empty else 0.0
                    # inc_p50 = final_p50 - start_cum_val
                    # c2.caption(f"Incremental P50 over forecast window: {inc_p50:,.0f} bbl")
                else:
                    c2.info("Not enough prediction data to build cumulative plot.")

    st.caption(
        "This page shows Darts-rendered PNGs when available "
        "(`caseB_series_###.png`, `caseC_series_###.png`). "
        "Optionally, you can also save by well id "
        "(`caseB_well_<id>.png`, `caseC_well_<id>.png`)."
    )




# ---- 5) Explainability ----
elif section == "Explainability":
    st.header("🪄 Explainability")

    local_dir     = Path(art["explain"]["local_dir"])
    attention_dir = Path(art["explain"]["attention_dir"])
    global_csv    = Path(art["explain"]["global_csv"])

    tabs = st.tabs(["Local Variable Importance", "Attention", "Global Variable Importance"])

    # Load local CSV once
    df_local = pd.read_csv(local_dir / "local_variable_importance.csv") \
               if (local_dir / "local_variable_importance.csv").exists() else pd.DataFrame()

    with tabs[0]:
        st.subheader("Local Variable Importance (per well)")
        if not local_dir.exists():
            st.info("No local explainability yet. Run `vm-explain`.")
        else:
            # show image by well_id (requires `--save-by well` in CLI)
            img_path = local_dir / f"local_variable_importance_well_{well_id}.png"
            if img_path.exists():
                # st.image(str(img_path), caption=f"Local VI – Well {well_id}", use_container_width=True)
                show_image_safe(img_path, "TFT forecasting architecture")
            else:
                st.caption("No local VI image for this well yet. Run `vm-explain --save-by well`.")

            # CSV table filtered by well if available
            csv_path = local_dir / "local_variable_importance.csv"
            if csv_path.exists():
                df_local = pd.read_csv(csv_path)
                if "well_id" in df_local.columns:
                    st.dataframe(df_local.query("well_id == @well_id"), use_container_width=True, height=360)
                else:
                    st.dataframe(df_local, use_container_width=True, height=360)

    with tabs[1]:
        st.subheader("Temporal Attention")
        if not attention_dir.exists():
            st.info("No attention plots yet. Run `vm-explain`.")
        else:
            attn_all   = attention_dir / f"attention_well_{well_id}.png"
            attn_heat  = attention_dir / f"attention_heatmap_well_{well_id}.png"

            shown_any = False
            if attn_all.exists():
                st.image(str(attn_all), caption=f"Attention (all) – Well {well_id}", use_container_width=True)
                shown_any = True
            if attn_heat.exists():
                st.image(str(attn_heat), caption=f"Attention (heatmap) – Well {well_id}", use_container_width=True)
                shown_any = True

            if not shown_any:
                st.caption("No attention image for this well yet. Run `vm-explain --save-by well --attention-types all,heatmap`.")

    with tabs[2]:
        st.subheader("Pseudo-global Variable Importance")
        if Path(global_csv).exists():
            df_global = pd.read_csv(global_csv)
            st.dataframe(df_global, use_container_width=True, height=420)

            block = st.selectbox("Block", sorted(df_global["block"].unique().tolist()))
            df_b = df_global[df_global["block"] == block].nlargest(20, "importance_mean")

            # Sort descending so largest importance is on top
            df_b = df_b.sort_values("importance_mean", ascending=True)

            fig = px.bar(
                df_b,
                x="importance_mean",
                y="feature",
                orientation="h",
                title=f"Top features – {block}",
            )
            fig.update_layout(
                yaxis=dict(
                    categoryorder="array",
                    categoryarray=df_b["feature"].tolist(),  # keep descending order
                    tickfont=dict(size=18)  # increase y-axis label font size
                ), 
                height = 700
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pseudo-global VI CSV yet. Run `vm-explain`.")

# ========== Footer ==========
st.markdown("---")
st.caption(
    "© Alexis Ortega • Vaca Muerta TFT • Streamlit dashboard • https://github.com/alexort74/ttf-vaca-muerta-oil-forecasting"
)
