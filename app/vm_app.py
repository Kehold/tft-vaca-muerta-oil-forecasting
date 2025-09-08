from __future__ import annotations
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


# Darts objects are optional – only needed if you later add “live” model actions
try:
    from darts.models import TFTModel  # noqa: F401
    from darts.timeseries import TimeSeries  # noqa: F401
except Exception:
    TFTModel = None
    TimeSeries = None

# ---- internal modules (your package) ----
from vm_tft.cfg import (
    ALL_FILE, TEST_FILE, ARTIFACTS,
    TIME_COL, GROUP_COL, TARGET, FREQ,
)
from vm_tft.io_utils import ensure_dir  # noqa: F401

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

def list_artifacts() -> dict:
    base = Path(ARTIFACTS)
    return {
        "models": {
            "best_from_search": base / "models" / "model_best_log_baseline.pt",
            "best_manual": base / "models" / "tft_best_manual.pt",
        },
        "cv": {
            "overall_metrics": base / "metrics" / "cv_overall_log_baseline.json",
            "per_series_window": base / "metrics" / "cv_metrics_per_series_window_log_baseline.csv",
        },
        "search": {
            "trials_csv": base / "metrics" / "trials_log_baseline.csv",
            "best_config": base / "metrics" / "search_best_log_baseline.json",
        },
        "test": {
            "caseA_metrics": base / "predictions" / "caseA_overall_mean.json",
            "caseB_metrics": base / "predictions" / "caseB_overall_mean.json",
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
#     <p style='margin: 0; font-size: 12px;'>© 2025 Alexis Ortega | <a href="https://github.com/alexort74"></a></p>
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
    st.header("🏠 Home")

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Goals / Scope")
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
        st.image("data/images/tft.png", use_container_width=True, caption="TFT forecasting architecture")


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

    preds_dir = art["test"]["predictions_dir"]
    caseA_metrics = read_metrics_json(art["test"]["caseA_metrics"])
    caseB_metrics = read_metrics_json(art["test"]["caseB_metrics"])

    # metrics tables (formatted) – same as you already implemented
    c1, c2, _ = st.columns(3)
    if caseA_metrics:
        c1.subheader("Case A (fixed horizon)")
        dfA = pd.DataFrame(
            [(k, caseA_metrics[k]) for k in ["mql_0_50", "rmse", "mae", "smape", "mic_10_90"] if k in caseA_metrics],
            columns=["Metric", "Value"],
        )
        if not dfA.empty:
            dfA["Value"] = dfA["Value"].map(lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else x)
            c1.table(dfA)
    if caseB_metrics:
        c2.subheader("Case B (until end of history)")
        dfB = pd.DataFrame(
            [(k, caseB_metrics[k]) for k in ["mql_0_50", "rmse", "mae", "smape", "mic_10_90"] if k in caseB_metrics],
            columns=["Metric", "Value"],
        )
        if not dfB.empty:
            dfB["Value"] = dfB["Value"].map(lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else x)
            c2.table(dfB)

    st.subheader("Per-well visualization (Darts plots if available)")
    if active_df.empty or well_id is None:
        st.info("Select a well to view predictions.")
        st.stop()

    series_idx = _series_index_for_well(df_test, well_id)  # use TEST ordering to mirror test_predict.py

    # Try to show pre-rendered Darts PNGs first (exact pred.plot style)
    colA, colB = st.columns(2)
    shown_any = False

    for case_tag, col in [("caseA", colA), ("caseB", colB)]:
        png_path = _find_prediction_image(Path(preds_dir or ""), well_id, series_idx, case_tag)
        if png_path is not None:
            col.image(str(png_path), caption=f"{case_tag.upper()} – Well {well_id}", use_container_width=True)
            shown_any = True

    # If no PNGs found, fall back to CSV-driven Matplotlib replica for the selected well
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
                "Run `vm-test-predict --save-plots --export-csv` to generate artifacts."
            )

    st.caption(
        "Tip: `vm-test-predict` currently saves PNGs as `caseA_series_###.png` / `caseB_series_###.png`. "
        "This page resolves those by mapping the selected well to the TEST series index. "
        "If you prefer file names by well ID, also save `caseA_well_<id>.png` / `caseB_well_<id>.png`."
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
                st.image(str(img_path), caption=f"Local VI – Well {well_id}", use_container_width=True)
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
                    categoryarray=df_b["feature"].tolist()  # keep descending order
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pseudo-global VI CSV yet. Run `vm-explain`.")

# ========== Footer ==========
st.markdown("---")
st.caption(
    "© Alexis Ortega • Vaca Muerta TFT • Streamlit dashboard • https://github.com/alexort74"
)
