from __future__ import annotations

# --- make local src/ importable on Streamlit Cloud ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root (app/ is one level below)
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# -----------------------------------------------------

import json
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Darts objects are optional – only needed if you later add “live” model actions
try:
    from darts.models import TFTModel  # noqa: F401
    from darts.timeseries import TimeSeries
except Exception:
    TFTModel = None
    TimeSeries = None

# ---- internal modules (your package) ----
from vm_tft.cfg import (
    ALL_FILE, TEST_FILE, ARTIFACTS,
    TIME_COL, GROUP_COL, TARGET, FREQ,
)
from vm_tft.io_utils import ensure_dir  # noqa: F401

from PIL import Image, UnidentifiedImageError

# =========================
# Page Settings
# =========================
st.set_page_config(
    page_title="Vaca Muerta – TFT Dashboard",
    layout="wide",
    page_icon="🛢️",
)

# ---------- Light CSS polish ----------
st.markdown("""
<style>
.block-container { padding-top: 1.1rem; }

/* Section "cards" */
.section-card {
  background: var(--secondary-background-color);
  border: 1px solid #E5E7EB;
  border-radius: 12px;
  padding: 14px 16px; margin-bottom: 12px;
}

/* Metrics: stronger number */
[data-testid="stMetricValue"] { font-weight: 700; font-size: 1.25rem; }

/* Tables */
thead tr th { background:#fafbff !important; font-weight:600 !important; }
tbody tr:nth-child(even) { background-color:#fbfbfc !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# Global Plot Style (uniform across app)
# =========================
def _set_plot_style():
    px.defaults.template = "simple_white"
    px.defaults.color_discrete_sequence = [
        "#2E86AB",  # blue
        "#F18F01",  # orange
        "#C73E1D",  # red
        "#4ECDC4",  # teal
        "#6C757D",  # gray
    ]
    vm_tpl = go.layout.Template()
    vm_tpl.layout.font = dict(
        family="Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
        size=14,
    )
    vm_tpl.layout.margin = dict(l=10, r=10, t=40, b=10)
    pio.templates["vm_white"] = vm_tpl
    pio.templates.default = "simple_white+vm_white"

_set_plot_style()

def unify_figure_layout(fig: go.Figure, *, title: Optional[str] = None, legend_bottom: bool = True, height: Optional[int] = None):
    if title:
        fig.update_layout(title=title)
    fig.update_layout(
        margin=dict(l=8, r=8, t=48, b=8),
        xaxis=dict(showgrid=True, gridcolor="#EEF2F7"),
        yaxis=dict(showgrid=True, gridcolor="#EEF2F7"),
    )
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0))
    if height:
        fig.update_layout(height=height)
    return fig

# =========================
# Utilities & Cache
# =========================

def show_image_safe(rel_path: str | Path, caption: str = "") -> None:
    img_path = ROOT / str(rel_path)
    if not img_path.exists():
        st.warning(f"Image not found: {img_path} (check name & case).")
        return
    try:
        img = Image.open(img_path)
    except (UnidentifiedImageError, OSError) as e:
        st.error(f"Could not open image {img_path.name}: {e}")
        return
    st.image(img, use_column_width=True, caption=caption)

@st.cache_data(show_spinner=False)
def load_csv(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=[TIME_COL])

@st.cache_data(show_spinner=False)
def read_metrics_json(path: Path | str) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def read_metrics_csv(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

# ----- ACF / PACF / STL -----
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

@st.cache_data(show_spinner=False)
def compute_pacf_for_well(df, well_id, target_col, nlags=36, difference=False):
    s = (
        df.loc[df[GROUP_COL] == well_id, target_col]
        .dropna().astype(float).to_numpy()
    )
    if difference:
        s = np.diff(s)
    if len(s) < 3:
        return np.array([]), np.array([]), 0.0
    max_allowed = max(1, (len(s) // 2) - 1)
    nlags = int(min(nlags, max_allowed))
    if nlags < 1:
        return np.array([]), np.array([]), 0.0
    pacf_vals = sm_pacf(s, nlags=nlags, method="ywadjusted")
    lags = np.arange(len(pacf_vals))
    ci_half = 1.96 / np.sqrt(len(s))
    return lags, pacf_vals, ci_half

@st.cache_data(show_spinner=False)
def compute_stl_for_well(df: pd.DataFrame, well_id: int, y_col: str, period: int):
    col = GROUP_COL if GROUP_COL in df.columns else "well_id"
    df_w = (df.loc[df[col] == well_id, [TIME_COL, y_col]]
              .dropna()
              .sort_values(TIME_COL)
              .set_index(TIME_COL))
    if df_w.empty or df_w.shape[0] < 2 * period:
        return None
    s = df_w[y_col].astype(float)
    res = STL(s, period=period, robust=True).fit()
    return {
        "index": s.index,
        "observed": s.values,
        "trend": res.trend.values,
        "seasonal": res.seasonal.values,
        "resid": res.resid.values,
    }

def plot_stl_small_multiples(stl_dict, title_prefix="STL Decomposition"):
    fig = go.Figure()
    if stl_dict is None:
        return unify_figure_layout(fig, title=title_prefix, height=700)
    idx = stl_dict["index"]
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
    )
    fig.add_trace(go.Scatter(x=idx, y=stl_dict["observed"], mode="lines", name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=stl_dict["trend"],    mode="lines", name="Trend"),    row=2, col=1)
    fig.add_trace(go.Scatter(x=idx, y=stl_dict["seasonal"], mode="lines", name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=idx, y=stl_dict["resid"],    mode="lines", name="Residual"), row=4, col=1)
    fig.add_hline(y=0.0, line=dict(width=1, color="black", dash="dot"), row=4, col=1)
    fig.update_layout(height=700, showlegend=False)
    return unify_figure_layout(fig, title=title_prefix, height=700)

# ----- Main TS plot + overlays -----
def plot_series_with_overlays(df_w: pd.DataFrame, y_col: str, title: str,
                              roll_win: int, show_mean: bool, show_median: bool) -> go.Figure:
    fig = go.Figure()
    if df_w.empty:
        return unify_figure_layout(fig, title=title, height=360)

    fig.add_trace(go.Scatter(
        x=df_w[TIME_COL], y=df_w[y_col], mode="lines", name=y_col
    ))
    if show_mean and roll_win > 1:
        rm = df_w[y_col].rolling(roll_win, min_periods=max(1, roll_win // 2)).mean()
        fig.add_trace(go.Scatter(
            x=df_w[TIME_COL], y=rm, mode="lines", name=f"Rolling mean ({roll_win})",
            line=dict(width=2)
        ))
    if show_median and roll_win > 1:
        rmed = df_w[y_col].rolling(roll_win, min_periods=max(1, roll_win // 2)).median()
        fig.add_trace(go.Scatter(
            x=df_w[TIME_COL], y=rmed, mode="lines", name=f"Rolling median ({roll_win})",
            line=dict(width=2, dash="dot")
        ))
    return unify_figure_layout(fig, title=title, height=360)

# ----- Forecast band (p10/p50/p90) helper -----
def plot_forecast_band_p10_p50_p90(
    hist: pd.DataFrame,
    pred: pd.DataFrame,
    y_col: str,
    title: str = "Forecast",
):
    band_color = "rgba(241, 143, 1, 0.22)"  # #F18F01 w/ alpha
    line_color_p50 = "#F18F01"
    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist[TIME_COL], y=hist[y_col],
            mode="lines", name="Actual (history)"
        ))
    if not pred.empty and {"p10", "p50", "p90"}.issubset(pred.columns):
        fig.add_trace(go.Scatter(
            x=pd.concat([pred[TIME_COL], pred[TIME_COL][::-1]]),
            y=pd.concat([pred["p90"], pred["p10"][::-1]]),
            fill="toself",
            name="P10–P90",
            line=dict(width=0),
            fillcolor=band_color,
            hoverinfo="skip",
        ))
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
    return unify_figure_layout(fig, title=title, height=360)

# ----- Baselines: overall means loader -----
@st.cache_data(show_spinner=False)
def load_baseline_overall_means(path: Path | str) -> pd.DataFrame:
    """
    Reads baselines_overall.json (model -> {rmse_mean, rmse_median, ...})
    and returns a tidy DataFrame with: model, rmse, mae, smape (means).
    Falls back to medians if means are missing.
    """
    data = read_metrics_json(path) or {}
    if not isinstance(data, dict) or not data:
        return pd.DataFrame(columns=["model", "rmse", "mae", "smape"])
    df = pd.DataFrame.from_dict(data, orient="index").reset_index().rename(columns={"index": "model"})
    for m in ["rmse", "mae", "smape"]:
        if f"{m}_mean" in df.columns:
            df[m] = df[f"{m}_mean"]
        elif f"{m}_median" in df.columns:
            df[m] = df[f"{m}_median"]
        else:
            df[m] = np.nan
    return df[["model", "rmse", "mae", "smape"]]

# ----- HW quick overlay calculation for the Explorer -----
@st.cache_data(show_spinner=False)
def hw_forecast(df_well: pd.DataFrame, target_col: str, horizon: int = 12):
    """
    Holt-Winters baseline: trend-only (additive), no seasonality (common for monthly oil rates).
    Returns DataFrame with [TIME_COL, actual, baseline_forecast].
    """
    ts = df_well[[TIME_COL, target_col]].dropna().sort_values(TIME_COL)
    y = ts[target_col].astype(float).values
    if len(y) < 6:
        last = y[-1] if len(y) else 0.0
        yhat_in = np.full_like(y, last, dtype=float)
        yhat_out = np.full(horizon, last, dtype=float)
    else:
        model = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        yhat_in = fit.fittedvalues
        yhat_out = fit.forecast(horizon)
    freq = pd.infer_freq(ts[TIME_COL]) or "MS"
    last_date = ts[TIME_COL].iloc[-1]
    future_idx = pd.date_range(last_date, periods=horizon+1, freq=freq)[1:]
    df = pd.DataFrame({
        TIME_COL: list(ts[TIME_COL]) + list(future_idx),
        "actual": list(ts[target_col]) + [np.nan]*horizon,
        "baseline_forecast": list(yhat_in) + list(yhat_out)
    })
    return df

# ----- Data slicing helpers -----
def list_artifacts() -> dict:
    base = Path(ARTIFACTS)
    return {
        "models": {
            "best_from_search": base / "models" / "model_best_bo_baseline_scaler.pt",
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
            "caseB_metrics": base / "predictions" / "caseB_overall_mean_bo_baseline.json",
            "predictions_dir": base / "predictions" / "plots",
        },
        "explain": {
            "local_dir": base / "explain" / "local",
            "attention_dir": base / "explain" / "attention",
            "global_csv": base / "explain" / "global" / "global_variable_importance_pseudo.csv",
        },
        "baselines": {
            "overall_json": base / "metrics" / "baselines_overall.json",
            "per_series_csv": base / "metrics" / "baselines_per_series.csv",
        },
    }

def get_well_ids(df: pd.DataFrame) -> List[int]:
    if df.empty:
        return []
    col = GROUP_COL if GROUP_COL in df.columns else "well_id"
    return sorted(df[col].dropna().unique().tolist())

def filter_df_by_well(df: pd.DataFrame, well_id: int) -> pd.DataFrame:
    col = GROUP_COL if GROUP_COL in df.columns else "well_id"
    return df[df[col] == well_id].sort_values(TIME_COL)

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

# ---------- Well selector by name (keeps well_id for the rest) ----------
def select_well_by_name(active_df: pd.DataFrame) -> Optional[int]:
    """
    Show a sidebar selectbox with well names. Returns the underlying well_id.
    Falls back to selecting by well_id if no 'well_name' column exists.
    """
    if active_df.empty:
        return None

    # If we have well_name, map to id; else, pick by id directly
    if "well_name" in active_df.columns:
        # Build a unique (well_id, well_name) map
        tmp = active_df[[GROUP_COL, "well_name"]].dropna().drop_duplicates()
        if tmp.empty:
            return None
        tmp = tmp.sort_values(["well_name", GROUP_COL])
        labels = [f"{row['well_name']}  ·  id={int(row[GROUP_COL])}" for _, row in tmp.iterrows()]
        ids    = tmp[GROUP_COL].astype(int).tolist()

        # best-effort default: last selected, or first item
        default_idx = 0
        if "well_sel_label" in st.session_state:
            try:
                default_idx = labels.index(st.session_state["well_sel_label"])
            except ValueError:
                default_idx = 0

        sel_label = st.sidebar.selectbox("Well", labels, index=default_idx if labels else 0, key="well_name_picker")
        st.session_state["well_sel_label"] = sel_label

        # Parse back id (split on last '=' and cast)
        try:
            chosen_id = int(sel_label.split("id=")[-1])
        except Exception:
            chosen_id = ids[0]
        return chosen_id

    # Fallback: select by id as before
    well_ids = get_well_ids(active_df)
    return st.sidebar.selectbox("Well (id)", well_ids, index=0 if well_ids else None, key="well_id_picker")

# =========================
# Page guides & tiny insights
# =========================

def page_guide(title: str, what: str, why: str, good_bad: str) -> None:
    """Compact explainer, consistent across pages."""
    with st.expander(f"🧭 What is this page?  —  {title}", expanded=False):
        st.markdown(
            f"""
- **What am I looking at?** {what}
- **Why is it relevant?** {why}
- **Is it good or bad?** {good_bad}
            """
        )

def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%" if np.isfinite(x) else "–"

def insight_time_series(df_w: pd.DataFrame, y_col: str) -> str:
    """Quick trend/volatility signal from the last ~24 points."""
    s = df_w[[TIME_COL, y_col]].dropna().sort_values(TIME_COL)
    if len(s) < 6:
        return "Not enough history to assess trend/volatility."
    last_n = min(24, len(s))
    y = s[y_col].to_numpy()[-last_n:]
    x = np.arange(last_n)
    slope = np.polyfit(x, y, 1)[0]
    rel_slope = slope / (np.mean(y) + 1e-9)  # per-step vs mean
    vol = np.std(y) / (np.mean(y) + 1e-9)
    trend_txt = (
        "roughly flat" if abs(rel_slope) < 0.002
        else ("upward" if slope > 0 else "downward")
    )
    return f"Recent trend: **{trend_txt}**; volatility (σ/μ): **{_fmt_pct(100*vol)}**."

def insight_acf(lags: np.ndarray, vals: np.ndarray, ci: float, expect_season_m=12) -> str:
    if lags.size == 0:
        return "No ACF computed."
    sig = [int(l) for l, v in zip(lags, vals) if l > 0 and abs(v) > ci]
    if not sig:
        return "No strong autocorrelation beyond lag 0 → weak persistence."
    season_note = ""
    if expect_season_m in sig:
        season_note = f" Notably, lag {expect_season_m} stands out → possible yearly pattern."
    return f"Significant autocorrelation at lags {sig[:5]} (first few).{season_note}"

def insight_pacf(lags: np.ndarray, vals: np.ndarray, ci: float) -> str:
    if lags.size == 0:
        return "No PACF computed."
    sig = [int(l) for l, v in zip(lags, vals) if l > 0 and abs(v) > ci]
    if not sig:
        return "No strong partial autocorrelation → limited direct AR structure."
    return f"Strong partial autocorrelation at lags {sig[:4]} → potential AR order hints."

def insight_cv_overall(overall: dict | None) -> str:
    if not overall:
        return "Overall CV metrics not found."
    rmse = overall.get("rmse", np.nan)
    mae  = overall.get("mae", np.nan)
    sm   = overall.get("smape", np.nan)
    tail = "heavy tails/outliers" if (rmse - mae) > 0.25 * (mae + 1e-9) else "balanced errors"
    return f"Typical error MAE≈**{mae:.1f}**, RMSE≈**{rmse:.1f}** ({tail}); sMAPE≈**{sm:.1f}%**."

def insight_baselines_table(df_overall: pd.DataFrame) -> str:
    if df_overall.empty or "model" not in df_overall.columns:
        return "No baseline summary yet."
    if "rmse" not in df_overall.columns:
        return "Baseline table missing RMSE."
    df = df_overall.set_index("model")
    if "TFT" not in df.index or df.drop(index=["TFT"], errors="ignore").empty:
        return "Waiting for both TFT and baselines to compare."
    tft = df.loc["TFT", "rmse"]
    baseline_block = df.drop(index=["TFT"], errors="ignore")["rmse"]
    best_b = baseline_block.idxmin()
    best_rmse = float(baseline_block.min())
    gap = (best_rmse - tft) / (tft + 1e-9) * 100
    if best_rmse < tft:
        return f"Best baseline **{best_b}** beats TFT by **{abs(gap):.1f}%** (RMSE)."
    return f"**TFT** beats best baseline (**{best_b}**) by **{abs(gap):.1f}%** (RMSE)."


# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("Vaca Muerta – TFT")
section = st.sidebar.radio(
    "Navigate",
    ["Home", "Data Explorer", "Model CV", "Baselines", "Randomized Search", "Test Predictions", "Explainability"],
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

# ✅ Select by well_name, keep using well_id everywhere else
well_id = select_well_by_name(active_df)

st.sidebar.markdown("---")

# Small context ribbon
with st.container():
    a, b, c = st.columns([1.2, 1, 1])
    a.markdown(f"**Dataset:** `{default_set}`")
    b.markdown(f"**Well ID:** `{well_id if well_id is not None else '—'}`")
    c.markdown(f"**Target:** `{TARGET}`")

# =========================
# Sections
# =========================

# ---- 0) Home ----

if section == "Home":
    page_guide(
        "Home",
        what="High-level description of the project and how to navigate the dashboard.",
        why="Sets context so viewers know where to find CV results, baselines, predictions, and explainability.",
        good_bad="A clear path from data → model → selection → test → explainability indicates a healthy, reproducible workflow."
    )

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
        show_image_safe("data/images/tft_paper.png", "TFT forecasting architecture")

# ---- 1) Data Explorer ----
elif section == "Data Explorer":
    
    page_guide(
        "Data Explorer",
        what="Monthly oil-rate history for the selected well with optional overlays (rolling stats, HW baseline), plus ACF/PACF and STL (optional).",
        why="Early sense of trend, volatility and memory structure helps interpret model performance and pick sensible baselines.",
        good_bad="Stable, gently trending series with short memory are easier; long tails in ACF or frequent step changes suggest operational effects to watch."
    )
    
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
        # --- Plot Selection ---
        st.markdown("**Plot Selection**")
        acf_kind = st.radio(
            "Correlation view", ("ACF", "PACF"),
            index=0,
            help="Show either autocorrelation (ACF) or partial autocorrelation (PACF).",
            key="acf_kind",
        )
        max_lags = st.slider("Max lags", 6, 60, 36, step=6, key="acf_lags")
        do_diff  = st.checkbox(
            "Difference series (Δy) before ACF/PACF",
            value=False,
            key="acf_diff",
            help="Compute correlations on first differences.",
        )

        st.markdown("---")
        # --- Overlays ---
        st.markdown("**Overlays**")
        roll_win   = st.slider("Rolling window (months)", 3, 24, 6, 1, key="roll_win")
        show_mean   = st.checkbox("Show rolling mean",   value=True,  key="roll_mean")
        show_median = st.checkbox("Show rolling median", value=False, key="roll_median")
        show_hw     = st.checkbox(
            "Show Holt–Winters baseline (HW)",
            value=False,
            key="overlay_hw",
            help="Quick trend-only baseline overlay; aggregate metrics live in the Baselines page.",
        )

        st.markdown("---")
        st.markdown("**STL decomposition**")
        stl_on = st.checkbox(
            "Enable STL",
            value=False,
            key="stl_on",
            help="Requires enough data; ~2×period recommended.",
        )
        stl_period = st.slider("STL period (months)", 6, 24, 12, 1, key="stl_period")

    # ---------- Main layout ----------
    left, right = st.columns([2, 1], gap="large")

    # Main time series with overlays (+ optional HW overlay)
    with left:
        st.subheader("Production time series")
        df_w = filter_df_by_well(active_df, well_id)
        if df_w.empty:
            st.info("Selected well has no rows.")
        else:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            fig_ts = plot_series_with_overlays(
                df_w, TARGET, f"Well {well_id} – {TARGET}",
                roll_win=roll_win, show_mean=show_mean, show_median=show_median
            )
            # HW overlay on same figure
            if show_hw:
                try:
                    df_f = hw_forecast(df_w, TARGET, horizon=12)
                    fig_ts.add_trace(go.Scatter(
                        x=df_f[TIME_COL],
                        y=df_f["baseline_forecast"],
                        mode="lines",
                        name="HW baseline",
                        line=dict(width=2, dash="dot")
                    ))
                except Exception as e:
                    st.caption(f"HW overlay unavailable for this well: {e}")
            st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})
            st.caption(insight_time_series(df_w, TARGET))
            st.caption("Note: HW overlay is a quick visual comparison. Full baseline metrics are in the **Baselines** page.")
            st.markdown('</div>', unsafe_allow_html=True)

    # ACF / PACF panel
    with right:
        # st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if acf_kind == "PACF":
            lags, vals, ci = compute_pacf_for_well(active_df, well_id, TARGET, max_lags, do_diff)
            st.subheader("Partial Autocorrelation (PACF)")
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            actual_nlags = max(0, len(vals) - 1)
            if actual_nlags and actual_nlags < max_lags:
                st.caption(f"Note: PACF max lags capped to {actual_nlags} due to sample-size limits.")
        else:
            lags, vals, ci = compute_acf_for_well(active_df, well_id, TARGET, max_lags, do_diff)
            st.subheader("Autocorrelation (ACF)")
            st.markdown('<div class="section-card">', unsafe_allow_html=True)

        if len(vals):
            fig = go.Figure()
            fig.add_vline(x=0, line_width=1)
            fig.add_hrect(y0=-ci, y1=ci, opacity=0.15, line_width=0)
            fig.add_bar(x=lags, y=vals)
            st.plotly_chart(unify_figure_layout(fig, height=300), use_container_width=True)
            
            if acf_kind == "PACF":
                st.caption(insight_pacf(lags, vals, ci))
            else:
                st.caption(insight_acf(lags, vals, ci))
        else:
            st.info("Not enough data points to compute correlations for this well.")

        # Quick summary
        if not df_w.empty:
            n = len(df_w)
            start = df_w[TIME_COL].min()
            end = df_w[TIME_COL].max()
            mean_v = df_w[TARGET].mean()
            std_v = df_w[TARGET].std()
            st.caption(
                f"**Summary** — n={n:,} • span: {start.date()} → {end.date()} • "
                f"mean={mean_v:,.1f} • std={std_v:,.1f}"
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # STL (full width) if enabled
    if stl_on:
        st.subheader("Seasonal decomposition (STL)")
        stl_res = compute_stl_for_well(active_df, well_id, TARGET, period=stl_period)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if stl_res is None:
            st.info("Not enough data for STL (need at least ~2×period).")
        else:
            fig_stl = plot_stl_small_multiples(stl_res, title_prefix=f"STL (Well {well_id}, period={stl_period})")
            st.plotly_chart(fig_stl, use_container_width=True, config={"displayModeBar": False})
            
            if stl_res is not None:
                # relative magnitudes of components
                obs = np.asarray(stl_res["observed"], float)
                trend = np.asarray(stl_res["trend"], float)
                seas = np.asarray(stl_res["seasonal"], float)
                resid = np.asarray(stl_res["resid"], float)
                def _share(a): 
                    v = np.var(a)
                    return v / (np.var(obs)+1e-9)
                st.caption(
                    f"STL shares — trend≈{_share(trend):.2f}, seasonal≈{_share(seas):.2f}, residual≈{_share(resid):.2f}. "
                    "Higher trend/seasonal shares → more predictable structure."
                )
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Raw table (filtered to the selected well)
    st.subheader("Raw data")
    df_w = filter_df_by_well(active_df, well_id)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if df_w.empty:
        st.info("Selected well has no rows.")
    else:
        st.dataframe(df_w.sort_values(TIME_COL).head(200), use_container_width=True, height=320)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- 2) Model CV ----
elif section == "Model CV":
    st.header("🧪 Cross-Validation Results")
    page_guide(
        "Model CV",
        what="Rolling-window validation metrics (overall + per series/window) produced with the same policy used later for selection.",
        why="Verifies generalization and stability before testing on unseen wells.",
        good_bad="Lower RMSE/MAE is better; sMAPE < 20–25% is often decent. Big RMSE−MAE gap implies outliers impacting squared error."
    )

    with st.sidebar.expander("Model CV controls", expanded=True):
        cv_hist_metric = st.radio(
            "Histogram metric",
            ["RMSE", "MAE", "sMAPE"],
            index=0,
            help="Choose which metric to visualize in the histogram."
        )

    overall_path = art["cv"]["overall_metrics"]
    per_series_path = art["cv"]["per_series_window"]

    overall = read_metrics_json(overall_path)
    per_series = read_metrics_csv(per_series_path)

    metric_map = {"RMSE": "rmse", "MAE": "mae", "sMAPE": "smape"}
    selected_label = locals().get("cv_hist_metric", "RMSE")
    hist_col = metric_map.get(selected_label, "rmse")

    left, right = st.columns([1.2, 1.8], gap="large")

    with left:
        st.subheader("Overall metrics")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)
        
        if overall:
            st.caption(insight_cv_overall(overall))

    with right:
        st.subheader(f"{selected_label} distribution")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if per_series.empty:
            st.info("No per-series/window CSV found.")
        else:
            fig = px.histogram(
                per_series,
                x=hist_col,
                nbins=40,
                title=f"{selected_label} distribution",
            )
            fig.update_traces(opacity=0.75)
            st.plotly_chart(unify_figure_layout(fig, height=360), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Per-series / window metrics")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if per_series.empty:
        st.info("No per-series/window CSV found.")
    else:
        st.dataframe(per_series, use_container_width=True, height=420)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- 3) Baselines ----
elif section == "Baselines":
    st.header("🪵 Baselines vs TFT")
    
    page_guide(
        "Baselines vs TFT",
        what="Simple, robust baselines (NaiveDrift, ETS/HW, Theta) evaluated with the same CV as TFT.",
        why="Contextualizes how much value TFT adds over non-neural methods.",
        good_bad="If TFT only ties baselines, revisit features/validation; a healthy gap (lower errors) supports using TFT in production."
    )

    bl_overall_path = art["baselines"]["overall_json"]
    bl_per_series_path = art["baselines"]["per_series_csv"]

    # Overall (TFT + baselines)
    tft_overall = read_metrics_json(art["cv"]["overall_metrics"]) or {}
    df_tft = pd.DataFrame([{
        "model": "TFT",
        "rmse":  tft_overall.get("rmse",  np.nan),
        "mae":   tft_overall.get("mae",   np.nan),
        "smape": tft_overall.get("smape", np.nan),
    }])

    df_base = load_baseline_overall_means(bl_overall_path)
    df_overall = pd.concat([df_tft, df_base], ignore_index=True)

    st.subheader("Overall metrics (lower is better)")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if df_overall.empty:
        st.info("No baseline summary found. Make sure `baselines_overall.json` exists.")
    else:
        df_overall[["rmse","mae","smape"]] = df_overall[["rmse","mae","smape"]].astype(float).round(3)
        # Optional delta vs TFT
        try:
            tft_rmse = float(df_tft["rmse"].iloc[0])
            df_overall["Δ RMSE vs TFT"] = (df_overall["rmse"] - tft_rmse).map(lambda x: f"{x:+,.3f}")
        except Exception:
            pass
        st.table(df_overall.set_index("model"))
        
        if not df_overall.empty:
            st.caption(insight_baselines_table(df_overall))
        
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Distributions across wells/windows
    st.subheader("Error distribution across wells/windows")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    bl_per = read_metrics_csv(bl_per_series_path)
    if bl_per.empty:
        st.caption("No per-window baseline CSV found. "
                   f"Expected at: `{bl_per_series_path}`.")
    else:
        metric = st.radio("Metric", ["rmse", "mae", "smape"], index=0, horizontal=True, key="bl_metric")
        fig = px.histogram(
            bl_per,
            x=metric,
            color="model",
            barmode="overlay",
            nbins=40,
            title=f"{metric.upper()} distributions – baselines"
        )
        fig.update_traces(opacity=0.65)
        fig.update_layout(bargap=0.03)
        st.plotly_chart(unify_figure_layout(fig, height=360), use_container_width=True)
        st.caption("Overlap of colored histograms: tighter and left-shifted ⇒ more consistent, lower error.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Per-well quick look (optional if Darts available)
    st.subheader("Per-well quick look (optional)")
    st.caption("Runs only if Darts is available here.")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if TimeSeries is None:
        st.info("Darts not available in this environment.")
    else:
        from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing as ETS
        from darts.models.forecasting.theta import Theta
        try:
            # Darts >= 0.30: use enums
            from darts.utils.utils import ModelMode as MM, SeasonalityMode as SM
        except Exception:
            MM = SM = None  # very old Darts fallback

        def make_hw_model():
            if MM is not None and SM is not None:
                return ETS(trend=MM.ADDITIVE, seasonal=SM.NONE, seasonal_periods=None)
            return ETS(trend=None, seasonal=None, seasonal_periods=None)

        def make_theta_model(ts_len: int, m: int = 12):
            period = m if ts_len >= 2 * m else None
            if SM is not None:
                return Theta(season_mode=SM.ADDITIVE, seasonality_period=period)
            return Theta(season_mode=None, seasonality_period=None)

        df_w = filter_df_by_well(active_df, well_id)
        if df_w.empty:
            st.info("Select a well with data.")
        else:
            ts = TimeSeries.from_dataframe(
                df_w[[TIME_COL, TARGET]].dropna(),
                time_col=TIME_COL,
                value_cols=TARGET,
                freq=FREQ,
            )
            h = st.slider("Horizon (months)", 6, 24, 12, 1, key="bl_demo_h")
            m1 = make_hw_model()
            m2 = make_theta_model(len(ts), m=12)
            f1 = m1.fit(ts).predict(h)
            f2 = m2.fit(ts).predict(h)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_w[TIME_COL], y=df_w[TARGET], mode="lines", name="Actual"))
            fig.add_trace(go.Scatter(x=f1.time_index, y=f1.values().flatten(), mode="lines", name="HW / ETS"))
            fig.add_trace(go.Scatter(x=f2.time_index, y=f2.values().flatten(), mode="lines", name="Theta"))
            fig = unify_figure_layout(fig, title=f"Well {well_id} – short baseline forecast", height=360)
            st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- 4) Randomized Search ----
elif section == "Randomized Search":
    st.header("🔍 Randomized Search")

    trials_csv = art["search"]["trials_csv"]
    best_cfg   = art["search"]["best_config"]

    trials = read_metrics_csv(trials_csv)

    exclude_cols = {
        "mql_0_50", "rmse", "mae", "smape", "mic_10_90",
        "input_chunk_length", "output_chunk_length",
    }
    hp_cols = [c for c in trials.columns if c not in exclude_cols] if not trials.empty else []

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
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.dataframe(trials, use_container_width=True, height=420)
        st.markdown('</div>', unsafe_allow_html=True)

        if x_hp is not None and color_hp is not None:
            fig = px.scatter(
                trials,
                x=x_hp,
                y=metric_opt,
                color=color_hp,
                size=metric_opt,
                size_max=18,
                hover_data=trials.columns,
                title=f"{metric_opt} vs {x_hp} (color={color_hp}, size={metric_opt})",
            )
            fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color="DarkSlateGrey")))
            st.plotly_chart(unify_figure_layout(fig, height=420), use_container_width=True)

    # Best configuration
    st.subheader("Best configuration")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    best = read_metrics_json(best_cfg)
    if isinstance(best, dict) and ("cfg" in best and "metrics" in best):
        cfg_dict = best["cfg"]
        df_hparams = pd.DataFrame(sorted(cfg_dict.items()), columns=["Hyperparameter", "Value"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Hyperparameters**")
            st.table(df_hparams.set_index("Hyperparameter"))
        metric_keys_whitelist = ["mql_0_50", "rmse", "mae", "smape", "mic_10_90"]
        metrics_raw = best["metrics"] or {}
        metrics_only = {k: metrics_raw[k] for k in metric_keys_whitelist if k in metrics_raw}
        order = [locals().get("metric_opt", "mql_0_50")] + [k for k in metric_keys_whitelist if k != locals().get("metric_opt", "mql_0_50")]
        metrics_ordered = [(k, metrics_only[k]) for k in order if k in metrics_only]
        df_metrics = pd.DataFrame(metrics_ordered, columns=["Metric", "Value"])
        df_metrics["Value"] = df_metrics["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        with c2:
            st.markdown("**Metrics**")
            st.table(df_metrics.set_index("Metric"))
    else:
        st.caption("No best_config.json yet or file not in expected format.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- 5) Test Predictions ----
elif section == "Test Predictions":
    st.header("🧪 Test Predictions (Warm-start)")

    preds_dir     = art["test"]["predictions_dir"]
    caseB_metrics = read_metrics_json(art["test"]["caseB_metrics"])

    # st.sidebar.caption("This page shows Case B metrics and a Case C (5-year) cumulative plot.")

    st.subheader("Per-well visualization (Darts)")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if active_df.empty or well_id is None:
        st.info("Select a well to view predictions.")
        st.stop()

    series_idx = _series_index_for_well(df_test, well_id)

    colB, colC = st.columns(2)
    shown_any = False
    for case_tag, col in [("caseB", colB), ("caseC", colC)]:
        png_path = _find_prediction_image(Path(preds_dir or ""), well_id, series_idx, case_tag)
        if png_path is not None:
            col.image(str(png_path), caption=f"{case_tag.upper()} – Well {well_id}", use_column_width=True)
            shown_any = True

    if not shown_any:
        per_well_csv = Path(preds_dir or "") / f"pred_well_{well_id}.csv"
        if per_well_csv.exists():
            df_pred = pd.read_csv(per_well_csv, parse_dates=[TIME_COL])
            df_w = filter_df_by_well(active_df, well_id)
            # replica in Plotly for consistency
            fig = go.Figure()
            hist = df_w[[TIME_COL, TARGET]].rename(columns={TARGET: "history"})
            fig.add_trace(go.Scatter(x=hist[TIME_COL], y=hist["history"], name="Warm start (history)", mode="lines"))
            if "y_true" in df_pred.columns:
                fig.add_trace(go.Scatter(x=df_pred[TIME_COL], y=df_pred["y_true"], name="Actual (truth)", mode="lines"))
            if {"p10","p50","p90"}.issubset(df_pred.columns):
                fig.add_trace(go.Scatter(
                    x=pd.concat([df_pred[TIME_COL], df_pred[TIME_COL][::-1]]),
                    y=pd.concat([df_pred["p90"], df_pred["p10"][::-1]]),
                    fill="toself", name="Pred 10–90%", line=dict(width=0), opacity=0.30
                ))
                fig.add_trace(go.Scatter(x=df_pred[TIME_COL], y=df_pred["p50"], name="Pred median (q0.50)", mode="lines"))
            st.plotly_chart(unify_figure_layout(fig, title=f"Well {well_id} – Forecast", height=420), use_container_width=True)
            st.dataframe(df_pred.tail(100), use_container_width=True)
        else:
            st.info(
                "No Darts PNG or CSV found for this well. "
                "Run `vm-test --case B --save-plots --export-csv` and/or "
                "`vm-test --case C --save-plots --export-csv` to generate artifacts."
            )

    c1, c2 = st.columns(2)

    # Case B metrics
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

    # Case C cumulative plot + KPIs
    def _plot_cumulative_oil_caseC(hist_df: pd.DataFrame,
                                   pred_df: pd.DataFrame,
                                   *,
                                   time_col: str,
                                   hist_cum_col: str,
                                   title: str):
        if hist_df.empty or pred_df.empty:
            return None, None
        hist_df = hist_df.sort_values(time_col).copy()
        pred_df = pred_df.sort_values(time_col).copy()
        pred_start = pred_df[time_col].min()
        hist_up_to_pred = hist_df[hist_df[time_col] <= pred_start].copy()
        start_cum = float(hist_up_to_pred[hist_cum_col].iloc[-1]) if not hist_up_to_pred.empty else 0.0
        hist_segment = hist_up_to_pred[[time_col, hist_cum_col]] if not hist_up_to_pred.empty else pd.DataFrame(columns=[time_col, hist_cum_col])

        if isinstance(TARGET, str) and "_bpd" in TARGET.lower():
            days = pred_df[time_col].dt.days_in_month.astype(float)
            vol_p10 = pred_df["p10"].astype(float) * days
            vol_p50 = pred_df["p50"].astype(float) * days
            vol_p90 = pred_df["p90"].astype(float) * days
        else:
            vol_p10 = pred_df["p10"].astype(float)
            vol_p50 = pred_df["p50"].astype(float)
            vol_p90 = pred_df["p90"].astype(float)

        cum_p10 = start_cum + vol_p10.cumsum()
        cum_p50 = start_cum + vol_p50.cumsum()
        cum_p90 = start_cum + vol_p90.cumsum()

        fig = go.Figure()
        if not hist_segment.empty:
            fig.add_trace(go.Scatter(
                x=hist_segment[time_col],
                y=hist_segment[hist_cum_col],
                mode="lines",
                name="Actual cumulative",
                line=dict(width=2)
            ))
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
        fig = unify_figure_layout(fig, title=title, height=440)
        final_p10_total = float(cum_p10.iloc[-1]) if len(cum_p10) else None
        final_p50_total = float(cum_p50.iloc[-1]) if len(cum_p50) else None
        final_p90_total = float(cum_p90.iloc[-1]) if len(cum_p90) else None
        return fig, final_p10_total, final_p50_total, final_p90_total

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
                    c2.plotly_chart(fig_cum, use_container_width=True)

    # st.caption(
    #     "This page shows Darts-rendered PNGs when available "
    #     "(`caseB_series_###.png`, `caseC_series_###.png`). "
    #     "Optionally, you can also save by well id "
    #     "(`caseB_well_<id>.png`, `caseC_well_<id>.png`)."
    # )

# ---- 6) Explainability ----
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
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if not local_dir.exists():
            st.info("No local explainability yet. Run `vm-explain`.")
        else:
            img_path = local_dir / f"local_variable_importance_well_{well_id}.png"
            if img_path.exists():
                show_image_safe(img_path, f"Local VI – Well {well_id}")
            else:
                st.caption("No local VI image for this well yet. Run `vm-explain --save-by well`.")
            csv_path = local_dir / "local_variable_importance.csv"
            if csv_path.exists():
                df_local = pd.read_csv(csv_path)
                if "well_id" in df_local.columns:
                    st.dataframe(df_local.query("well_id == @well_id"), use_container_width=True, height=360)
                else:
                    st.dataframe(df_local, use_container_width=True, height=360)

    with tabs[1]:
        st.subheader("Temporal Attention")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if not attention_dir.exists():
            st.info("No attention plots yet. Run `vm-explain`.")
        else:
            attn_all   = attention_dir / f"attention_well_{well_id}.png"
            attn_heat  = attention_dir / f"attention_heatmap_well_{well_id}.png"
            shown_any = False
            if attn_all.exists():
                show_image_safe(str(attn_all), f"Attention (all) – Well {well_id}")
                shown_any = True
            if attn_heat.exists():
                show_image_safe(str(attn_heat), f"Attention (heatmap) – Well {well_id}")
                shown_any = True
            if not shown_any:
                st.caption("No attention image for this well yet. Run `vm-explain --save-by well --attention-types all,heatmap`.")

    with tabs[2]:
        st.subheader("Pseudo-global Variable Importance")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if Path(global_csv).exists():
            df_global = pd.read_csv(global_csv)
            st.dataframe(df_global, use_container_width=True, height=420)
            block = st.selectbox("Block", sorted(df_global["block"].unique().tolist()))
            df_b = df_global[df_global["block"] == block].nlargest(20, "importance_mean")
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
                    categoryarray=df_b["feature"].tolist(),
                    tickfont=dict(size=18)
                ),
                height=700
            )
            st.plotly_chart(unify_figure_layout(fig, height=700), use_container_width=True)
        else:
            st.info("No pseudo-global VI CSV yet. Run `vm-explain`.")

# ========== Footer ==========
st.markdown("---")
st.caption("© Alexis Ortega • Vaca Muerta TFT • Streamlit dashboard •  https://tft-vaca-muerta-oil-forecasting.streamlit.app/")
