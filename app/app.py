# src/vm_tft/app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# Optional plotting libs (comment/uncomment based on your taste)
import plotly.express as px
import plotly.graph_objects as go

# Darts objects are optional – we only import if a model is present
try:
    from darts.models import TFTModel
    from darts.timeseries import TimeSeries
except Exception:
    TFTModel = None
    TimeSeries = None

# ---- internal modules (your package) ----
from vm_tft.cfg import (
    ALL_FILE, TEST_FILE, ARTIFACTS,
    TIME_COL, GROUP_COL, TARGET,
    FREQ,
)
from vm_tft.io_utils import ensure_dir


# =========================
# Streamlit Page Settings
# =========================
st.set_page_config(
    page_title="Vaca Muerta – TFT Dashboard",
    layout="wide",
    page_icon="🛢️",
)

# =========================
# Utilities & Cache
# =========================

def file_sig(path: Path) -> str | None:
    """
    Lightweight signature for a file so Streamlit cache invalidates when content changes.
    Uses (mtime_ns, size) to avoid hashing large files.
    """
    try:
        stt = Path(path).stat()
        return f"{stt.st_mtime_ns}-{stt.st_size}"
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_csv(path: Path, _sig: str | None = None) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    # _sig is unused here, just part of the cache key
    df = pd.read_csv(path, parse_dates=[TIME_COL])
    return df

@st.cache_data(show_spinner=False)
def read_metrics_json(path: Path, _sig: str | None = None) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def read_metrics_csv(path: Path, _sig: str | None = None) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_tft(model_path: Path):
    if not model_path.exists() or TFTModel is None:
        return None
    return TFTModel.load(str(model_path))

def list_artifacts() -> dict:
    """Discover common artifacts produced by your CLIs."""
    base = Path(ARTIFACTS)
    return {
        "models": {
            "best_from_search": base / "models" / "tft_best_from_search.pt",
            "best_manual": base / "models" / "tft_best_manual.pt",
        },
        "cv": {
            "overall_metrics": base / "metrics" / "cv_overall.json",
            "per_series_window": base / "metrics" / "cv_metrics_per_series_window.csv",
        },
        "search": {
            "trials_csv": base / "metrics" / "search_trials.csv",
            "best_config": base / "metrics" / "search_best.json",
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

def plot_forecast_band(
    hist: pd.DataFrame,
    pred: pd.DataFrame,
    y_col: str,
    q_low: str = "p10",
    q_med: str = "p50",
    q_hi: str = "p90",
    title: str = "Forecast",
):
    fig = go.Figure()

    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist[TIME_COL], y=hist[y_col],
            mode="lines", name="Actual (history)"
        ))

    if not pred.empty and {q_low, q_med, q_hi}.issubset(pred.columns):
        # band
        fig.add_trace(go.Scatter(
            x=pd.concat([pred[TIME_COL], pred[TIME_COL][::-1]]),
            y=pd.concat([pred[q_hi], pred[q_low][::-1]]),
            fill="toself", name=f"{q_low.upper()}–{q_hi.upper()}",
            line=dict(width=0), opacity=0.25
        ))
        fig.add_trace(go.Scatter(
            x=pred[TIME_COL], y=pred[q_med],
            mode="lines", name=q_med.upper(), line=dict(width=2)
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
    ["Data Explorer", "Model CV", "Randomized Search", "Test Predictions", "Explainability"],
)

# global controls
art = list_artifacts()
df_all  = load_csv(ALL_FILE,  _sig=file_sig(ALL_FILE))
df_test = load_csv(TEST_FILE, _sig=file_sig(TEST_FILE))

st.sidebar.subheader("Global controls")
default_set = st.sidebar.selectbox("Active dataset", ["Test", "Train/All"], index=0)
active_df = df_test if default_set == "Test" else df_all

well_ids = get_well_ids(active_df)
well_id = st.sidebar.selectbox("Well", well_ids, index=0 if well_ids else None)

quantile_low  = st.sidebar.slider("PI low (q_low)", 0.0, 0.5, 0.10, 0.05)
quantile_high = st.sidebar.slider("PI high (q_high)", 0.5, 1.0, 0.90, 0.05)

st.sidebar.markdown("---")
st.sidebar.caption("Artifacts root: `{}`".format(ARTIFACTS))

# Sidebar force refresh button
if st.sidebar.button("🔄 Force refresh"):
    st.cache_data.clear()
    st.rerun()


# =========================
# Sections
# =========================

# ---- 1) Data Explorer ----
if section == "Data Explorer":
    st.header("📊 Data Explorer")

    if active_df.empty:
        st.warning("No data found. Run `vm-prepare-data` first.")
        st.stop()

    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.subheader("Production time series")
        if well_id is not None:
            df_w = filter_df_by_well(active_df, well_id)
            if df_w.empty:
                st.info("Selected well has no rows.")
            else:
                plot_series_px(df_w, TARGET, f"Well {well_id} – {TARGET}")
        else:
            st.info("Select a well in the sidebar.")

    with c2:
        st.subheader("Static covariates (sample)")
        static_cols = [c for c in active_df.columns if c not in [TIME_COL, GROUP_COL, TARGET] and not c.endswith("_bpd") and not c.endswith("_mscf")]
        show_cols = [c for c in static_cols if active_df[c].nunique(dropna=True) > 1][:8]
        if show_cols:
            st.dataframe(active_df[[GROUP_COL] + show_cols].drop_duplicates(GROUP_COL).head(50))
        else:
            st.caption("No static covariates to display.")

    st.subheader("Raw data (head)")
    st.dataframe(active_df.head(200), use_container_width=True)


# ---- 2) Model CV ----
elif section == "Model CV":
    st.header("🧪 Cross-Validation Results")

    overall_path = art["cv"]["overall_metrics"]
    per_series_path = art["cv"]["per_series_window"]

    overall    = read_metrics_json(overall_path,    _sig=file_sig(overall_path))
    per_series = read_metrics_csv(per_series_path,  _sig=file_sig(per_series_path))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    if overall:
        metric_card("RMSE", f"{overall.get('rmse', np.nan):.2f}")
        metric_card("MAE", f"{overall.get('mae', np.nan):.2f}")
        metric_card("sMAPE", f"{overall.get('smape', np.nan):.2f}")
        metric_card("MQL@0.50", f"{overall.get('mql_0_50', np.nan):.2f}")
        metric_card("MIC@10–90", f"{overall.get('mic_10_90', np.nan):.3f}")
        metric_card("MAPE", f"{overall.get('mape', np.nan):.2f}")
    else:
        st.info("No CV overall metrics found yet.")

    st.subheader("Per-series / window metrics")
    if per_series.empty:
        st.info("No per-series/window CSV found.")
    else:
        st.dataframe(per_series, use_container_width=True, height=420)

        # Quick distribution
        st.subheader("RMSE distribution")
        fig = px.histogram(per_series, x="rmse", nbins=40, title="RMSE")
        st.plotly_chart(fig, use_container_width=True)


# ---- 3) Randomized Search ----
elif section == "Randomized Search":
    st.header("🔍 Randomized Search")

    trials_csv = art["search"]["trials_csv"]
    best_cfg   = art["search"]["best_config"]

    trials = read_metrics_csv(trials_csv, _sig=file_sig(trials_csv))
    if trials.empty:
        st.info("No trials summary yet. Run `vm-search-fit`.")
    else:
        st.subheader("Trials summary")
        st.dataframe(trials, use_container_width=True, height=420)

        metric_col = st.selectbox("Optimize metric to visualize", ["mql_0_50", "rmse", "mae", "smape", "mic_10_90"], index=0)

        # Try to show a simple hyperparam vs metric scatter
        hp_x = st.selectbox("X hyperparam", [c for c in trials.columns if c not in ["mql_0_50","rmse","mae","smape","mic_10_90"]], index=0)
        hp_y = st.selectbox("Y hyperparam", [c for c in trials.columns if c not in ["mql_0_50","rmse","mae","smape","mic_10_90", hp_x]], index=0)
        fig = px.scatter(trials, x=hp_x, y=metric_col, color=hp_y, title=f"{metric_col} vs {hp_x} (color={hp_y})")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Best configuration")
    best = read_metrics_json(best_cfg, _sig=file_sig(best_cfg))
    if best:
        st.json(best)
    else:
        st.caption("No best_config.json yet.")


# ---- 4) Test Predictions ----
elif section == "Test Predictions":
    st.header("🧪 Test Predictions (Warm-start)")

    # expected prediction plots directory (your CLI can save per-well PNGs or CSV)
    preds_dir = art["test"]["predictions_dir"]
    caseA_metrics = read_metrics_json(art["test"]["caseA_metrics"], _sig=file_sig(art["test"]["caseA_metrics"]))
    caseB_metrics = read_metrics_json(art["test"]["caseB_metrics"], _sig=file_sig(art["test"]["caseB_metrics"]))


    c1, c2, c3 = st.columns(3)
    if caseA_metrics:
        c1.subheader("Case A (fixed horizon)")
        c1.json(caseA_metrics)
    if caseB_metrics:
        c2.subheader("Case B (until end of history)")
        c2.json(caseB_metrics)

    st.subheader("Per-well visualization")
    if active_df.empty or well_id is None:
        st.info("Select a well to view predictions.")
    else:
        # If you saved CSV per well with columns:
        # date, y_true, p10, p50, p90 (or y_hat for deterministic)
        per_well_csv = Path(preds_dir or "") / f"pred_well_{well_id}.csv"
        if per_well_csv.exists():
            df_pred = pd.read_csv(per_well_csv, parse_dates=[TIME_COL])
            df_w = filter_df_by_well(active_df, well_id)

            # rename quantile cols based on sidebar selection if needed
            ql = f"p{int(quantile_low*100)}" if f"p{int(quantile_low*100)}" in df_pred.columns else "p10"
            qh = f"p{int(quantile_high*100)}" if f"p{int(quantile_high*100)}" in df_pred.columns else "p90"

            plot_forecast_band(
                hist=df_w[[TIME_COL, TARGET]],
                pred=df_pred,
                y_col=TARGET,
                q_low=ql, q_med="p50", q_hi=qh,
                title=f"Well {well_id} – Forecast",
            )
            st.dataframe(df_pred.tail(100), use_container_width=True)
        else:
            st.info("No saved prediction CSV/plot for this well yet. Run `vm-test-predict` with export.")

    st.caption("Tip: have your test CLI export per-well CSVs to `artifacts/test/plots/pred_well_<id>.csv`.")


# ---- 5) Explainability ----
elif section == "Explainability":
    st.header("🪄 Explainability")

    local_dir     = Path(art["explain"]["local_dir"])
    attention_dir = Path(art["explain"]["attention_dir"])
    global_csv    = Path(art["explain"]["global_csv"])

    tabs = st.tabs(["Local VI", "Attention", "Global VI"])

    # helper: load local CSV once and possibly map well_id -> series_idx
    local_csv_path = local_dir / "local_variable_importance.csv"
    df_local = read_metrics_csv(local_csv_path, _sig=file_sig(local_csv_path)) if local_csv_path.exists() else pd.DataFrame()

    def _series_idx_for_well(wid: int) -> Optional[int]:
        if df_local.empty or "well_id" not in df_local.columns:
            return None
        try:
            return int(df_local.loc[df_local["well_id"] == wid, "series_idx"].iloc[0])
        except Exception:
            return None

    with tabs[0]:
        st.subheader("Local Variable Importance (per well)")

        # Handle both "dir" and "csv file" cases robustly
        local_path = Path(local_dir)
        if not local_path.exists():
            st.info("No local explainability yet. Run `vm-explain`.")
        else:
            base_dir = local_path if local_path.is_dir() else local_path.parent

            # Try to show well-id image
            img_path = base_dir / f"local_variable_importance_well_{well_id}.png"
            if img_path.exists():
                st.image(str(img_path), caption=f"Local VI – Well {well_id}", use_container_width=True)
            else:
                st.caption("No local VI image for this well yet. Make sure you ran `vm-explain` with `--save-by well`.")

            # Show CSV (filter by well_id if present)
            csv_path = base_dir / "local_variable_importance.csv"
            if csv_path.exists():
                df_local = read_metrics_csv(csv_path, _sig=file_sig(csv_path))
                if "well_id" in df_local.columns:
                    st.dataframe(
                        df_local.query("well_id == @well_id"),
                        use_container_width=True, height=400
                    )
                elif "series_idx" in df_local.columns:
                    st.dataframe(
                        df_local,  # fallback: show all if no well_id column
                        use_container_width=True, height=400
                    )
                else:
                    st.dataframe(df_local, use_container_width=True, height=400)
            else:
                st.caption("No `local_variable_importance.csv` found in the local explainability folder.")


    with tabs[1]:
        st.subheader("Temporal Attention")
        attn_dir = Path(attention_dir)
        if not attn_dir.exists():
            st.info("No attention plots yet. Run `vm-explain`.")
        else:
            attn_all   = attn_dir / f"attention_well_{well_id}.png"
            attn_heat  = attn_dir / f"attention_heatmap_well_{well_id}.png"

            shown_any = False
            if attn_all.exists():
                st.image(str(attn_all), caption=f"Attention (all) – Well {well_id}", use_container_width=True)
                shown_any = True
            if attn_heat.exists():
                st.image(str(attn_heat), caption=f"Attention (heatmap) – Well {well_id}", use_container_width=True)
                shown_any = True

            if not shown_any:
                st.caption("No attention image for this well yet. Make sure you ran `vm-explain` with `--save-by well`.")


    with tabs[2]:
        st.subheader("Pseudo-global Variable Importance")
        if global_csv.exists():
            df_global = read_metrics_csv(global_csv, _sig=file_sig(global_csv))
            st.dataframe(df_global, use_container_width=True, height=420)
            block = st.selectbox("Block", sorted(df_global["block"].unique().tolist()))
            df_b = df_global[df_global["block"] == block].nlargest(20, "importance_mean")
            fig = px.bar(df_b, x="importance_mean", y="feature", orientation="h",
                         title=f"Top features – {block}")
            fig.update_layout(
                    yaxis=dict(categoryorder="total ascending")  # highest importance at top
                    )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pseudo-global VI CSV yet. Run `vm-explain`.")


# ========== Footer ==========
st.markdown("---")
st.caption(
    "© Vaca Muerta TFT • Streamlit dashboard • By Alexis Ortega • "
    
    "Data paths read from `vm_tft.cfg`, artifacts discovered in `artifacts/`."
)
