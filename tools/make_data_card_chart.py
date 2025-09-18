# tools/make_data_card_chart.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- repo/paths (adjust if needed)
ART = Path("data/artifacts")
OUT = ART / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# --- import your cfg to stay in sync with the app
from vm_tft.cfg import ALL_FILE, TEST_FILE, TIME_COL, GROUP_COL

def load_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, parse_dates=[TIME_COL])

def months_of_history(df: pd.DataFrame) -> pd.Series:
    # Count actual observed months per well (robust to gaps)
    return (df
            .dropna(subset=[TIME_COL, GROUP_COL])
            .groupby(GROUP_COL)[TIME_COL]
            .nunique()
            .rename("months"))

def make_strip(series: pd.Series,
               palette="#2E86AB",
               width=800, height=240) -> go.Figure:
    s = series.dropna().astype(int).sort_values()
    if s.empty:
        # empty placeholder
        fig = go.Figure().update_layout(width=width, height=height)
        return fig

    # summary stats
    vmin   = int(s.min())
    vmed   = float(s.median())
    vmax   = int(s.max())

    # build “strip”: y=0 with a tiny jitter to avoid overplotting
    rng = np.random.default_rng(42)
    jitter = (rng.random(len(s)) - 0.5) * 0.08  # tiny vertical jitter

    fig = go.Figure()

    # points
    fig.add_trace(go.Scatter(
        x=s.values,
        y=np.zeros(len(s)) + jitter,
        mode="markers",
        name="Well history (months)",
        marker=dict(size=6, opacity=0.55, line=dict(width=0.5, color="rgba(0,0,0,0.35)"),
                    color=palette),
        hovertemplate="Months: %{x}<extra></extra>"
    ))

    # vertical markers for min / median / max
    def vline(x, label, color):
        fig.add_vline(x=x, line_width=2, line_color=color)
        fig.add_annotation(
            x=x, y=0.42, xref="x", yref="paper",
            text=f"{label}",
            showarrow=False, font=dict(size=11, color=color),
            yanchor="bottom"
        )

    vline(vmin,   f"Min {vmin}",   "#6C757D")
    vline(vmed,   f"Median {vmed:.0f}", "#F18F01")
    vline(vmax,   f"Max {vmax}",   "#C73E1D")

    fig.update_layout(
        title="History length per well (months)",
        width=width, height=height,
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(title="", showgrid=True, zeroline=False,
                   gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        showlegend=False,
        font=dict(family="Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=13),
    )
    return fig

def main():
    df_all  = load_csv(Path(ALL_FILE))
    df_test = load_csv(Path(TEST_FILE))
    # You can switch to df_test if you want the strip only for test wells.
    months = months_of_history(df_all)

    vmin = int(months.min()) if len(months) else 0
    vmed = float(months.median()) if len(months) else 0.0
    vmax = int(months.max()) if len(months) else 0

    print(f"[Data card] History length (months) — Min={vmin}  Median={vmed:.0f}  Max={vmax}")

    fig = make_strip(months, palette="#2E86AB", width=800, height=240)

    # Export PNG (requires: pip install -U kaleido)
    out_png = OUT / "data_history_strip_800x240.png"
    try:
        import plotly.io as pio
        pio.write_image(fig, out_png, scale=2)
        print(f"Saved: {out_png}")
    except Exception as e:
        print("Tip: pip install -U kaleido to save PNG. Opening interactive window instead.")
        fig.show()

if __name__ == "__main__":
    main()
    
    


import pandas as pd
import plotly.graph_objects as go
from typing import Optional

TIME_COL = "date"
TARGET   = "oil_rate_bpd"     # if endswith "bpd" we convert to monthly bbl for cum calc

def make_per_well_interpretability_figure(
    hist_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    local_vi_df: Optional[pd.DataFrame] = None,
    *,
    time_col: str = "date",
    y_col: str = "oil_rate_bpd",
    hist_cum_col: Optional[str] = "cum_oil_bbl",
    well_label: Optional[str] = None,
):
    """
    Build a single interpretability figure:
      - history (left of split)
      - prediction band (P10–P90) + P50 (right of split)
      - optional y_true inside prediction window if present
      - vertical dashed line at prediction start
      - outside-plot annotations:
          top: "5-yr cumulative P50 = X bbl (Δ vs history = Y bbl)"
          bottom: "Top drivers (local VI): A • B • C"
    """
    # ---- guards
    if pred_df is None or pred_df.empty:
        raise ValueError("pred_df is empty (need at least time_col + p10/p50/p90).")
    need_cols = {time_col, "p50"}
    if not need_cols.issubset(pred_df.columns):
        raise ValueError(f"pred_df must contain {need_cols}.")

    # ---- basic pieces
    pred = pred_df.sort_values(time_col).copy()
    pred_start = pred[time_col].min()

    # History strictly before prediction start (matches Darts’ PNGs)
    hist = pd.DataFrame(columns=[time_col, y_col])
    if hist_df is not None and not hist_df.empty:
        hist = hist_df[[time_col, y_col]].dropna().sort_values(time_col)
        hist = hist.loc[hist[time_col] < pred_start]

    # Optional truth line in prediction window (for Case B)
    has_truth = "y_true" in pred.columns and pred["y_true"].notna().any()

    # ---- coverage (good calibration chip)
    coverage = None
    if {"p10", "p90"}.issubset(pred.columns) and has_truth:
        yy = pred.dropna(subset=["p10", "p90", "y_true"])
        inside = (yy["y_true"] >= yy["p10"]) & (yy["y_true"] <= yy["p90"])
        coverage = float(inside.mean()) if len(yy) else None  # 0–1

    # ---- cumulative 5-yr P50 (from pred_start forward)
    # If target looks like *_bpd, convert to monthly barrels
    def _is_bpd(name: str) -> bool:
        return isinstance(name, str) and ("_bpd" in name.lower() or "per_day" in name.lower())

    p50 = pred["p50"].astype(float).clip(lower=0.0)
    if _is_bpd(y_col):
        months = pd.to_datetime(pred[time_col]).dt.days_in_month.astype(float)
        monthly_vol = p50.values * months.values
    else:
        monthly_vol = p50.values

    # starting cumulative from history (if a cumulative column exists)
    start_cum = None
    if hist_cum_col and hist_df is not None and not hist_df.empty and hist_cum_col in hist_df.columns:
        up_to = hist_df.loc[hist_df[time_col] <= pred_start, [time_col, hist_cum_col]].dropna()
        if not up_to.empty:
            start_cum = float(up_to[hist_cum_col].iloc[-1])
    if start_cum is None:
        start_cum = 0.0

    final_p50_total = start_cum + float(np.cumsum(monthly_vol)[-1])
    delta_vs_hist = final_p50_total - start_cum

    # ---- Build figure
    fig = go.Figure()

    # History
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist[time_col], y=hist[y_col],
            mode="lines", name="Actual (history)",
            line=dict(width=2, color="#2F3B48")
        ))

    # Prediction band
    if {"p10", "p90"}.issubset(pred.columns):
        fig.add_trace(go.Scatter(
            x=pd.concat([pred[time_col], pred[time_col][::-1]]),
            y=pd.concat([pred["p90"], pred["p10"][::-1]]),
            fill="toself", name="P10–P90",
            line=dict(width=0), opacity=0.25,
            fillcolor="rgba(241, 143, 1, 0.22)",  # #F18F01 with alpha
            hoverinfo="skip",
        ))

    # P50
    fig.add_trace(go.Scatter(
        x=pred[time_col], y=pred["p50"],
        mode="lines", name="P50",
        line=dict(width=2, color="#F18F01")
    ))

    # Truth continuation (if available)
    if has_truth:
        yy = pred.dropna(subset=["y_true"])
        fig.add_trace(go.Scatter(
            x=yy[time_col], y=yy["y_true"],
            mode="lines", name="Actual (truth)",
            line=dict(width=2, color="#4E79A7")
        ))

    # Vertical split line
    fig.add_vline(
        x=pred_start, line_width=2, line_dash="dash", line_color="red",
        annotation_text="", annotation_position="top"
    )

    # Titles & axes
    ttl = f"Per-well example" if well_label is None else f"Per-well example – {well_label}"
    fig.update_layout(
        title=ttl,
        xaxis_title="Date",
        yaxis_title=y_col,
        margin=dict(l=10, r=10, t=110, b=90),  # extra room for outside annotations
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )

    # ---- OUTSIDE annotations
    # Top badge: cumulative summary (above the plot)
    fig.add_annotation(
        x=0.5, y=1.14, xref="paper", yref="paper",
        text=f"5-yr cumulative P50 = <b>{final_p50_total:,.0f}</b> bbl"
             f"  (Δ vs history = <b>{delta_vs_hist:,.0f}</b> bbl)",
        showarrow=False,
        align="center",
        font=dict(size=13),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#DDD",
        borderwidth=1,
        borderpad=6,
    )

    # Small chip on the top-right: coverage
    if coverage is not None:
        fig.add_annotation(
            x=1.0, y=1.14, xref="paper", yref="paper",
            xanchor="right",
            text=f"Coverage (P10–P90): <b>{100*coverage:.0f}%</b>",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#DDD",
            borderwidth=1,
            borderpad=4,
        )

    # Bottom chip: top local drivers
    if local_vi_df is not None and not local_vi_df.empty:
        # expects columns ['feature','importance'] already filtered to this well
        top_feats = (local_vi_df.sort_values("importance", ascending=False)
                                 .head(3)["feature"].tolist())
        if top_feats:
            text = "Top drivers (local VI): " + " • ".join(top_feats)
            fig.add_annotation(
                x=0.0, y=-0.20, xref="paper", yref="paper",
                xanchor="left",
                text=text,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#DDD",
                borderwidth=1,
                borderpad=6,
            )

    return fig




df_all  = load_csv(Path(ALL_FILE))
df_test = load_csv(Path(TEST_FILE))

active_df = df_test.copy()
well_id = 162353

# 1) history
cols = [TIME_COL, TARGET]
if "cum_oil_bbl" in active_df.columns:
    cols.append("cum_oil_bbl")
hist_df = (
    active_df.loc[active_df[GROUP_COL] == well_id, cols]
             .dropna(subset=[TARGET])
             .sort_values(TIME_COL)
             .copy()
)

# 2) predictions

ART = Path("data/artifacts")
preds_dir = ART / "predictions/plots"
pred_df = pd.read_csv(Path(preds_dir) / f"pred_well_{well_id}.csv", parse_dates=[TIME_COL]) \
            .sort_values(TIME_COL)

# 3) local VI (optional)
vi_csv = ART / "explain/local/local_variable_importance.csv"

local_vi_df = None
if vi_csv.exists():
    vi_all = pd.read_csv(vi_csv)
    if "well_id" in vi_all.columns:
        local_vi_df = vi_all.loc[vi_all["well_id"] == well_id, ["feature", "importance"]]
        
        
# 4) well label (use your well_name mapping if you have it)
well_label = "Well 162352"

# 5) figure

fig = make_per_well_interpretability_figure(hist_df, pred_df, local_vi_df, well_label=well_label)


fig


from vm_tft.cfg import TARGET, TEST_FILE, GROUP_COL, TIME_COL
import pandas as pd

wid = 162352
df_test = pd.read_csv(TEST_FILE, parse_dates=[TIME_COL])

print("TARGET:", TARGET)
print("raw oil max:", df_test.loc[df_test[GROUP_COL]==wid, "oil_rate_bpd"].max() if "oil_rate_bpd" in df_test else "n/a")
print("target max:", df_test.loc[df_test[GROUP_COL]==wid, TARGET].max())

# Where does the target peak occur?
print(
    df_test.loc[df_test[GROUP_COL]==wid, [TIME_COL, TARGET]]
           .sort_values(TIME_COL)
           .head(12)
)




import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Mapper
series = TimeSeries.from_values(np.array([1e0, 1e1, 1e2, 1e3]))
transformer = Mapper(np.log10)
series_transformed = transformer.transform(series)
print(series_transformed)


series_restaured = transformer.inverse_transform(series_transformed)
print(series_restaured)


from darts.dataprocessing.transformers import InvertibleMapper, Scaler
from darts.dataprocessing.pipeline import Pipeline as DartsPipeline

series = TimeSeries.from_values(np.array([1e0, 1e1, 1e2, 1e3]))
transformer = InvertibleMapper(np.log10, lambda x: 10**x)
series_transformed = transformer.transform(series)
print(series_transformed)

series_restaured = transformer.inverse_transform(series_transformed)
print(series_restaured)



series = TimeSeries.from_values(np.array([1e0, 1e1, 1e2, 1e3]))
log_mapper = InvertibleMapper(fn=np.log1p, inverse_fn=np.expm1, name="log1p")

y_scaler = DartsPipeline([log_mapper, Scaler()])

y_scaler.fit(series)
series_transformed = y_scaler.transform(series)
print(series_transformed)

series_restaured = y_scaler.inverse_transform(series_transformed)
print(series_restaured)


series_all = TimeSeries.from_values(np.array([1e0, 1e1, 1e2, 1e3, 2e2, 4e1, 1e3, 1e2, 5e0, 7e1, 2e0, 4e1, 3e2, 2e3, 2e2, 8e1, 1e3, 6e2, 8e0, 7e1]))

series_test = TimeSeries.from_values(np.array([8e0, 2e1, 16e2, 3e3]))

log_mapper = InvertibleMapper(fn=np.log1p, inverse_fn=np.expm1, name="log1p")

y_scaler = DartsPipeline([log_mapper, Scaler()])

y_scaler.fit(series_all)
series_test_transformed = y_scaler.transform(series_test)
print(series_test_transformed)

series_test_restaured = y_scaler.inverse_transform(series_test_transformed)
print(series_test_restaured)
print(series_test)
