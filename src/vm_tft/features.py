# Feature engineering & future covariates

import pandas as pd, numpy as np
from pandas.tseries.frequencies import to_offset

def month_diff_idx(idx: pd.DatetimeIndex, completion: pd.Timestamp) -> np.ndarray:
    return (idx.year - completion.year) * 12 + (idx.month - completion.month)

def build_future_covariates_well_age_full(df, time_col, group_col, completion_col, horizon, freq="MS"):
    df = df.copy()
    df[completion_col] = pd.to_datetime(df[completion_col])
    off = to_offset(freq)
    frames = []

    for gid, g in df[[group_col, time_col, completion_col]].dropna().groupby(group_col):
        start_hist = g[time_col].min()
        end_hist   = g[time_col].max()
        end_full   = end_hist + horizon * off
        idx = pd.date_range(start=start_hist, end=end_full, freq=freq)
        comp = pd.to_datetime(g[completion_col].iloc[0])
        age = month_diff_idx(idx, comp)
        age = np.clip(age, 0, None)
        frames.append(pd.DataFrame({group_col: gid, time_col: idx, "well_age": age}))

    return pd.concat(frames, ignore_index=True)


# put this next to build_future_covariates_well_age_full (e.g., in vm_tft/features.py)
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from typing import Optional, Tuple
try:
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _months_since(idx_like, anchor):
    """
    Months between idx_like and anchor. Works for Series, Index, or DatetimeIndex.
    """
    idx = pd.DatetimeIndex(idx_like)          # <- handles Series/Index robustly
    anchor = pd.Timestamp(anchor)
    return ((idx.year - anchor.year) * 12 + (idx.month - anchor.month)).astype(float)


def _arps_hyperbolic(t, qi, Di, b):
    # t in months; Di per-month; b dimensionless
    t = np.asarray(t, dtype=float)
    b = np.maximum(b, 1e-6)
    return qi / np.power(1.0 + b * Di * np.maximum(t, 0.0), 1.0 / b)

def _fit_arps_first_k_months(
    df_well: pd.DataFrame,
    k: int,
    time_col: str,
    target_col: str,
) -> Optional[Tuple[float, float, float]]:
    """Return (qi, Di, b) or None if it cannot be fit robustly."""
    d = df_well[[time_col, target_col]].dropna().sort_values(time_col)
    if len(d) < max(6, k//2):
        return None
    d_k = d.head(k).copy()
    t = _months_since(d_k[time_col], d_k[time_col].iloc[0])
    y = d_k[target_col].astype(float).values

    # init + bounds
    qi0 = max(y[0], np.percentile(y, 90))
    Di0 = 0.10
    b0  = 0.7
    lb  = [1e-2, 1e-4, 0.0]
    ub  = [1e7,  2.0,  2.5]

    if _HAS_SCIPY:
        try:
            popt, _ = curve_fit(_arps_hyperbolic, t, y, p0=[qi0, Di0, b0],
                                bounds=(lb, ub), maxfev=10000)
            qi, Di, b = map(float, popt)
            return qi, Di, b
        except Exception:
            pass

    # Fallback: exponential (b≈0)
    # qi * exp(-Di * t) → quick robust fit via least squares in log-space
    t = t.reshape(-1, 1)
    y_pos = np.clip(y, 1e-6, None)
    A = np.hstack([np.ones_like(t), -t])          # log y = log qi - Di * t
    beta, *_ = np.linalg.lstsq(A, np.log(y_pos), rcond=None)
    qi = float(np.exp(beta[0])); Di = float(beta[1]); b = 1e-3
    if not np.isfinite(qi) or not np.isfinite(Di):
        return None
    Di = max(Di, 1e-5)
    return qi, Di, b

def build_future_covariates_dca_full(
    df: pd.DataFrame,
    time_col: str,
    group_col: str,
    target_col: str,
    *,
    horizon: int,
    freq: str = "MS",
    fit_k: int = 24,
    params_out: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each well (group_col):
      - fit Arps on the first `fit_k` months (target_col),
      - build index from start_hist .. end_hist + horizon,
      - compute DCA curve as 'dca_bpd' for all those dates.

    Returns:
      df_dca: [group_col, time_col, dca_bpd]
      and optionally params_df: [group_col, qi, Di, b] if params_out=True
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    off = to_offset(freq)

    frames, pars = [], []
    for gid, g in df[[group_col, time_col, target_col]].dropna().groupby(group_col):
        g = g.sort_values(time_col)
        start_hist = g[time_col].min()
        end_hist   = g[time_col].max()
        end_full   = end_hist + horizon * off
        idx = pd.date_range(start=start_hist, end=end_full, freq=freq)

        params = _fit_arps_first_k_months(g, k=fit_k, time_col=time_col, target_col=target_col)
        if params is None:
            # If we can’t fit, just forward-fill last observed value through horizon (safe fallback)
            last = float(g[target_col].iloc[-1])
            dca_vals = np.full(len(idx), last, dtype=float)
        else:
            qi, Di, b = params
            t = _months_since(idx, start_hist)
            dca_vals = _arps_hyperbolic(t, qi, Di, b)
            pars.append({group_col: gid, "qi": qi, "Di": Di, "b": b})

        frames.append(pd.DataFrame({group_col: gid, time_col: idx, "dca_bpd": dca_vals}))

    df_dca = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[group_col, time_col, "dca_bpd"])
    if params_out:
        params_df = pd.DataFrame(pars) if pars else pd.DataFrame(columns=[group_col, "qi", "Di", "b"])
        return df_dca, params_df
    return df_dca
