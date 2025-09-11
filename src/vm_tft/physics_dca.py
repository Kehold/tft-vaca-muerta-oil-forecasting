import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def arps_hyperbolic(t, qi, Di, b):
    # t in months from start, Di per-month, b hyperbolic exponent
    return qi / np.power(1.0 + b * Di * np.maximum(t, 0.0), 1.0 / np.maximum(b, 1e-6))

def fit_arps_first_k_months(df_well: pd.DataFrame, k: int, target_col: str, time_col: str) -> tuple[float,float,float]:
    """
    Fit hyperbolic Arps to the first k months of a single well.
    Returns (qi, Di, b). Robust-ish bounds; tune as needed.
    """
    d = df_well.sort_values(time_col).copy()
    d = d.loc[d[target_col].notna()]
    if len(d) < max(6, k//2):  # need a few points
        raise RuntimeError("Not enough points to fit DCA")

    # build t=0.. for first k rows
    d_k = d.head(k).copy()
    d_k["tmo"] = np.arange(len(d_k), dtype=float)  # months since start
    y = d_k[target_col].values.astype(float)
    t = d_k["tmo"].values

    # initial guesses and bounds
    qi0 = max(y[0], np.percentile(y, 90))
    Di0 = 0.10  # ~10%/month starting guess (tune for your basin)
    b0  = 0.7
    bounds = ([1e-2, 1e-4, 0.0], [1e6, 2.0, 2.5])  # qi:[.01,1e6], Di:[1e-4,2], b:[0,2.5]

    # fit on log-space weights to reduce early dominance (optional)
    try:
        popt, _ = curve_fit(arps_hyperbolic, t, y, p0=[qi0, Di0, b0], bounds=bounds, maxfev=10000)
    except Exception:
        # fallback: exponential (b≈0) by fixing b small
        def arps_exp(t, qi, Di):
            return qi * np.exp(-Di * t)
        popt_e, _ = curve_fit(arps_exp, t, y, p0=[qi0, 0.05], bounds=([1e-2, 1e-5], [1e6, 2.0]), maxfev=10000)
        qi, Di = float(popt_e[0]), float(popt_e[1])
        return qi, Di, 1e-3
    qi, Di, b = map(float, popt)
    return qi, Di, b

def make_dca_curve(df_well: pd.DataFrame, qi: float, Di: float, b: float, time_col: str) -> pd.Series:
    d = df_well.sort_values(time_col).copy()
    d["tmo"] = (d[time_col].dt.to_period("M") - d[time_col].dt.to_period("M").min()).apply(lambda x: x.n).astype(float)
    return pd.Series(arps_hyperbolic(d["tmo"].values, qi, Di, b), index=d[time_col])
