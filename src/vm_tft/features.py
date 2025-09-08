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
