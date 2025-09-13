# src/vm_tft/datasets.py
from typing import Tuple, List, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer

from .cfg import TIME_COL, GROUP_COL, TARGET, FREQ, STATIC_CATS, STATIC_REALS

# Columns that should be zero-filled when reindexing to monthly frequency
ZERO_FILL_COLS: List[str] = [
    "oil_rate_bpd", "gas_rate_mscfd", "water_rate_bpd",
    "oil_bbl", "gas_mscf", "water_bbl", "operation_time",
]

# --------- Reindex function for monthly frequency
def reindex_group(
    group: pd.DataFrame,
    *,
    time_col: str = TIME_COL,
    freq: str = "MS",
    zero_fill_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Reindex a single-well DataFrame to a **complete monthly** time grid.

    - Creates a continuous monthly DatetimeIndex from the well's min→max date.
    - Zero-fills rate/volume columns (configurable).
    - Forward-fills all other columns to keep latest known metadata.

    Parameters
    ----------
    group : pd.DataFrame
        Rows for a single well (already grouped). Must contain `time_col`.
    time_col : str, default=TIME_COL
        Name of the time column (datetime-like).
    freq : str, default="MS"
        Pandas offset alias (monthly start).
    zero_fill_cols : Optional[Sequence[str]]
        Columns to fill with 0 after reindex. If None, uses ZERO_FILL_COLS.

    Returns
    -------
    pd.DataFrame
        Reindexed DataFrame with the same columns, monthly frequency, and no gaps.
    """
    zcols = list(zero_fill_cols) if zero_fill_cols is not None else ZERO_FILL_COLS

    # ensure datetime
    g = group.copy()
    g[time_col] = pd.to_datetime(g[time_col])

    # build full monthly index and reindex
    full_idx = pd.date_range(start=g[time_col].min(), end=g[time_col].max(), freq=freq)
    reindexed = (
        g.set_index(time_col)
         .reindex(full_idx)              # introduces NaNs for newly created months
         .reset_index()
         .rename(columns={"index": time_col})
    )

    # zero-fill specific numeric columns if present
    cols_present = [c for c in zcols if c in reindexed.columns]
    if cols_present:
        reindexed[cols_present] = reindexed[cols_present].fillna(0)

    # forward-fill the rest (metadata / static info)
    other_cols = [c for c in reindexed.columns if c not in cols_present and c != time_col]
    if other_cols:
        reindexed[other_cols] = reindexed[other_cols].ffill()

    return reindexed


# --------- Make pipelines
def make_pipelines() -> Tuple[Pipeline, OrdinalEncoder]:
    """
    Build scikit-learn transformers used for **static covariates**:

    - Numeric: median imputation → MinMax scaling.
    - Categorical: ordinal encoding with a reserved code for unknown categories.

    Returns
    -------
    (num_pipe, cat_enc)
        num_pipe : Pipeline
            Applies median imputation then MinMax scaling.
        cat_enc : OrdinalEncoder
            Encodes categories; unseen categories map to -1.
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", MinMaxScaler()),
    ])
    cat_enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.int64,
    )
    return num_pipe, cat_enc
    
# --------- DataFrame to Darts TimeSeries
def ts_from_df(df: pd.DataFrame, cols: Sequence[str]) -> Optional[List[TimeSeries]]:
    """
    Convert a wide DataFrame into a **grouped list of Darts TimeSeries**.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain TIME_COL and GROUP_COL and the requested `cols`.
    cols : Sequence[str]
        Value columns to extract. If empty, returns None (safe no-op).

    Returns
    -------
    Optional[List[TimeSeries]]
        One TimeSeries per group_id. None if `cols` is empty.
    """
    if not cols:
        return None  # avoid Darts error on empty value_cols

    return TimeSeries.from_group_dataframe(
        df=df[[TIME_COL, GROUP_COL] + list(cols)].copy(),
        time_col=TIME_COL,
        group_cols=GROUP_COL,
        value_cols=list(cols),
        freq=FREQ,
    )


def build_all_timeseries(df_all: pd.DataFrame,
                         df_future_cov: Optional[pd.DataFrame]) -> Tuple[List[TimeSeries], Optional[List[TimeSeries]], Optional[List[TimeSeries]], StaticCovariatesTransformer, Scaler, Scaler, Scaler]:
    # target + statics
    series = TimeSeries.from_group_dataframe(
        df=df_all[[TIME_COL, GROUP_COL, TARGET] + STATIC_CATS + STATIC_REALS].copy(),
        time_col=TIME_COL, group_cols=GROUP_COL, value_cols=TARGET,
        static_cols=STATIC_CATS + STATIC_REALS, freq=FREQ
    )
    past = ts_from_df(df_all, [])  # placeholder if you later want past covs from df_all
    # past covs: pass explicitly if you need; for now keep external

    future = None
    if df_future_cov is not None:
        future = TimeSeries.from_group_dataframe(
            df=df_future_cov, time_col=TIME_COL, group_cols=GROUP_COL,
            value_cols=["well_age"], freq=FREQ
        )

    num_pipe, cat_enc = make_pipelines()
    static_tr = StaticCovariatesTransformer(
        transformer_num=num_pipe, transformer_cat=cat_enc,
        cols_num=STATIC_REALS, cols_cat=STATIC_CATS,
    )
    series = static_tr.fit_transform(series)

    y_scaler = Scaler(); series_scaled = y_scaler.fit_transform(series)
    # past/future scalers are created by caller (you already have this pattern)
    return series_scaled, static_tr, y_scaler
