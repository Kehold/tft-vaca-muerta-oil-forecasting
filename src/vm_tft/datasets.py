from typing import Tuple, List, Optional
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer

from .cfg import (TIME_COL, GROUP_COL, TARGET, FREQ, STATIC_CATS, STATIC_REALS)

# --------- Reindex function for monthly frequency
def reindex_group(group, time_col='date', freq='MS', zero_fill_cols=["oil_rate_bpd","gas_rate_mscfd", "water_rate_bpd", "oil_bbl", "gas_mscf", "water_bbl", "operation_time"]):
    full_index = pd.date_range(start=group[time_col].min(), end=group[time_col].max(), freq=freq)
    reindexed = group.set_index(time_col).reindex(full_index).reset_index()
    # Fill specific columns with 0
    reindexed[zero_fill_cols] = reindexed[zero_fill_cols].fillna(0)
    # Forward-fill all other columns
    other_cols = [col for col in reindexed.columns if col not in zero_fill_cols and col != 'index']
    reindexed[other_cols] = reindexed[other_cols].ffill()
    return reindexed

def make_pipelines():
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scale", MinMaxScaler())])
    cat_enc  = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    return num_pipe, cat_enc

def ts_from_df(df: pd.DataFrame, cols: List[str]) -> List[TimeSeries]:
    return TimeSeries.from_group_dataframe(
        df=df[[TIME_COL, GROUP_COL] + cols].copy(),
        time_col=TIME_COL, group_cols=GROUP_COL, value_cols=cols, freq=FREQ
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
