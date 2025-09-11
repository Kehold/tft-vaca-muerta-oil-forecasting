# src/vm_tft/cli/prepare_data.py
# Create a patched, cleaner version of prepare_data_.py with:
# - Relative features (rate_rel_3m, rate_rel_peak, cum_oil_rel)
# - Centralized handling for "bad wells" and "bad date intervals"
# - Clear patch markers to show where to replace blocks in the user's file
# - Writes processed CSVs to PROC_DIR and keeps the same interface

# src/vm_tft/cli/prepare_data.py
from __future__ import annotations
import pandas as pd
import numpy as np

from vm_tft.cfg import (
    RAW_DIR, PROC_DIR, ALL_FILE, TEST_FILE,
    INPUT_CHUNK, FORECAST_H, N_WINDOWS, BUFFER,
    TIME_COL, TARGET
)
from vm_tft.datasets import reindex_group
from vm_tft.io_utils import ensure_dir

# ==============================
# Configuration blocks (TRAIN)
# ==============================

# Train Set bad wells (EXACTLY as provided)
BAD_WELLS: set[int] = {
    154212,154777,154790,154929,154931,
    155239,156489,156698,156759,156967,
    157331,157862,157871,157872,157973,
    158756,158894,159136,
    159280,159314,159381,159464,159501,
    159581,159884,160051,160146,160742,
    161460,161513,161531,161533,161603,
    161845,
}

# “Bad date” intervals replacing row-wise filters; (start_inclusive, end_inclusive)
BAD_DATE_INTERVALS: dict[int, list[tuple[str, str]]] = {
    156332: [("2022-01-01", "2100-01-01")],
    156334: [("2023-06-01", "2100-01-01")],
    156475: [("2023-08-01", "2100-01-01")],
    156491: [("2022-05-01", "2100-01-01")],
    156492: [("2022-12-01", "2100-01-01")],
    156493: [("2021-11-01", "2100-01-01")],
    156339: [("2023-02-01", "2100-01-01")],
    156553: [("2022-09-01", "2100-01-01")],
    156707: [("2022-05-01", "2100-01-01")],
    156710: [("2022-05-01", "2100-01-01")],
    156757: [("2022-08-01", "2100-01-01")],
    156758: [("2022-05-01", "2100-01-01")],
    156760: [("2022-01-01", "2100-01-01")],
    156761: [("2022-10-01", "2100-01-01")],
    156969: [("2022-10-01", "2100-01-01")],
    156973: [("2022-07-01", "2100-01-01")],
    157112: [("2022-11-01", "2100-01-01")],
    157113: [("2023-02-01", "2100-01-01")],
    157114: [("2021-11-01", "2100-01-01")],
    157190: [("2022-10-01", "2100-01-01")],
    157191: [("2022-11-01", "2100-01-01")],
    157329: [("2022-02-01", "2100-01-01")],
    157330: [("2021-07-01", "2100-01-01")],
    157334: [("2022-02-01", "2100-01-01")],
    157336: [("2022-02-01", "2100-01-01")],
    157873: [("2021-07-01", "2100-01-01")],
    157875: [("2023-02-01", "2100-01-01")],
    157876: [("2023-02-01", "2100-01-01")],
    157972: [("2021-12-01", "2100-01-01")],
    157977: [("2023-03-01", "2100-01-01")],
    157978: [("2023-09-01", "2100-01-01")],
    157986: [("2022-08-01", "2100-01-01")],
    158042: [("2023-08-01", "2100-01-01")],
    158046: [("2022-01-01", "2100-01-01")],
    158148: [("2021-12-01", "2100-01-01")],
    158149: [("2022-01-01", "2100-01-01")],
    158150: [("2022-05-01", "2100-01-01")],
    158151: [("2024-05-01", "2100-01-01")],
    158152: [("2024-01-01", "2100-01-01")],
    158427: [("2023-07-01", "2100-01-01")],
    158443: [("2024-03-01", "2100-01-01")],
    158560: [("2023-03-01", "2100-01-01")],
    158561: [("2023-01-01", "2100-01-01")],
    158753: [("2022-12-01", "2100-01-01")],
    158756: [("2023-03-01", "2100-01-01")],
    158757: [("2024-03-01", "2100-01-01")],
    158895: [("2022-06-01", "2100-01-01")],
    158974: [("2024-10-01", "2100-01-01")],
    159075: [("2024-09-01", "2100-01-01")],
    159135: [("2024-01-01", "2100-01-01")],
    159222: [("2024-10-01", "2100-01-01")],
    159223: [("2024-01-01", "2100-01-01")],
    159224: [("2024-02-01", "2100-01-01")],
    159226: [("2022-11-01", "2100-01-01")],
    159227: [("2024-11-01", "2100-01-01")],
    159281: [("2023-10-01", "2100-01-01")],
    159379: [("2022-08-01", "2100-01-01")],
    159493: [("2023-09-01", "2100-01-01")],
    159710: [("2024-11-01", "2100-01-01")],
    159847: [("2024-11-01", "2100-01-01")],
    160053: [("2022-06-01", "2100-01-01")],
    160054: [("2024-08-01", "2100-01-01")],
    160148: [("2024-11-01", "2100-01-01")],
    160333: [("2024-09-01", "2100-01-01")],
    160432: [("2024-08-01", "2100-01-01")],
    160436: [("2024-09-01", "2100-01-01")],
    160575: [("2024-07-01", "2100-01-01")],
    160755: [("2023-11-01", "2100-01-01")],
    160866: [("2024-09-01", "2100-01-01")],
    160925: [("2024-10-01", "2100-01-01")],
    160930: [("2024-11-01", "2100-01-01")],
    161374: [("2024-10-01", "2100-01-01")],
}

# Wells to exclude by name (non PnP completions, etc.)
EXCLUDE_WELL_NAMES: set[str] = {
    'BCeCg-111(h)','BCeCf-101(h)','BCeCf-102(h)',
    'BCeCf-105(h)','BCeCg-112(h)','BCeCf-108(h)',
    'BCeCf-106(h)','LEsc.e-3(h)','LLL-1573(h)'
}

EPS = 1e-6


def _drop_bad_date_intervals(df: pd.DataFrame, well_col: str, date_col: str) -> pd.DataFrame:
    """Remove rows that fall inside BAD_DATE_INTERVALS for their well_id."""
    if not BAD_DATE_INTERVALS:
        return df
    df = df.copy()
    mask = pd.Series(False, index=df.index)
    for wid, ranges in BAD_DATE_INTERVALS.items():
        m = (df[well_col] == wid)
        if not m.any():
            continue
        for start, end in ranges:
            start = pd.to_datetime(start)
            end   = pd.to_datetime(end)
            mask |= (m & (df[date_col] >= start) & (df[date_col] <= end))
    return df.loc[~mask]


def _add_relative_features(df: pd.DataFrame, group_col: str, time_col: str, target_col: str) -> pd.DataFrame:
    """Add rate_rel_3m, rate_rel_peak, cum_oil_rel (if cum column exists)."""
    df = df.sort_values([group_col, time_col]).copy()

    def first3_mean(s: pd.Series) -> float:
        return s.head(3).mean()

    first3 = (df.groupby(group_col)[target_col].apply(first3_mean).rename("first3_avg"))
    peak = df.groupby(group_col)[target_col].max().rename("peak")

    df = df.merge(first3, on=group_col).merge(peak, on=group_col)
    df["rate_rel_3m"]   = df[target_col] / (df["first3_avg"] + EPS)
    df["rate_rel_peak"] = df[target_col] / (df["peak"] + EPS)

    if "cum_oil_bbl" in df.columns:
        def cum_at_3m(s: pd.Series) -> float:
            if len(s) >= 3:
                return float(s.iloc[2])
            return float(s.iloc[-1]) if len(s) else np.nan
        first3_cum = df.groupby(group_col)["cum_oil_bbl"].transform(cum_at_3m)
        df["cum_oil_rel"] = df["cum_oil_bbl"] / (first3_cum + EPS)
    else:
        df["cum_oil_rel"] = np.nan

    return df


def main() -> None:
    ensure_dir(PROC_DIR)

    # raw files
    prod_csv  = RAW_DIR / "vm_production.csv"
    wells_csv = RAW_DIR / "vm_wells.csv"

    if not prod_csv.exists() or not wells_csv.exists():
        raise FileNotFoundError(
            f"Missing raw files under {RAW_DIR}. "
            f"Expected: {prod_csv.name}, {wells_csv.name}"
        )

    # ---------- Load ----------
    prod = pd.read_csv(prod_csv)
    wells = pd.read_csv(wells_csv)

    prod['production_date'] = pd.to_datetime(prod['production_date'])
    wells['completion_date'] = pd.to_datetime(wells['completion_date'], errors='coerce')

    # ---------- Keep essentials ----------
    keep_cols = [
        'well_id','well_name','production_date','eff_prod_day','horizontal_length','area',
        'operator','well_type','produced_fluid','fluid_type','campaign','number_of_stages',
        'fluid_volume_m3','proppant_volume_lbm','oil_month_bbl','gas_month_mscf','water_month_bbl',
        'oil_month_bpd','gas_month_mscf_d','water_month_bpd','cum_oil_bbl','cum_gas_mscf','cum_water_bbl'
    ]
    for col in keep_cols:
        if col not in prod.columns:
            prod[col] = np.nan

    prod = prod[keep_cols].copy()

    # ---------- Standardize names ----------
    prod = prod.rename(columns={
        'production_date':'date',
        'oil_month_bbl':'oil_bbl',
        'gas_month_mscf':'gas_mscf',
        'water_month_bbl':'water_bbl',
        'oil_month_bpd':'oil_rate_bpd',
        'gas_month_mscf_d':'gas_rate_mscfd',
        'water_month_bpd':'water_rate_bpd',
        'eff_prod_day':'operation_time',
    })

    # ---------- Merge wells metadata ----------
    merged = prod.merge(
        wells[['well_id','completion_date','total_depth']].copy(),
        on='well_id', how='left'
    )

    # ---------- Base filters ----------
    merged = merged[merged['well_type'] == 'Horizontal']
    merged = merged[merged['horizontal_length'] >= 800]
    merged = merged[merged['produced_fluid'] == 'Oil']
    merged = merged[merged['fluid_type'] == 'Black_Oil']
    merged = merged[merged['campaign'] >= 2016]
    merged = merged[~merged['well_name'].isin(EXCLUDE_WELL_NAMES)]

    # ---------- Imputation + missing flags ----------
    merged["fluid_volume_m3_missing"]      = merged["fluid_volume_m3"].isna().astype(int)
    merged["proppant_volume_lbm_missing"]  = merged["proppant_volume_lbm"].isna().astype(int)
    merged["number_of_stages_missing"]     = merged["number_of_stages"].isna().astype(int)
    merged["total_depth_missing"]          = merged["total_depth"].isna().astype(int)

    # Horizontal wells with NaN stages → set stages=0 and volumes=0
    cond_nan_stages = (merged['well_type'] == 'Horizontal') & (merged['number_of_stages'].isna())
    merged.loc[cond_nan_stages, ['fluid_volume_m3','proppant_volume_lbm']] = 0
    merged.loc[cond_nan_stages, 'number_of_stages'] = 0

    # Compute typical per-stage volumes using valid wells
    cond_valid_stages = (
        (merged['well_type'] == 'Horizontal') &
        (merged['number_of_stages'].notna()) &
        (merged['number_of_stages'] != 0)
    )
    valid = merged[cond_valid_stages & merged['fluid_volume_m3'].notna() & merged['proppant_volume_lbm'].notna()]
    avg_fluid_per_stage = (valid['fluid_volume_m3'] / valid['number_of_stages']).mean()
    avg_prop_per_stage  = (valid['proppant_volume_lbm'] / valid['number_of_stages']).mean()

    print(f"Average fluid per stage: {avg_fluid_per_stage:.2f} m3/stage")
    print(f"Average proppant per stage: {avg_prop_per_stage:.2f} lbm/stage")

    # Impute missing volumes where stages>0
    cond_impute = cond_valid_stages & (merged['fluid_volume_m3'].isna() | merged['proppant_volume_lbm'].isna())
    merged.loc[cond_impute, 'fluid_volume_m3'] = merged.loc[cond_impute, 'fluid_volume_m3'] \
        .fillna(merged['number_of_stages'] * avg_fluid_per_stage)
    merged.loc[cond_impute, 'proppant_volume_lbm'] = merged.loc[cond_impute, 'proppant_volume_lbm'] \
        .fillna(merged['number_of_stages'] * avg_prop_per_stage)

    # total_depth
    merged["total_depth"] = merged["total_depth"].fillna(merged["total_depth"].mean())
    
    # -------------------
    # Filter wells with enough history (train set)
    # -------------------
    need_len = INPUT_CHUNK + FORECAST_H * N_WINDOWS + BUFFER
    n_per_well = merged.groupby('well_id')['date'].transform("size")
    merged_train = merged[n_per_well >= need_len].copy()
    

    # ---------- Drop bad wells and bad dates ----------
    merged = merged.loc[~merged['well_id'].isin(BAD_WELLS)].copy()
    merged['date'] = pd.to_datetime(merged['date'])
    merged = _drop_bad_date_intervals(merged, well_col='well_id', date_col='date')

    # ---------- Add relative features (for PAST covariates) ----------
    merged_train = _add_relative_features(
        df=merged, group_col='well_id', time_col='date', target_col=TARGET  # TARGET == "oil_rate_bpd"
    )
    
    # Apply reindexing per well_id
    df_train_reindexed = merged_train.groupby('well_id').apply(lambda x: reindex_group(x, 'date', freq='MS')).reset_index(drop=True)

    df_train_reindexed = df_train_reindexed.rename(columns={'index': 'date'})
    df_train_reindexed['well_id'] = df_train_reindexed['well_id'].astype(int)

    # ===================
    # TRAIN (“all”) set
    # ===================
    need_len = INPUT_CHUNK + FORECAST_H * N_WINDOWS + BUFFER
    n_per_well = df_train_reindexed.groupby('well_id')['date'].transform("size")
    df_train_reindexed = df_train_reindexed[n_per_well >= need_len].copy()
    
    print(f"[TRAIN] Wells: {df_train_reindexed['well_id'].nunique()}")
    print(f"[TRAIN] Months mean/min/max: "
        f"{df_train_reindexed.groupby('well_id')['date'].count().mean():.1f} / "
        f"{df_train_reindexed.groupby('well_id')['date'].count().min()} / "
        f"{df_train_reindexed.groupby('well_id')['date'].count().max()}")
    
        # Engineer well_age using month difference 
    df_train_reindexed['well_age'] = (
        (df_train_reindexed['date'].dt.to_period("M") -
        df_train_reindexed['completion_date'].dt.to_period("M"))
        .apply(lambda x: x.n)
    )
    
    # Drop negatives (pre-completion months)
    df_train_reindexed = df_train_reindexed[df_train_reindexed['well_age'].isna() | (df_train_reindexed['well_age'] >= 0)]

    # Clip just in case (keeps 0 as min)
    df_train_reindexed['well_age'] = df_train_reindexed['well_age'].clip(lower=0)
    
    # De-duplicate to a single record per (well_id, month)
    df_train_reindexed = df_train_reindexed.sort_values(['well_id','date'])
    df_train_reindexed = df_train_reindexed.drop_duplicates(subset=['well_id','date'], keep='last')
    

    # Emit TFT-friendly indices/IDs
    df_train_reindexed = df_train_reindexed.sort_values(['well_id','date']).reset_index(drop=True)
    df_train_reindexed['group_id'] = df_train_reindexed['well_id']    # match cfg.GROUP_COL
    df_train_reindexed['time_idx'] = df_train_reindexed.groupby('well_id').cumcount()


    df_train_reindexed.to_csv(ALL_FILE, index=False)
    print(f"Wrote TRAIN (‘all’) → {ALL_FILE}")

    # ===================
    # TEST set (per your spec)
    # ===================
    n_per_well_all = merged.groupby('well_id')['date'].transform("size")
    merged_test = merged[(n_per_well_all >= need_len - BUFFER) & (n_per_well_all < need_len)].copy()

    # Additional test-only bad wells
    TEST_BAD_WELLS = {157567,160943,}
    merged_test = merged_test.loc[~merged_test['well_id'].isin(TEST_BAD_WELLS)]

    # Reindex to MS per well, deduplicate, engineer well_age
    df_test_reindexed = (
        merged_test
        .groupby('well_id', group_keys=False)
        .apply(lambda x: reindex_group(x, 'date', freq='MS'))
        .reset_index(drop=True)
    )
    df_test_reindexed = df_test_reindexed.rename(columns={'index': 'date'})
    df_test_reindexed['well_id'] = df_test_reindexed['well_id'].astype(int)
    df_test_reindexed = df_test_reindexed.sort_values(['well_id','date'])
    df_test_reindexed = df_test_reindexed.drop_duplicates(subset=['well_id','date'], keep='last')

    # well_age in months since completion; drop negative
    df_test_reindexed['well_age'] = (
        (df_test_reindexed['date'].dt.to_period("M") -
         df_test_reindexed['completion_date'].dt.to_period("M"))
        .apply(lambda x: x.n)
    )
    df_test_reindexed = df_test_reindexed[
        df_test_reindexed['well_age'].isna() | (df_test_reindexed['well_age'] >= 0)
    ]
    df_test_reindexed['well_age'] = df_test_reindexed['well_age'].clip(lower=0)

    # TFT indices/IDs
    df_test_reindexed = df_test_reindexed.sort_values(['well_id','date']).reset_index(drop=True)
    df_test_reindexed['group_id'] = df_test_reindexed['well_id']
    df_test_reindexed['time_idx'] = df_test_reindexed.groupby('well_id').cumcount()

    print(f"[TEST] Wells: {df_test_reindexed['well_id'].nunique()}")
    print(f"[TEST] Months mean: {df_test_reindexed.groupby('well_id')['date'].count().mean():.1f}")

    df_test_reindexed.to_csv(TEST_FILE, index=False)
    print(f"✅ Data prepared → {ALL_FILE.name}, {TEST_FILE.name} in {PROC_DIR}")


if __name__ == "__main__":
    main()

