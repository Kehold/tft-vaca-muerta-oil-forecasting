# src/vm_tft/cli/prepare_data.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from vm_tft.cfg import RAW_DIR, PROC_DIR, INPUT_CHUNK, FORECAST_H, N_WINDOWS, BUFFER, TARGET, K_DCA

from vm_tft.datasets import reindex_group
from vm_tft.io_utils import ensure_dir
from vm_tft.physics_dca import fit_arps_first_k_months, make_dca_curve

EPS = 1e-6

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
    # sanity: ensure processed dir exists
    ensure_dir(PROC_DIR)

    # raw files
    prod_csv  = RAW_DIR / "vm_production.csv"
    wells_csv = RAW_DIR / "vm_wells.csv"
    
    # processed files
    all_out  = PROC_DIR / "vm_tft_all.csv"
    test_out = PROC_DIR / "vm_tft_test.csv"

    if not prod_csv.exists() or not wells_csv.exists():
        raise FileNotFoundError(
            f"Missing raw files under {RAW_DIR}. "
            f"Expected: {prod_csv.name}, {wells_csv.name}"
        )

    # TODO: paste your full preparation pipeline here and write:
    
    # Load data
    prod = pd.read_csv(prod_csv)
    wells = pd.read_csv(wells_csv)
    
    prod['production_date'] = pd.to_datetime(prod['production_date'])
    wells['completion_date'] = pd.to_datetime(wells['completion_date'], errors='coerce')
    
    # Keep essentials
    keep_cols = ['well_id','well_name',  'production_date', 'eff_prod_day', 'horizontal_length', 'area', 
    'operator', 'well_type', 'produced_fluid', 'fluid_type',
    'campaign', 'number_of_stages', 'fluid_volume_m3', 'proppant_volume_lbm', 'oil_month_bbl', 'gas_month_mscf',
    'water_month_bbl', 'oil_month_bpd', 'gas_month_mscf_d', 'water_month_bpd', 'cum_oil_bbl', 'cum_gas_mscf', 'cum_water_bbl']
    
    for col in keep_cols:
        if col not in prod.columns:
            prod[col] = np.nan
    prod = prod[keep_cols].copy()
    
    # Standardize names
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
    
    # ---------- Merge wells metadata
    merged = prod.merge(
        wells[['well_id','completion_date','total_depth']].copy(),
        on='well_id', how='left'
    )
    
    # ---------- Filter Horizontal Wells
    merged = merged[merged['well_type']=='Horizontal']

    # ---------- Filter Horizontal Length > 800m
    merged = merged[merged['horizontal_length']>=800]


    # ---------- Filter Oil Wells
    merged = merged[merged['produced_fluid']=='Oil']

    # ---------- Filter Oil Wells
    merged = merged[merged['fluid_type']=='Black_Oil']

    # ---------- Campaign > 2015
    merged = merged[merged['campaign']>=2016]

    # ---------- Not plug and perf completions
    well_list = ['BCeCg-111(h)','BCeCf-101(h)', 'BCeCf-102(h)',
                'BCeCf-105(h)', 'BCeCg-112(h)', 'BCeCf-108(h)',
                'BCeCf-106(h)', 'LEsc.e-3(h)', 'LLL-1573(h)']

    merged = merged[~merged['well_name'].isin(well_list)]
    
    # ---------- Imputing missing values
    #-------- Creating Flags for Missing Values
    merged["fluid_volume_m3_missing"] = merged["fluid_volume_m3"].isna().astype(int)
    merged["proppant_volume_lbm_missing"] = merged["proppant_volume_lbm"].isna().astype(int)
    merged["number_of_stages_missing"] = merged["number_of_stages"].isna().astype(int)
    # merged["horizontal_length_missing"] = merged["horizontal_length"].isna().astype(int)
    merged["total_depth_missing"] = merged["total_depth"].isna().astype(int)
    
    #-------- Imputing fluid volume and proppant volume, horizontal wells
    #-------- Horizontal wells with NAN stages
    condition = (merged['well_type'] == 'Horizontal') & (merged['number_of_stages'].isna())
    columns_to_impute_first = ['fluid_volume_m3', 'proppant_volume_lbm']
    
    #-------- Fill NaN with 0 for fluid_volume_m3 and proppant_volume_lbm where condition is True
    merged.loc[condition, columns_to_impute_first] = merged.loc[condition, columns_to_impute_first].fillna(0)

    #-------- Imputing number of stages
    merged.loc[condition, 'number_of_stages'] = merged.loc[condition, 'number_of_stages'].fillna(0)

    #-------- Imputing fluid volume and proppant volume, horizontal wells
    #-------- Horizontal wells with NOT NaN stages
    condition = (merged['well_type'] == 'Horizontal') & (merged['number_of_stages'] != 0) & (~merged['number_of_stages'].isna())
    
    valid_wells = merged[condition & (~merged['fluid_volume_m3'].isna()) & (~merged['proppant_volume_lbm'].isna())]
    
    avg_fluid_per_stage = (valid_wells['fluid_volume_m3'] / valid_wells['number_of_stages']).mean()
    
    avg_proppant_per_stage = (valid_wells['proppant_volume_lbm'] / valid_wells['number_of_stages']).mean()
    
    print(f"Average fluid per stage: {avg_fluid_per_stage:.2f} m3/stage")
    print(f"Average proppant per stage: {avg_proppant_per_stage:.2f} lbm/stage")
    
    #-------- Imputing fluid volume and proppant volume, horizontal wells
    #-------- Apply imputation only where fluid_volume_m3 or proppant_volume_lbm is NaN
    impute_condition = condition & (merged['fluid_volume_m3'].isna() | merged['proppant_volume_lbm'].isna())
    
    merged.loc[impute_condition, 'fluid_volume_m3'] = merged.loc[impute_condition, 'fluid_volume_m3'].fillna(
        merged['number_of_stages'] * avg_fluid_per_stage)
    
    merged.loc[impute_condition, 'proppant_volume_lbm'] = merged.loc[impute_condition, 'proppant_volume_lbm'].fillna(
    merged['number_of_stages'] * avg_proppant_per_stage)
    
    #-------- Imputing total_depth
    mean_value = merged['total_depth'].mean()
    merged["total_depth"] = merged["total_depth"].fillna(mean_value)
    
    # ---------- Add relative features (for PAST covariates) ----------
    merged = _add_relative_features(
        df=merged, group_col='well_id', time_col='date', target_col=TARGET  # TARGET == "oil_rate_bpd"
    )
    
    
    # ---------- Add DCA features (for PAST covariates) ----------
    # K_DCA = 24  or 12/18; must be <= typical warm length
    rows = []
    dca_curves = []

    for wid, dfw in merged.groupby("well_id"):
        try:
            qi, Di, b = fit_arps_first_k_months(dfw, k=K_DCA, target_col="oil_rate_bpd", time_col="date")
            dca_series = make_dca_curve(dfw, qi, Di, b, time_col="date").rename("dca_bpd")
            dca_curves.append(
                pd.DataFrame({"well_id": wid, "date": dca_series.index, "dca_bpd": dca_series.values})
            )
            rows.append({"well_id": wid, "qi": qi, "Di": Di, "b": b})
        except Exception:
            # If fit fails, leave NaNs; model can rely on other covariates
            pass

    dca_df = pd.concat(dca_curves, ignore_index=True) if dca_curves else pd.DataFrame(columns=["well_id","date","dca_bpd"])
    pars_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["well_id","qi","Di","b"])

    # merge back
    merged_dca = merged.merge(dca_df, on=["well_id","date"], how="left").merge(pars_df, on="well_id", how="left")

    
    # -------------------
    # Filter wells with enough history (train set)
    # -------------------
    need_len = INPUT_CHUNK + FORECAST_H * N_WINDOWS + BUFFER
    n_per_well = merged_dca.groupby('well_id')['date'].transform("size")
    merged_train = merged_dca[n_per_well >= need_len].copy()
    
    # -------------------
    # Filter wells with erratic history
    # -------------------
    
    bad_wells = [154212,154777,154790,154929,154931,
             155239,156489,156698,156759,156967,
             157331,157862,157871,157872,157973,
             158756,158894,159136,
             159280,159314,159381,159464,159501,
             159581,159884,160051,160146,160742,
             161460,161513,161531,161533,161603,
             161845,
             ]
    
    merged_train = merged_train.loc[~merged_train.well_id.isin(bad_wells)]
    
    # -------------------
    # Filter erratic history
    # -------------------
    merged_train = merged_train[~ ((merged_train['well_id']==156332) & (merged_train['date']>'2022-01-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156334) & (merged_train['date']>'2023-06-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156475) & (merged_train['date']>'2023-08-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156491) & (merged_train['date']>'2022-05-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156492) & (merged_train['date']>'2022-12-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156493) & (merged_train['date']>'2021-11-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156339) & (merged_train['date']>'2023-02-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156553) & (merged_train['date']>'2022-09-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156707) & (merged_train['date']>'2022-05-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156710) & (merged_train['date']>'2022-05-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156757) & (merged_train['date']>'2022-08-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156758) & (merged_train['date']>'2022-05-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156760) & (merged_train['date']>'2022-01-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156761) & (merged_train['date']>'2022-10-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156969) & (merged_train['date']>'2022-10-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==156973) & (merged_train['date']>'2022-07-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157112) & (merged_train['date']>'2022-11-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157113) & (merged_train['date']>'2023-02-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157114) & (merged_train['date']>'2021-11-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157190) & (merged_train['date']>'2022-10-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157191) & (merged_train['date']>'2022-11-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157329) & (merged_train['date']>'2022-02-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157330) & (merged_train['date']>'2021-07-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157334) & (merged_train['date']>'2022-02-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157336) & (merged_train['date']>'2022-02-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157873) & (merged_train['date']>'2021-07-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157875) & (merged_train['date']>'2023-02-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157876) & (merged_train['date']>'2023-02-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157972) & (merged_train['date']>'2021-12-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157977) & (merged_train['date']>'2023-03-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157978) & (merged_train['date']>'2023-09-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==157986) & (merged_train['date']>'2022-08-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158042) & (merged_train['date']>'2023-08-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158046) & (merged_train['date']>'2022-01-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158148) & (merged_train['date']>'2021-12-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158149) & (merged_train['date']>'2022-01-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158150) & (merged_train['date']>'2022-05-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158151) & (merged_train['date']>'2024-05-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158152) & (merged_train['date']>'2024-01-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158427) & (merged_train['date']>'2023-07-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158443) & (merged_train['date']>'2024-03-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158560) & (merged_train['date']>'2023-03-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158561) & (merged_train['date']>'2023-01-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158753) & (merged_train['date']>'2022-12-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158756) & (merged_train['date']>'2023-03-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158757) & (merged_train['date']>'2024-03-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158895) & (merged_train['date']>'2022-06-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==158974) & (merged_train['date']>'2024-10-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==159075) & (merged_train['date']>'2024-09-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==159135) & (merged_train['date']>'2024-01-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==159222) & (merged_train['date']>'2024-10-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==159223) & (merged_train['date']>'2024-01-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==159224) & (merged_train['date']>'2024-02-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==159226) & (merged_train['date']>'2022-11-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==159227) & (merged_train['date']>'2024-11-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==159281) & (merged_train['date']>'2023-10-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==159379) & (merged_train['date']>'2022-08-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==159493) & (merged_train['date']>'2023-09-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==159710) & (merged_train['date']>'2024-11-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==159847) & (merged_train['date']>'2024-11-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==160053) & (merged_train['date']>'2022-06-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==160054) & (merged_train['date']>'2024-08-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==160148) & (merged_train['date']>'2024-11-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==160333) & (merged_train['date']>'2024-09-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==160432) & (merged_train['date']>'2024-08-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==160436) & (merged_train['date']>'2024-09-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==160575) & (merged_train['date']>'2024-07-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==160755) & (merged_train['date']>'2023-11-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==160866) & (merged_train['date']>'2024-09-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==160925) & (merged_train['date']>'2024-10-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==160930) & (merged_train['date']>'2024-11-01'))]
    merged_train = merged_train[~ ((merged_train['well_id']==161374) & (merged_train['date']>'2024-10-01'))]


    # Apply reindexing per well_id
    df_train_reindexed = merged_train.groupby('well_id').apply(lambda x: reindex_group(x, 'date', freq='MS')).reset_index(drop=True)

    df_train_reindexed = df_train_reindexed.rename(columns={'index': 'date'})
    df_train_reindexed['well_id'] = df_train_reindexed['well_id'].astype(int)
    
    # -------------------
    # Filter wells with enough history (after re-index)
    # -------------------
    need_len = INPUT_CHUNK + FORECAST_H * N_WINDOWS + BUFFER
    n_per_well = df_train_reindexed.groupby('well_id')['date'].transform("size")
    df_train_reindexed = df_train_reindexed[n_per_well >= need_len].copy()

    print(f"Remaining wells: {df_train_reindexed['well_id'].nunique()}")
    print(f"Months per well mean: {df_train_reindexed.groupby('well_id')['date'].count().mean()}")
    print(f"Months per well min: {df_train_reindexed.groupby('well_id')['date'].count().min()}")
    print(f"Months per well max: {df_train_reindexed.groupby('well_id')['date'].count().max()}")
    
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
    
    # ---------------------------------------
    # De-duplicate to a single record per (well_id, month)
    df_train_reindexed = df_train_reindexed.sort_values(['well_id','date'])
    df_train_reindexed = df_train_reindexed.drop_duplicates(subset=['well_id','date'], keep='last')
    
    # ---------- Indices for TFT
    df_train_reindexed = df_train_reindexed.sort_values(['well_id','date']).reset_index(drop=True)
    df_train_reindexed['time_idx'] = df_train_reindexed.groupby('well_id').cumcount()
    df_train_reindexed['group_id'] = df_train_reindexed['well_id']
    
    # ---------- Save train / validation set
    df_train_reindexed.to_csv(all_out, index=False)
    
    
    # -------------------
    # Filter test set
    # -------------------
    
    need_len = INPUT_CHUNK + FORECAST_H * N_WINDOWS + BUFFER
    n_per_well = merged_dca.groupby('well_id')['date'].transform("size")
    merged_test = merged_dca[(n_per_well >= need_len-BUFFER) & (n_per_well < need_len)].copy()
    
    bad_wells = [157567,160943,]

    merged_test = merged_test.loc[~merged_test.well_id.isin(bad_wells)]
    
    # Apply reindexing per well_id
    df_test_reindexed = merged_test.groupby('well_id').apply(lambda x: reindex_group(x, 'date', freq='MS')).reset_index(drop=True)
    df_test_reindexed = df_test_reindexed.rename(columns={'index': 'date'})
    df_test_reindexed['well_id'] = df_test_reindexed['well_id'].astype(int)
    
    # ---------------------------------------
    # De-duplicate to a single record per (well_id, month)
    df_test_reindexed = df_test_reindexed.sort_values(['well_id','date'])
    df_test_reindexed = df_test_reindexed.drop_duplicates(subset=['well_id','date'], keep='last')
    
    # Engineer well_age
    # Drop pre-completion months if negative ages would occur
    # Engineer well_age using month difference 
    df_test_reindexed['well_age'] = (
        (df_test_reindexed['date'].dt.to_period("M") -
        df_test_reindexed['completion_date'].dt.to_period("M"))
        .apply(lambda x: x.n)
    )

    # Drop negatives (pre-completion months)
    df_test_reindexed = df_test_reindexed[df_test_reindexed['well_age'].isna() | (df_test_reindexed['well_age'] >= 0)]

    # Clip just in case (keeps 0 as min)
    df_test_reindexed['well_age'] = df_test_reindexed['well_age'].clip(lower=0)
    
    # ---------- Indices for TFT
    df_test_reindexed = df_test_reindexed.sort_values(['well_id','date']).reset_index(drop=True)
    df_test_reindexed['time_idx'] = df_test_reindexed.groupby('well_id').cumcount()
    df_test_reindexed['group_id'] = df_test_reindexed['well_id']
    
    print(f"Remaining wells: {df_test_reindexed['well_id'].nunique()}")
    print(f"Months per well mean: {df_test_reindexed.groupby('well_id')['date'].count().mean()}")
    print(f"Months per well min: {df_test_reindexed.groupby('well_id')['date'].count().min()}")
    print(f"Months per well max: {df_test_reindexed.groupby('well_id')['date'].count().max()}")
    
    # ---------- Save test set
    df_test_reindexed.to_csv(test_out, index=False)

    print(f"✅ Data prepared → {all_out.name}, {test_out.name} in {PROC_DIR}")

if __name__ == "__main__":
    main()