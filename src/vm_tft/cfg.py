from pathlib import Path

# base dirs
PROJ_ROOT   = Path(__file__).resolve().parents[2]
DATA_DIR    = PROJ_ROOT / "data"
RAW_DIR     = DATA_DIR / "raw"
PROC_DIR    = DATA_DIR / "processed"
ARTIFACTS   = DATA_DIR / "artifacts"

# files
ALL_FILE    = PROC_DIR / "vm_tft_all.csv" 
TEST_FILE   = PROC_DIR / "vm_tft_test.csv" 

# modeling constants
TIME_COL    = "date"
GROUP_COL   = "group_id"
TARGET      = "oil_rate_bpd"
FREQ        = "MS"

INPUT_CHUNK = 24
FORECAST_H  = 6
N_WINDOWS   = 3
BUFFER      = 6

# covariates
STATIC_CATS  = ["operator", "area"]
STATIC_REALS = ["campaign", "horizontal_length","total_depth",
                "number_of_stages","fluid_volume_m3","proppant_volume_lbm",
                # "number_of_stages_missing","fluid_volume_m3_missing",
                # "proppant_volume_lbm_missing","total_depth_missing",
                # "qi","Di","b" # DCA constants
                ]

PAST_REALS   = ["gas_rate_mscfd", "water_rate_bpd",
                # "rate_rel_3m", "rate_rel_peak", "cum_oil_rel", 
                # "dca_bpd" # new variables
                ]   
FUTURE_REALS = ["well_age", 
                # "dca_bpd_log1p"
                ]

# training
N_EPOCHS    = 30
BATCH_SIZE  = 64
RANDOM_SEED = 42

# DCA
K_DCA = 24  # or 12/18; must be <= typical warm length
