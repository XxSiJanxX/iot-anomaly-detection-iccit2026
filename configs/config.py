"""
Central configuration for all project parameters.
Edit paths and hyperparameters here - nowhere else.
"""

import os

# ---------------------------------------------
# PATHS
# ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
DATA_CROSS = os.path.join(BASE_DIR, "data", "cross_dataset")

# REVISION (camera-ready): all outputs of the corrected pipeline go to a
# separate folder so the original submission's outputs/ stay untouched and
# no stale model .pkl from the old pipeline can be silently re-loaded by the
# per-classifier resume logic. Override with the IOT_OUTPUT_DIR env variable.
OUT_ROOT    = os.environ.get("IOT_OUTPUT_DIR",
                             os.path.join(BASE_DIR, "outputs_revision"))
OUT_RESULTS = os.path.join(OUT_ROOT, "results")
OUT_PLOTS   = os.path.join(OUT_ROOT, "plots")
OUT_MODELS  = os.path.join(OUT_ROOT, "models")
OUT_LOGS    = os.path.join(OUT_ROOT, "logs")
OUT_STATE   = os.path.join(OUT_ROOT, "run_state")   # resume markers (run_all.py)

# Row cap applied ONCE, before de-duplication and splitting (CICIoT2023 only)
MAX_ROWS = 10_000_000

# ---------------------------------------------
# DATASET SETTINGS
# Each dataset entry:
#   "label_col"   : column name for the binary label (0=normal, 1=attack)
#   "drop_cols"   : identifier columns to always remove
#   "subsample"   : training subsample for LinearSVC / KNN (None = full set)
# ---------------------------------------------
DATASETS = {
    "nbaiot": {
        "label_col": "label",          # 0 = benign, 1 = attack (after encoding)
        "drop_cols": [],
        "subsample": 500_000,
        "raw_subdir": "nbaiot",        # data/raw/nbaiot/
    },
    "ciciot2023": {
        "label_col": "label",
        "drop_cols": [],
        "subsample": 500_000,
        "raw_subdir": "ciciot2023",
    },
    "iot23": {
        "label_col": "label",
        # No known leakage columns documented for IoT-23 at this time.
        "drop_cols": [],
        "subsample": 500_000,
        "raw_subdir": "iot23",
    },
}

DATASET_NAMES = list(DATASETS.keys())   # ["nbaiot", "unsw", "ciciot2023"]

# ---------------------------------------------
# SPLIT RATIOS
# ---------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ---------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------
CORRELATION_THRESHOLD  = 0.95   # drop one from each highly correlated pair
RF_IMPORTANCE_CUMSUM   = 0.95   # keep features covering 95 % cumulative importance
OUTLIER_STD_THRESHOLD  = 5.0    # drop rows beyond N std-devs from training mean

# ---------------------------------------------
# SUPERVISED CLASSIFIERS - hyperparameter search spaces
# ---------------------------------------------
RF_PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [None, 10, 20, 30],
    "min_samples_split":[2, 5, 10],
    "max_features":     ["sqrt", "log2"],
}

XGB_PARAM_GRID = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [3, 6, 9],
    "learning_rate":     [0.01, 0.05, 0.1, 0.2],
    "subsample":         [0.7, 0.8, 1.0],
    "colsample_bytree":  [0.7, 0.8, 1.0],
}

DT_PARAM_GRID = {
    "max_depth":         [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "criterion":         ["gini", "entropy"],
}

SVC_PARAM_GRID = {
    "C":        [0.01, 0.1, 1.0, 10.0],
    "max_iter": [1000, 2000],
}

KNN_PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 11],
    "metric":      ["euclidean", "manhattan"],
}

CV_FOLDS          = 5
RANDOMSEARCH_ITER = 20   # iterations for RandomizedSearchCV
# Hyperparameter search optimises MCC (consistent with the paper's metric);
# the best supervised model per dataset is chosen by VALIDATION-set MCC.
CV_SCORING        = "matthews_corrcoef"

# ---------------------------------------------
# STATISTICAL TESTS (revision)
# ---------------------------------------------
STATS_POOL_N   = 100_000          # rows of the (pre-SMOTE) training pool used
STATS_SEEDS    = [42, 7, 123, 2024, 31337]   # repeated 5x2cv seeds
BOOTSTRAP_N     = 1000            # bootstrap resamples for MCC/F1/FPR/recall CIs
BOOTSTRAP_N_AUC = 200             # (AUC is slower; fewer resamples)

# ---------------------------------------------
# VAE HYPERPARAMETERS
# ---------------------------------------------
VAE_LATENT_DIM    = 32
VAE_HIDDEN_DIMS   = [128, 64]   # encoder layers (decoder mirrors this)
VAE_DROPOUT_RATE  = 0.2
VAE_L2_REG        = 0.001
VAE_LEARNING_RATE = 0.001
VAE_BATCH_SIZE    = 256
VAE_MAX_EPOCHS    = 100
VAE_PATIENCE      = 10          # early stopping
VAE_BETA_RANGE    = [0.5, 1.0, 1.5, 2.0]   # β values to search on val set

# ---------------------------------------------
# HYBRID ENSEMBLE
# ---------------------------------------------
# Weight search space for weighted-voting ensemble (sup_weight + vae_weight = 1)
# Includes 1.0 (= supervised only) so the search can say "VAE adds nothing".
ENSEMBLE_WEIGHT_STEPS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ---------------------------------------------
# SHAP
# ---------------------------------------------
SHAP_TOP_N_FEATURES = 20
SHAP_BACKGROUND_SAMPLES = 100   # background sample size for TreeExplainer

# ---------------------------------------------
# STREAMLIT SIMULATION
# ---------------------------------------------
SIM_BATCH_SIZE   = 1       # records processed per step (1 = true streaming)
SIM_SLEEP_MS     = 50      # ms pause between records (controls visual speed)
SIM_ALERT_LOG    = os.path.join(OUT_LOGS, "alert_log.csv")
