"""
pipeline_vae.py
---------------
Separate preprocessing pipeline for the VAE.

  - VAE is trained on BENIGN (normal) traffic ONLY, no SMOTE
  - Uses the SAME prepared (de-duplicated) DataFrame and the SAME split as
    the supervised pipeline. REVISION: the old version re-split the raw data
    and never de-duplicated the VAE's training rows (it reused the fitted
    cleaner via transform(), which has no de-dup step), so the two branches
    were trained on different data.
  - Correlation filter + MinMax scaler fitted on benign training rows only.
    MinMaxScaler(clip=True): out-of-range values seen at inference are
    clamped to [0,1] (required by the BCE reconstruction loss).
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.pipeline_supervised import split_dataset
from utils.logger import get_logger

log = get_logger(__name__)


def run_vae_pipeline(df, dataset_name: str, cleaner: DataCleaner) -> dict:
    """
    df      : the same prepared DataFrame given to run_supervised_pipeline
    cleaner : the DataCleaner already fitted by the supervised pipeline
    """
    log.info(f"[{dataset_name}] === VAE Pipeline Start ===")
    train_df, val_df, test_df = split_dataset(df)

    # Training rows: same cleaning, benign outliers removed (training only)
    train_df = cleaner.transform(train_df, drop_outliers=True)
    val_df   = cleaner.transform(val_df)
    test_df  = cleaner.transform(test_df)

    feat_cols = [c for c in train_df.columns if c not in ["label", "attack_cat"]]
    train_benign = train_df[train_df["label"] == 0][feat_cols]
    log.info(f"[{dataset_name}] Benign training samples: {len(train_benign):,} "
             f"({len(train_benign)/len(train_df):.1%} of the training split)")

    engineer = FeatureEngineer(dataset_name + "_vae")
    engineer._drop_correlated(train_benign, fit=True)
    remaining = [c for c in train_benign.columns if c not in engineer._dropped_corr]
    engineer.selected_features = remaining
    log.info(f"[{dataset_name}] VAE features (correlation filter only): "
             f"{len(remaining)}")

    engineer._scaler = MinMaxScaler(clip=True)
    X_train_benign = engineer._scaler.fit_transform(train_benign[remaining])
    X_val  = engineer.transform(val_df[feat_cols])
    X_test = engineer.transform(test_df[feat_cols])

    log.info(f"[{dataset_name}] === VAE Pipeline Complete ===")
    return {
        "X_train_benign": X_train_benign,
        "X_val":          X_val,
        "y_val":          val_df["label"].values,
        "X_test":         X_test,
        "y_test":         test_df["label"].values,
        "feature_names":  remaining,
        "engineer":       engineer,
        "n_features":     len(remaining),
        "n_benign_train": int(len(train_benign)),
    }
