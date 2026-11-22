"""
pipeline_supervised.py
----------------------
End-to-end preprocessing pipeline for supervised classifiers.

Order (camera-ready revision):
  0. (done before this function, in prepare.py) row cap + de-duplication on
     the FULL dataset, so identical observations cannot straddle partitions
  1. Split -> train (70%) / val (15%) / test (15%), stratified
  2. Clean (fit on train, apply to all; no rows dropped from val/test)
  3. Feature engineering - correlation filter, RF importance, MinMax scale
     (fit on train, apply to all)
  4. Cap the training pool (pre-SMOTE) to `pool_n` rows
  5. NO SMOTE HERE. SMOTE is now applied INSIDE every cross-validation fold
     (imblearn Pipeline in supervised_classifiers.py), so synthetic samples
     built from a validation fold's neighbours never leak into its training
     folds (Reviewer 3, comment 2).

Returns:
    X_train, y_train  (scaled, NOT oversampled -- the pre-SMOTE pool)
    X_val,   y_val    (scaled)
    X_test,  y_test   (scaled)
    X_train_raw / raw_feature_names : cleaned but unselected/unscaled pool
        rows, used by the 5x2cv test to refit feature selection + scaling
        inside each fold
    X_train_pre / X_test_pre / pre_feature_names : raw (unscaled)
        post-correlation-filter arrays for the cross-dataset experiments
    split_stats : counts for the paper's dataset table
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.prepare import split_overlap
from utils.logger import get_logger
from configs.config import VAL_RATIO, TEST_RATIO, RANDOM_SEED

log = get_logger(__name__)


def split_dataset(df: pd.DataFrame):
    """Stratified 70/15/15 split -- shared by both pipelines (same seed)."""
    val_test_ratio = VAL_RATIO + TEST_RATIO
    val_of_remaining = VAL_RATIO / val_test_ratio
    train_df, temp_df = train_test_split(
        df, test_size=val_test_ratio, stratify=df["label"],
        random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_of_remaining), stratify=temp_df["label"],
        random_state=RANDOM_SEED)
    return train_df, val_df, test_df


def _class_counts(y) -> dict:
    y = np.asarray(y)
    return {"rows": int(len(y)), "benign": int((y == 0).sum()),
            "attack": int((y == 1).sum())}


def run_supervised_pipeline(
    df: pd.DataFrame,
    dataset_name: str,
    pool_n: int = None,
) -> dict:
    """
    df     : prepared DataFrame (capped + de-duplicated, see prepare.py)
    pool_n : cap on the pre-SMOTE training pool (None = no cap)
    """
    log.info(f"[{dataset_name}] === Supervised Pipeline Start ===")
    split_stats = {}

    # -- STEP 1: SPLIT -------------------------------------------------
    train_df, val_df, test_df = split_dataset(df)
    log.info(f"[{dataset_name}] Split -> train: {len(train_df):,} | "
             f"val: {len(val_df):,} | test: {len(test_df):,}")
    ov = split_overlap(train_df, val_df, test_df)
    split_stats["partition_overlap_after_dedup"] = ov
    log.info(f"[{dataset_name}] Feature-vector overlap between partitions "
             f"(expected 0): {ov}")

    # -- STEP 2: CLEAN (fit on train only) -----------------------------
    cleaner = DataCleaner(dataset_name)
    train_df = cleaner.fit_transform(train_df)
    val_df   = cleaner.transform(val_df)
    test_df  = cleaner.transform(test_df)

    feat_cols = [c for c in train_df.columns if c not in ["label", "attack_cat"]]
    X_train_raw = train_df[feat_cols]
    y_train     = train_df["label"].values
    X_val_raw, y_val   = val_df[feat_cols],  val_df["label"].values
    X_test_raw, y_test = test_df[feat_cols], test_df["label"].values

    split_stats["train_after_cleaning"] = _class_counts(y_train)
    split_stats["val"]  = _class_counts(y_val)
    split_stats["test"] = _class_counts(y_test)

    # -- STEP 3: FEATURE ENGINEERING (fit on train only) ---------------
    engineer = FeatureEngineer(dataset_name)
    X_train_scaled = engineer.fit_transform(X_train_raw, pd.Series(y_train))
    X_val_scaled   = engineer.transform(X_val_raw)
    X_test_scaled  = engineer.transform(X_test_raw)
    feature_names  = engineer.selected_features
    split_stats["n_raw_features"] = len(feat_cols)
    split_stats["n_after_corr_filter"] = len(feat_cols) - len(engineer._dropped_corr)
    split_stats["n_selected_features"] = len(feature_names)

    # Raw (unscaled) post-correlation-filter arrays for cross-dataset use.
    # Kept unscaled so derived features (e.g. a byte sum) are computed on
    # real values before any scaling.
    pre_feature_names = [c for c in feat_cols if c not in engineer._dropped_corr]
    X_train_pre = X_train_raw[pre_feature_names].values
    X_test_pre  = X_test_raw[pre_feature_names].values
    X_train_rawall = X_train_raw.values.astype(np.float32)

    # -- STEP 4: CAP THE PRE-SMOTE TRAINING POOL ------------------------
    if pool_n is not None and len(X_train_scaled) > pool_n:
        keep_ratio = pool_n / len(X_train_scaled)
        idx = np.arange(len(X_train_scaled))
        idx_keep, _ = train_test_split(idx, train_size=keep_ratio,
                                       stratify=y_train,
                                       random_state=RANDOM_SEED)
        idx_keep = np.sort(idx_keep)
        X_train_scaled = X_train_scaled[idx_keep]
        X_train_pre    = X_train_pre[idx_keep]
        X_train_rawall = X_train_rawall[idx_keep]
        y_train        = y_train[idx_keep]
        log.info(f"[{dataset_name}] Training pool capped to {len(idx_keep):,} "
                 f"rows (stratified) | attack rate {y_train.mean():.2%}")
    split_stats["train_pool"] = _class_counts(y_train)

    log.info(f"[{dataset_name}] SMOTE will be applied inside each CV fold "
             f"(not here) | pool: {len(y_train):,} rows")
    log.info(f"[{dataset_name}] === Supervised Pipeline Complete ===")

    return {
        "X_train":           X_train_scaled,
        "y_train":           y_train,
        "X_val":             X_val_scaled,
        "y_val":             y_val,
        "X_test":            X_test_scaled,
        "y_test":            y_test,
        "feature_names":     feature_names,
        "cleaner":           cleaner,
        "engineer":          engineer,
        "X_train_raw":       X_train_rawall,
        "raw_feature_names": feat_cols,
        "X_train_pre":       X_train_pre,
        "X_test_pre":        X_test_pre,
        "pre_feature_names": pre_feature_names,
        "y_train_pre":       y_train,
        "split_stats":       split_stats,
    }
