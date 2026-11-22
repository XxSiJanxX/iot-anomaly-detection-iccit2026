"""
cleaner.py
----------
Stage 2 of the preprocessing pipeline.
All cleaning is fitted on the TRAINING SET ONLY and applied to val/test.

Operations (in order):
  1. Drop configured identifier columns
  2. Drop duplicate rows
  3. Drop columns with zero / near-zero variance (fit on train)
  4. Encode categorical columns with label encoding (fit on train)
  5. Replace infinities with NaN
     - Drop columns where >60% of values are NaN
     - Fill remaining NaNs with column median (preserves attack rows)
  6. Drop rows beyond OUTLIER_STD_THRESHOLD standard deviations (fit on train)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils.logger import get_logger
from configs.config import DATASETS, OUTLIER_STD_THRESHOLD

log = get_logger(__name__)


class DataCleaner:
    """
    Stateful cleaner - call fit_transform(train_df) on training data,
    then transform(df) on val/test.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name   = dataset_name
        self.drop_cols      = DATASETS[dataset_name].get("drop_cols", [])
        self._zero_var_cols: list     = []
        self._high_nan_cols: list     = []   # cols dropped due to >60% NaN
        self._col_medians:   pd.Series = None  # for NaN imputation
        self._cat_encoders:  dict     = {}
        self._train_means:   pd.Series = None
        self._train_stds:    pd.Series = None
        self._numeric_cols:  list     = []

    # ---------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df and return cleaned df (call on TRAINING SET only)."""
        df = df.copy()
        label = df["label"].copy()
        df = df.drop(columns=["label"], errors="ignore")

        # 1. Drop identifier columns
        df = self._drop_id_cols(df)

        # 2. Drop duplicates
        before = len(df)
        df.drop_duplicates(inplace=True)
        label = label.loc[df.index]
        log.info(f"[{self.dataset_name}] Duplicates removed: {before - len(df):,}")

        # 3. Zero/near-zero variance - fit
        numeric = df.select_dtypes(include=[np.number])
        self._zero_var_cols = numeric.columns[numeric.std() < 1e-8].tolist()
        df.drop(columns=self._zero_var_cols, inplace=True)
        log.info(f"[{self.dataset_name}] Zero-variance cols dropped: {len(self._zero_var_cols)}")

        # 4. Categorical encoding - fit
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self._cat_encoders[col] = le
        if cat_cols:
            log.info(f"[{self.dataset_name}] Encoded categorical cols: {cat_cols}")

        # 5. Infinities -> NaN, then impute
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop columns where >60% values are NaN
        nan_frac = df.isna().mean()
        self._high_nan_cols = nan_frac[nan_frac > 0.6].index.tolist()
        if self._high_nan_cols:
            df.drop(columns=self._high_nan_cols, inplace=True)
            log.info(f"[{self.dataset_name}] High-NaN cols dropped (>60%): "
                     f"{len(self._high_nan_cols)}")

        # Fit column medians on remaining NaNs, then fill
        self._col_medians = df.median(numeric_only=True)
        nan_count = df.isna().sum().sum()
        df.fillna(self._col_medians, inplace=True)
        df.fillna(0, inplace=True)   # fallback for all-NaN columns
        log.info(f"[{self.dataset_name}] NaN values imputed: {nan_count:,} "
                 f"(median fill — 0 rows dropped)")

        # 6. Outlier removal - fit mean/std on NORMAL traffic only
        # For IDS datasets, attack traffic legitimately has extreme statistical
        # values (high byte counts, unusual packet rates etc). Removing these
        # as "outliers" would bias the model toward only mild attacks.
        # We fit the outlier bounds on NORMAL (label=0) samples only,
        # then apply to normal samples only — attack samples are never dropped.
        self._numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove label from numeric cols if present
        numeric_no_label = [c for c in self._numeric_cols if c != "label"]

        normal_mask = (label == 0).values
        if normal_mask.sum() > 100:
            self._train_means = df.loc[normal_mask, numeric_no_label].mean()
            self._train_stds  = df.loc[normal_mask, numeric_no_label].std().replace(0, 1e-8)
        else:
            self._train_means = df[numeric_no_label].mean()
            self._train_stds  = df[numeric_no_label].std().replace(0, 1e-8)

        # Only drop outlier rows from NORMAL traffic
        z_scores = (df.loc[normal_mask, numeric_no_label] - self._train_means).abs()                    / self._train_stds
        normal_outlier_mask = (z_scores > OUTLIER_STD_THRESHOLD).any(axis=1)
        normal_indices = df.index[normal_mask]
        outlier_indices = normal_indices[normal_outlier_mask]

        before = len(df)
        df = df.drop(index=outlier_indices)
        label = label.loc[df.index]
        log.info(f"[{self.dataset_name}] Outlier rows dropped (normal traffic only): "
                 f"{before - len(df):,}")

        df["label"] = label.values
        df.reset_index(drop=True, inplace=True)
        log.info(f"[{self.dataset_name}] Clean training set: {len(df):,} rows | "
                 f"Attack rate: {df['label'].mean():.2%}")
        return df

    # ---------------------------------------------
    def transform(self, df: pd.DataFrame,
                  drop_outliers: bool = False) -> pd.DataFrame:
        """Apply fitted cleaning to val or test set (no refitting).

        REVISION: rows are no longer dropped from val/test. The old version
        removed "outlier" rows from the evaluation sets based on their TRUE
        label (normal rows only), which a deployed detector cannot do and
        which removed the hardest benign examples from the FPR estimate.
        drop_outliers=True is only used for the VAE's benign TRAINING rows.
        """
        df = df.copy()
        label = df["label"].copy()
        df = df.drop(columns=["label"], errors="ignore")

        # 1. Drop id cols
        df = self._drop_id_cols(df)

        # 2. Drop zero-variance cols
        df.drop(columns=[c for c in self._zero_var_cols if c in df.columns],
                inplace=True)

        # 3. Apply fitted encoders (VECTORIZED -- see note below)
        #
        # PERFORMANCE NOTE: the previous implementation called
        # le.transform([x]) once PER ROW via .apply(), which on a 1.5M-row
        # val/test set with 5 categorical columns means up to ~7.5 million
        # individual sklearn encoder calls. This was the actual cause of
        # the pipeline appearing "stuck" for 15-30 minutes on every
        # dataset's VAE pipeline transform step (N-BaIoT, UNSW, IoT-23
        # all hit this same bottleneck, not separate issues each time).
        #
        # Fixed via a vectorized pandas map() using a dict built from
        # le.classes_ -> encoded index, which processes the entire column
        # in one operation instead of one Python-level call per row.
        for col, le in self._cat_encoders.items():
            if col in df.columns:
                class_to_code = {cls: idx for idx, cls in enumerate(le.classes_)}
                df[col] = (
                    df[col].astype(str)
                    .map(class_to_code)
                    .fillna(-1)
                    .astype(int)
                )

        # 4. Infinities -> NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop same high-NaN cols as training
        drop_present = [c for c in self._high_nan_cols if c in df.columns]
        if drop_present:
            df.drop(columns=drop_present, inplace=True)

        # Impute with training medians
        df.fillna(self._col_medians, inplace=True)
        df.fillna(0, inplace=True)

        # 5. Outlier filter -- training rows only (see docstring)
        numeric_present = [c for c in self._numeric_cols
                           if c in df.columns and c != "label"]
        means_p = self._train_means.reindex(numeric_present).fillna(0)
        stds_p  = self._train_stds.reindex(numeric_present).fillna(1)
        normal_t = (label == 0).values
        if drop_outliers and normal_t.sum() > 0:
            z_t = (df.loc[normal_t, numeric_present] - means_p).abs() / stds_p
            drop_idx = df.index[normal_t][(z_t > OUTLIER_STD_THRESHOLD).any(axis=1)]
            df = df.drop(index=drop_idx)
            label = label.loc[df.index]

        df["label"] = label.values
        df.reset_index(drop=True, inplace=True)
        return df

    # ---------------------------------------------
    def _drop_id_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
        return df
