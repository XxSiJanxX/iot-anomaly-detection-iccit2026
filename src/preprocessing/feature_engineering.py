"""
feature_engineering.py
----------------------
Stage 3: Feature selection and scaling.
Everything is FIT on the training set only, then APPLIED to val/test.

Steps:
  1. Remove highly correlated features (Pearson > threshold)
  2. Select top features by Random Forest importance (cumulative 95%)
  3. Min-Max scale to [0, 1]

Returns transformed X arrays and the list of selected feature names.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from utils.logger import get_logger
from configs.config import (
    CORRELATION_THRESHOLD,
    RF_IMPORTANCE_CUMSUM,
    OUT_MODELS,
    RANDOM_SEED,
)

log = get_logger(__name__)


class FeatureEngineer:
    """
    Stateful feature selector + scaler.
    fit_transform(X_train, y_train) → fitted, transformed X_train
    transform(X)                    → apply same selection + scaler
    """

    def __init__(self, dataset_name: str):
        self.dataset_name      = dataset_name
        self.selected_features: list = []
        self._scaler           = MinMaxScaler()
        self._dropped_corr:    list = []

    # ─────────────────────────────────────────────
    def fit_transform(self,
                      X_train: pd.DataFrame,
                      y_train: pd.Series) -> np.ndarray:
        """Fit on training data and return scaled numpy array."""

        X = X_train.copy()

        # Step 1: Correlation filter
        X = self._drop_correlated(X, fit=True)

        # Step 2: RF importance filter
        X = self._select_by_importance(X, y_train, fit=True)

        # Step 3: Min-Max scaling
        X_scaled = self._scaler.fit_transform(X)

        log.info(f"[{self.dataset_name}] Features after selection: "
                 f"{len(self.selected_features)}")
        return X_scaled

    # ─────────────────────────────────────────────
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted feature selection + scaler (no refitting)."""
        X = X.copy()

        # Drop same correlated cols as training
        drop_present = [c for c in self._dropped_corr if c in X.columns]
        X.drop(columns=drop_present, inplace=True)

        # Keep only selected features (fill missing with 0 for robustness)
        for feat in self.selected_features:
            if feat not in X.columns:
                X[feat] = 0.0
        X = X[self.selected_features]

        return self._scaler.transform(X)

    # ─────────────────────────────────────────────
    def save(self, path: str):
        joblib.dump(self, path)
        log.info(f"FeatureEngineer saved → {path}")

    @staticmethod
    def load(path: str) -> "FeatureEngineer":
        return joblib.load(path)

    # ─────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────
    def _drop_correlated(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        if fit:
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            self._dropped_corr = [
                col for col in upper.columns
                if any(upper[col] > CORRELATION_THRESHOLD)
            ]
            log.info(f"[{self.dataset_name}] Correlated cols dropped: "
                     f"{len(self._dropped_corr)}")

        drop_present = [c for c in self._dropped_corr if c in X.columns]
        return X.drop(columns=drop_present)

    def _select_by_importance(self,
                               X: pd.DataFrame,
                               y: pd.Series,
                               fit: bool = False) -> pd.DataFrame:
        if fit:
            log.info(f"[{self.dataset_name}] Fitting preliminary RF for importance…")
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            )
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=X.columns)
            importances_sorted = importances.sort_values(ascending=False)
            cumsum = importances_sorted.cumsum()
            self.selected_features = cumsum[
                cumsum <= RF_IMPORTANCE_CUMSUM
            ].index.tolist()

            # Ensure we always keep at least the top feature
            if not self.selected_features:
                self.selected_features = [importances_sorted.index[0]]

            log.info(f"[{self.dataset_name}] RF importance: kept "
                     f"{len(self.selected_features)} / {len(X.columns)} features")

        for feat in self.selected_features:
            if feat not in X.columns:
                X[feat] = 0.0
        return X[self.selected_features]
