"""
supervised_classifiers.py
-------------------------
Trains, tunes, and persists all five supervised classifiers.

Models:
  - Random Forest      (RandomizedSearchCV)
  - XGBoost            (RandomizedSearchCV)
  - Decision Tree      (GridSearchCV)
  - LinearSVC          (GridSearchCV, subsample)
  - K-Nearest Neighbors(GridSearchCV, subsample)

Each classifier is wrapped in a TrainedClassifier dataclass that holds
the fitted model, best params, CV results, and timing info.

REVISION (Reviewer 3, comment 2): every estimator is now an imblearn
Pipeline([SMOTE, classifier]). imblearn applies the sampler only during
fit(), so inside RandomizedSearchCV/GridSearchCV SMOTE is re-fitted on the
training folds of each split and the held-out fold stays purely real data.
At predict() time the SMOTE step is skipped. The search optimises MCC.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from utils.logger import get_logger
from utils.timer import Timer
from configs.config import (
    RF_PARAM_GRID, XGB_PARAM_GRID, DT_PARAM_GRID,
    SVC_PARAM_GRID, KNN_PARAM_GRID,
    CV_FOLDS, RANDOMSEARCH_ITER, RANDOM_SEED, CV_SCORING,
    DATASETS, OUT_MODELS,
)

log = get_logger(__name__)


@dataclass
class TrainedClassifier:
    name:          str
    model:         Any
    best_params:   dict
    cv_score_mean:    float
    cv_score_std:     float
    train_time_s:  float
    model_size_mb: float = 0.0
    used_subsample: bool = False
    cv_metric:     str   = CV_SCORING
    val_mcc:       float = None     # filled in by main.stage_train

    @property
    def estimator(self):
        """The bare fitted classifier (last pipeline step) -- for SHAP."""
        m = self.model
        if hasattr(m, "steps"):
            m = m.steps[-1][1]
        return m

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            # Normalise to [0, 1] via sigmoid
            return 1 / (1 + np.exp(-scores))
        else:
            return self.predict(X).astype(float)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TrainedClassifier":
        return joblib.load(path)


# ------------------------------------------------------------------------------
# INTERNAL HELPERS
# ------------------------------------------------------------------------------

def _cv():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)


def smote_pipeline(clf):
    """SMOTE inside the estimator, so it is refitted per CV training fold."""
    return ImbPipeline([("smote", SMOTE(random_state=RANDOM_SEED)),
                        ("clf", clf)])


def _prefixed(grid: dict) -> dict:
    return {f"clf__{k}": v for k, v in grid.items()}


def _unprefixed(params: dict) -> dict:
    return {k.split("__", 1)[-1]: v for k, v in params.items()}


def _run_randomized(estimator, param_grid, X, y, name):
    search = RandomizedSearchCV(
        smote_pipeline(estimator),
        _prefixed(param_grid),
        n_iter=RANDOMSEARCH_ITER,
        scoring=CV_SCORING,
        cv=_cv(),
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    with Timer(f"{name} RandomizedSearchCV") as t:
        search.fit(X, y)
    return search, t.elapsed


def _run_grid(estimator, param_grid, X, y, name):
    search = GridSearchCV(
        smote_pipeline(estimator),
        _prefixed(param_grid),
        scoring=CV_SCORING,
        cv=_cv(),
        n_jobs=-1,
        verbose=0,
    )
    with Timer(f"{name} GridSearchCV") as t:
        search.fit(X, y)
    return search, t.elapsed


def _model_size_mb(model) -> float:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = f.name
    try:
        joblib.dump(model, tmp_path)
        size = os.path.getsize(tmp_path) / (1024 ** 2)
    finally:
        os.remove(tmp_path)
    return round(size, 3)


def _subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    return X[idx], y[idx]


# ------------------------------------------------------------------------------
# PUBLIC TRAINERS
# ------------------------------------------------------------------------------

def train_random_forest(X_train, y_train, dataset_name) -> TrainedClassifier:
    log.info(f"[{dataset_name}] Training Random Forest...")
    rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    search, elapsed = _run_randomized(rf, RF_PARAM_GRID, X_train, y_train, "RF")
    best = search.best_estimator_
    return TrainedClassifier(
        name="RandomForest",
        model=best,
        best_params=_unprefixed(search.best_params_),
        cv_score_mean=search.cv_results_["mean_test_score"][search.best_index_],
        cv_score_std=search.cv_results_["std_test_score"][search.best_index_],
        train_time_s=round(elapsed, 2),
        model_size_mb=_model_size_mb(best),
    )


def train_xgboost(X_train, y_train, dataset_name) -> TrainedClassifier:
    log.info(f"[{dataset_name}] Training XGBoost...")
    # SMOTE (inside the pipeline) balances each training fold, so no extra
    # class re-weighting is applied here.
    xgb = XGBClassifier(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    search, elapsed = _run_randomized(xgb, XGB_PARAM_GRID, X_train, y_train, "XGB")
    best = search.best_estimator_
    return TrainedClassifier(
        name="XGBoost",
        model=best,
        best_params=_unprefixed(search.best_params_),
        cv_score_mean=search.cv_results_["mean_test_score"][search.best_index_],
        cv_score_std=search.cv_results_["std_test_score"][search.best_index_],
        train_time_s=round(elapsed, 2),
        model_size_mb=_model_size_mb(best),
    )


def train_decision_tree(X_train, y_train, dataset_name) -> TrainedClassifier:
    log.info(f"[{dataset_name}] Training Decision Tree...")
    dt = DecisionTreeClassifier(random_state=RANDOM_SEED)
    search, elapsed = _run_grid(dt, DT_PARAM_GRID, X_train, y_train, "DT")
    best = search.best_estimator_
    return TrainedClassifier(
        name="DecisionTree",
        model=best,
        best_params=_unprefixed(search.best_params_),
        cv_score_mean=search.cv_results_["mean_test_score"][search.best_index_],
        cv_score_std=search.cv_results_["std_test_score"][search.best_index_],
        train_time_s=round(elapsed, 2),
        model_size_mb=_model_size_mb(best),
    )


def train_linear_svc(X_train, y_train, dataset_name,
                     subsample_n: int = None) -> TrainedClassifier:
    log.info(f"[{dataset_name}] Training LinearSVC...")
    used_sub = False
    X, y = X_train, y_train
    if subsample_n and len(X) > subsample_n:
        X, y = _subsample(X, y, subsample_n)
        used_sub = True
        log.info(f"[{dataset_name}] LinearSVC subsample: {subsample_n:,} records")

    # GridSearch over LinearSVC params with SMOTE inside each fold
    base = LinearSVC(random_state=RANDOM_SEED, dual=False)
    search = GridSearchCV(smote_pipeline(base), _prefixed(SVC_PARAM_GRID),
                          scoring=CV_SCORING, cv=_cv(), n_jobs=-1, verbose=0)
    with Timer("LinearSVC GridSearchCV") as t:
        search.fit(X, y)

    # Re-fit with best params, calibrated (for predict_proba), SMOTE first
    best_params = _unprefixed(search.best_params_)
    final_base = LinearSVC(**best_params, random_state=RANDOM_SEED, dual=False)
    final_model = smote_pipeline(CalibratedClassifierCV(final_base, cv=3))
    final_model.fit(X, y)

    return TrainedClassifier(
        name="LinearSVC",
        model=final_model,
        best_params=best_params,
        cv_score_mean=search.cv_results_["mean_test_score"][search.best_index_],
        cv_score_std=search.cv_results_["std_test_score"][search.best_index_],
        train_time_s=round(t.elapsed, 2),
        model_size_mb=_model_size_mb(final_model),
        used_subsample=used_sub,
    )


def train_knn(X_train, y_train, dataset_name,
              subsample_n: int = None) -> TrainedClassifier:
    log.info(f"[{dataset_name}] Training K-Nearest Neighbors...")
    used_sub = False
    X, y = X_train, y_train
    if subsample_n and len(X) > subsample_n:
        X, y = _subsample(X, y, subsample_n)
        used_sub = True
        log.info(f"[{dataset_name}] KNN subsample: {subsample_n:,} records")

    knn = KNeighborsClassifier(n_jobs=-1)
    search, elapsed = _run_grid(knn, KNN_PARAM_GRID, X, y, "KNN")
    best = search.best_estimator_

    return TrainedClassifier(
        name="KNN",
        model=best,
        best_params=_unprefixed(search.best_params_),
        cv_score_mean=search.cv_results_["mean_test_score"][search.best_index_],
        cv_score_std=search.cv_results_["std_test_score"][search.best_index_],
        train_time_s=round(elapsed, 2),
        model_size_mb=_model_size_mb(best),
        used_subsample=used_sub,
    )


# ------------------------------------------------------------------------------
# TRAIN ALL
# ------------------------------------------------------------------------------

def train_all_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dataset_name: str,
    subsample_n: int = None,
) -> dict:
    """
    Train all five classifiers and return a dict of TrainedClassifier objects.
    If a model is already saved on disk, it is loaded instead of retrained.

    Parameters
    ----------
    X_train      : scaled training features (pre-SMOTE pool; SMOTE is
                   applied inside every CV fold and in the final refit)
    y_train      : training labels
    dataset_name : used for logging and saving
    subsample_n  : subsample size for LinearSVC and KNN
    """
    os.makedirs(OUT_MODELS, exist_ok=True)
    results = {}

    trainers = [
        ("RandomForest", lambda: train_random_forest(X_train, y_train, dataset_name)),
        ("XGBoost",      lambda: train_xgboost(X_train, y_train, dataset_name)),
        ("DecisionTree", lambda: train_decision_tree(X_train, y_train, dataset_name)),
        ("LinearSVC",    lambda: train_linear_svc(X_train, y_train, dataset_name, subsample_n)),
        ("KNN",          lambda: train_knn(X_train, y_train, dataset_name, subsample_n)),
    ]

    for name, trainer_fn in trainers:
        save_path = os.path.join(OUT_MODELS, f"{dataset_name}_{name}.pkl")
        # Resume: load from disk if already trained
        if os.path.exists(save_path):
            try:
                clf = TrainedClassifier.load(save_path)
                results[name] = clf
                log.info(f"[{dataset_name}] {name} already trained — loaded from disk | "
                         f"CV {clf.cv_metric}: {clf.cv_score_mean:.4f}")
                continue
            except Exception as e:
                log.warning(f"[{dataset_name}] {name} load failed ({e}), retraining...")
        try:
            clf = trainer_fn()
            clf.save(save_path)
            results[name] = clf
            log.info(f"[{dataset_name}] {name} done | "
                     f"CV {clf.cv_metric}: {clf.cv_score_mean:.4f} +/- {clf.cv_score_std:.4f} | "
                     f"Time: {clf.train_time_s}s | "
                     f"Size: {clf.model_size_mb}MB")
        except Exception as e:
            log.error(f"[{dataset_name}] {name} FAILED: {e}")

    return results
