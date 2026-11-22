"""
ablation.py  (camera-ready revision)
------------------------------------
Ablation on the held-out test set:

  A  Supervised only (best classifier by validation MCC), threshold 0.5
  A' Supervised only, decision threshold tuned on VALIDATION MCC
     (the baseline Reviewer 3, comment 8 asks for: does the hybrid's gain
     survive against simply moving the supervised threshold?)
  B  VAE only (supervised-calibrated threshold)
  B' VAE only, LABEL-FREE threshold (p99 of benign training scores)
  C  Hybrid OR-logic
  D  Hybrid weighted voting

Every row carries bootstrap 95% CIs. Paired comparisons on the SAME test
records (McNemar + paired-bootstrap CI of the MCC difference) are written to
<dataset>_paired_tests.csv for D vs A, D vs A', C vs A, A' vs A.
Ensemble size/training time = sum of the constituent models.
"""

import os
import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_model
from src.evaluation.uncertainty import (mcnemar, paired_bootstrap_diff,
                                        tune_threshold_mcc)
from utils.logger import get_logger
from configs.config import OUT_RESULTS

log = get_logger(__name__)


class _ThresholdedClassifier:
    """Best supervised model with a validation-tuned decision threshold."""
    def __init__(self, clf, threshold):
        self.clf, self.threshold = clf, threshold
        self.train_time_s, self.model_size_mb = clf.train_time_s, clf.model_size_mb
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
    def predict(self, X):
        return (self.clf.predict_proba(X) >= self.threshold).astype(int)


class _VAEAdapter:
    """Lets the VAE be called with the supervised X (row-aligned arrays)."""
    def __init__(self, vae, X_vae_full, labelfree=False):
        self._vae, self._X_vae, self._labelfree = vae, X_vae_full, labelfree
    def _rows(self, X_sup):
        return self._X_vae if len(X_sup) == len(self._X_vae) else None
    def predict(self, X_sup):
        Xv = self._X_vae
        return (self._vae.predict_labelfree(Xv) if self._labelfree
                else self._vae.predict(Xv))
    def predict_proba(self, X_sup):
        return self._vae.anomaly_score(self._X_vae)


def run_ablation(best_sup_clf, trained_vae, ensemble_or, ensemble_weighted,
                 X_test, y_test, dataset_name, X_test_vae,
                 X_val, y_val, preds_store=None) -> pd.DataFrame:
    os.makedirs(OUT_RESULTS, exist_ok=True)
    preds = {} if preds_store is None else preds_store

    thr, thr_val_mcc = tune_threshold_mcc(y_val, best_sup_clf.predict_proba(X_val))
    log.info(f"[{dataset_name}] Tuned supervised threshold = {thr} "
             f"(val MCC {thr_val_mcc:.4f})")
    tuned = _ThresholdedClassifier(best_sup_clf, thr)

    ens_size = ensemble_weighted.model_size_mb
    ens_time = ensemble_weighted.train_time_s
    configs = [
        ("A_Supervised_Only", best_sup_clf, best_sup_clf.train_time_s, best_sup_clf.model_size_mb),
        ("A2_Supervised_TunedThreshold", tuned, best_sup_clf.train_time_s, best_sup_clf.model_size_mb),
        ("B_VAE_Only", _VAEAdapter(trained_vae, X_test_vae), trained_vae.train_time_s, trained_vae.model_size_mb),
        ("B2_VAE_LabelFreeThreshold", _VAEAdapter(trained_vae, X_test_vae, True), trained_vae.train_time_s, trained_vae.model_size_mb),
        ("C_Hybrid_OR", ensemble_or, ens_time, ens_size),
        ("D_Hybrid_Weighted", ensemble_weighted, ens_time, ens_size),
    ]

    rows = []
    for name, model, t_time, size in configs:
        log.info(f"[{dataset_name}] Ablation config: {name}")
        r = evaluate_model(model=model, X_test=X_test, y_test=y_test,
                           model_name=name, dataset_name=dataset_name,
                           train_time_s=t_time, model_size_mb=size,
                           measure_latency=name not in ("B_VAE_Only",
                                                        "B2_VAE_LabelFreeThreshold"),
                           preds_store=preds)
        r["ablation_config"] = name
        if name == "A2_Supervised_TunedThreshold":
            r["decision_threshold"] = thr
        if name == "D_Hybrid_Weighted":
            r["sup_weight"] = getattr(ensemble_weighted._ens, "sup_weight", None) \
                if hasattr(ensemble_weighted, "_ens") else None
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_RESULTS, f"{dataset_name}_ablation.csv"), index=False)

    # Paired tests on identical test records
    pairs = [("D_Hybrid_Weighted", "A_Supervised_Only"),
             ("D_Hybrid_Weighted", "A2_Supervised_TunedThreshold"),
             ("C_Hybrid_OR", "A_Supervised_Only"),
             ("A2_Supervised_TunedThreshold", "A_Supervised_Only")]
    prow = []
    for a, b in pairs:
        pa, pb = preds[a][0], preds[b][0]
        prow.append({"dataset": dataset_name, "model_a": a, "model_b": b,
                     **paired_bootstrap_diff(y_test, pa, pb),
                     **mcnemar(y_test, pa, pb)})
    pdf = pd.DataFrame(prow)
    pdf.to_csv(os.path.join(OUT_RESULTS, f"{dataset_name}_paired_tests.csv"), index=False)
    log.info(f"[{dataset_name}] Paired tests:\n{pdf.to_string()}")
    return df
