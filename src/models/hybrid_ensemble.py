"""
hybrid_ensemble.py
------------------
Two hybrid ensemble configurations (ablation study configs 3 & 4):

  OR-Logic:        flag if EITHER supervised OR VAE fires
  Weighted Voting: weighted combination of confidence scores,
                   weight ratio tuned on validation set

NOTE: Supervised model and VAE operate in DIFFERENT feature spaces.
  - sup: RF-importance-selected features (e.g. 21)
  - vae: correlation-filtered features   (e.g. 54)

REVISION:
  * VAE scores are mapped to [0,1] by their empirical CDF over the BENIGN
    validation scores (fraction of benign validation records scoring lower).
    The old min-max normalisation used the validation min/max, so a single
    extreme score squashed every other record to ~0 and the VAE contributed
    nothing (this is why the old hybrids' ROC-AUC equalled XGBoost's).
  * Weighted voting: weight chosen by VALIDATION MCC (was F1); the grid
    includes sup_weight=1.0, i.e. "supervised only".
  * OR-logic continuous score (for ROC-AUC) = max(sup prob, VAE ECDF score).

Keras VAE models cannot be pickled directly via joblib.
Use HybridEnsemble.save() / HybridEnsemble.load() which saves
only the non-Keras state and reattaches the VAE on load.
"""

import os
import numpy as np
import joblib
from sklearn.metrics import f1_score, matthews_corrcoef
from utils.logger import get_logger
from configs.config import ENSEMBLE_WEIGHT_STEPS, OUT_MODELS

log = get_logger(__name__)


class HybridEnsemble:
    def __init__(self, supervised_clf, trained_vae,
                 mode: str = "weighted"):
        assert mode in ("or", "weighted"), f"Unknown mode: {mode}"
        self.sup = supervised_clf
        self.vae = trained_vae          # TrainedVAE wrapper (not raw Keras model)
        self.mode = mode
        self.sup_weight: float = 0.5
        self.vae_weight: float = 0.5
        self._vae_score_min: float = 0.0
        self._vae_score_max: float = 1.0
        self._benign_ref: np.ndarray = None    # sorted benign val VAE scores

    # ─────────────────────────────────────────────
    def fit_reference(self, X_val_vae: np.ndarray, y_val: np.ndarray):
        raw_vae = self.vae.anomaly_score(X_val_vae)
        self._vae_score_min = float(raw_vae.min())
        self._vae_score_max = float(raw_vae.max())
        self._benign_ref = np.sort(raw_vae[np.asarray(y_val) == 0])

    def tune_weights(self, X_val_sup: np.ndarray, y_val: np.ndarray,
                     X_val_vae: np.ndarray):
        self.fit_reference(X_val_vae, y_val)
        if self.mode != "weighted":
            return

        sup_scores = self.sup.predict_proba(X_val_sup)
        vae_scores = self._normalize_vae_scores(X_val_vae)
        best_mcc, best_sw = -2.0, 1.0
        self.tuning_log = []
        for sw in ENSEMBLE_WEIGHT_STEPS:
            vw = 1.0 - sw
            preds = ((sw * sup_scores + vw * vae_scores) >= 0.5).astype(int)
            m = matthews_corrcoef(y_val, preds)
            self.tuning_log.append((sw, round(float(m), 4)))
            if m > best_mcc + 1e-9:
                best_mcc, best_sw = m, sw

        self.sup_weight = best_sw
        self.vae_weight = round(1.0 - best_sw, 2)
        self.val_mcc = round(float(best_mcc), 4)
        log.info(f"[Ensemble-Weighted] Val MCC per sup_weight: {self.tuning_log}")
        log.info(f"[Ensemble-Weighted] Best weights: sup={best_sw:.2f}, "
                 f"vae={self.vae_weight:.2f} | Val MCC={best_mcc:.4f}")

    # ─────────────────────────────────────────────
    def predict(self, X_sup: np.ndarray,
                X_vae: np.ndarray = None) -> np.ndarray:
        X_vae = X_vae if X_vae is not None else X_sup
        if self.mode == "or":
            return self._or_predict(X_sup, X_vae)
        else:
            return self._weighted_predict(X_sup, self.sup_weight,
                                          self.vae_weight, X_vae)

    def predict_proba(self, X_sup: np.ndarray,
                      X_vae: np.ndarray = None) -> np.ndarray:
        X_vae = X_vae if X_vae is not None else X_sup
        sup_scores = self.sup.predict_proba(X_sup)
        vae_scores = self._normalize_vae_scores(X_vae)
        if self.mode == "or":
            return np.maximum(sup_scores, vae_scores)
        return self.sup_weight * sup_scores + self.vae_weight * vae_scores

    # ─────────────────────────────────────────────
    def _or_predict(self, X_sup, X_vae):
        sup_preds = self.sup.predict(X_sup)
        vae_preds = self.vae.predict(X_vae)
        return ((sup_preds == 1) | (vae_preds == 1)).astype(int)

    def _weighted_predict(self, X_sup, sw, vw, X_vae=None):
        X_vae = X_vae if X_vae is not None else X_sup
        sup_scores = self.sup.predict_proba(X_sup)
        vae_scores = self._normalize_vae_scores(X_vae)
        return ((sw * sup_scores + vw * vae_scores) >= 0.5).astype(int)

    def _normalize_vae_scores(self, X_vae):
        raw = self.vae.anomaly_score(X_vae)
        ref = getattr(self, "_benign_ref", None)
        if ref is not None and len(ref) > 0:
            # empirical CDF of benign validation scores
            return np.searchsorted(ref, raw, side="right") / len(ref)
        mn, mx = self._vae_score_min, self._vae_score_max
        if mx - mn < 1e-10:
            return np.zeros_like(raw)
        return np.clip((raw - mn) / (mx - mn), 0.0, 1.0)

    # ─────────────────────────────────────────────
    # SAVE / LOAD — excludes Keras VAE from pickle
    # ─────────────────────────────────────────────
    def save(self, path: str, vae_dir: str):
        """
        Save ensemble state (non-Keras parts only).
        The VAE is already saved separately in vae_dir — just record the path.
        """
        state = {
            "sup":              self.sup,
            "mode":             self.mode,
            "sup_weight":       self.sup_weight,
            "vae_weight":       self.vae_weight,
            "_vae_score_min":   self._vae_score_min,
            "_vae_score_max":   self._vae_score_max,
            "_benign_ref":      self._benign_ref,
            "val_mcc":          getattr(self, "val_mcc", None),
            "tuning_log":       getattr(self, "tuning_log", None),
            "vae_dir":          vae_dir,       # path to reload VAE from
            "vae_input_dim":    self.vae.input_dim,
        }
        joblib.dump(state, path)
        log.info(f"Ensemble saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "HybridEnsemble":
        """Load ensemble and re-attach the VAE from its saved directory."""
        from src.models.vae import TrainedVAE
        state = joblib.load(path)
        vae = TrainedVAE.load(state["vae_dir"], state["vae_input_dim"])
        obj = cls.__new__(cls)
        obj.sup             = state["sup"]
        obj.vae             = vae
        obj.mode            = state["mode"]
        obj.sup_weight      = state["sup_weight"]
        obj.vae_weight      = state["vae_weight"]
        obj._vae_score_min  = state["_vae_score_min"]
        obj._vae_score_max  = state["_vae_score_max"]
        obj._benign_ref     = state.get("_benign_ref")
        obj.val_mcc         = state.get("val_mcc")
        obj.tuning_log      = state.get("tuning_log")
        return obj

    @property
    def model_size_mb(self) -> float:
        """Ensemble footprint = its constituent models (Reviewer 3, comment 7)."""
        return round(float(getattr(self.sup, "model_size_mb", 0.0) or 0.0)
                     + float(getattr(self.vae, "model_size_mb", 0.0) or 0.0), 3)

    @property
    def train_time_s(self) -> float:
        return round(float(getattr(self.sup, "train_time_s", 0.0) or 0.0)
                     + float(getattr(self.vae, "train_time_s", 0.0) or 0.0), 2)


# ──────────────────────────────────────────────────────────────────────────────
# FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def build_ensembles(best_sup_clf, trained_vae,
                    X_val_sup: np.ndarray, y_val: np.ndarray,
                    X_val_vae: np.ndarray,
                    dataset_name: str) -> dict:
    or_ens = HybridEnsemble(best_sup_clf, trained_vae, mode="or")
    or_ens.fit_reference(X_val_vae, y_val)
    wv_ens = HybridEnsemble(best_sup_clf, trained_vae, mode="weighted")
    wv_ens.tune_weights(X_val_sup, y_val, X_val_vae)

    log.info(f"[{dataset_name}] Ensembles built: OR-logic and Weighted-Voting")
    return {"or": or_ens, "weighted": wv_ens}
