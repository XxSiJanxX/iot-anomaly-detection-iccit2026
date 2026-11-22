"""
metrics.py
----------
Compute all evaluation metrics defined in the proposal (Section 7.1).

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, MCC, FPR
         + Training time, Inference latency (ms/sample), Model size (MB)

Returns a flat dict ready for saving to CSV or a results table.
"""

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
)
from utils.logger import get_logger

log = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None,   # probability / anomaly score for ROC-AUC
) -> dict:
    """
    Compute all classification metrics.

    Parameters
    ----------
    y_true  : true binary labels
    y_pred  : predicted binary labels
    y_score : continuous score for ROC-AUC (optional)

    Returns
    -------
    dict with all metric values
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    mcc       = matthews_corrcoef(y_true, y_pred)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    roc_auc = None
    if y_score is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_score)
        except Exception:
            roc_auc = None

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "roc_auc":   round(roc_auc, 4) if roc_auc is not None else None,
        "mcc":       round(mcc,       4),
        "fpr":       round(fpr,       4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


def measure_inference_latency(model, X: np.ndarray,
                               n_samples: int = 1000) -> float:
    """
    Measure average inference latency per sample in milliseconds.
    Runs n_samples individual predictions and returns mean ms/sample.

    Two different adapter classes wrap models with a separate VAE-space
    array, and both ignore the row-count of whatever they're handed:

    - main.py's _EnsembleAdapter (attrs: _ens, _X_vae) calls
      self._ens.predict(X_sup, self._X_vae) -- always the FULL X_vae
      array. Slicing X alone breaks row-count alignment and throws
      "operands could not be broadcast together with shapes (10,) (N,)".
    - ablation.py's _VAEAdapter (attrs: _vae, _X_vae) is worse: its
      predict() ignores the X_sup argument's shape entirely and always
      calls self._vae.predict(self._X_vae) -- the FULL VAE test array,
      every single call. No exception is thrown, so this one doesn't
      show up as a warning; it just silently reruns a full-dataset VAE
      forward pass on every one of the ~1000 timing iterations. Observed
      in practice: ablation's "B_VAE_Only" step took ~20 minutes for the
      exact same metric that took 11 seconds in the main evaluate loop
      (which calls TrainedVAE.predict() directly, with no adapter).

    Fix: if the model exposes a paired VAE array, slice X_vae with the
    exact same row positions as X_sup on every call below, and call
    through to whichever underlying model attribute is present, instead
    of relying on either adapter's default of the full array. X_sup and
    X_vae are row-aligned from the same test split, so slicing both with
    the same indices keeps predictions correct.
    """
    X_vae_paired = getattr(model, "_X_vae", None)
    ens = getattr(model, "_ens", None)
    vae = getattr(model, "_vae", None)

    def _predict(idx: slice):
        if X_vae_paired is not None and ens is not None:
            return ens.predict(X[idx], X_vae_paired[idx])
        if X_vae_paired is not None and vae is not None:
            return vae.predict(X_vae_paired[idx])
        return model.predict(X[idx])

    # Warm-up pass
    _ = _predict(slice(0, min(10, len(X))))

    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    times = []
    for i in indices:
        start = time.perf_counter()
        try:
            _ = _predict(slice(i, i + 1))
        except Exception:
            # Fall back to batch prediction and divide by batch size
            start = time.perf_counter()
            lo, hi = max(0, i - 50), i + 50
            _ = _predict(slice(lo, hi))
            elapsed = (time.perf_counter() - start) * 1000 / (hi - lo)
            times.append(elapsed)
            continue
        times.append((time.perf_counter() - start) * 1000)

    return round(float(np.mean(times)), 4)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    dataset_name: str,
    train_time_s: float = 0.0,
    model_size_mb: float = 0.0,
    measure_latency: bool = True,
    compute_ci: bool = True,
    preds_store: dict = None,
) -> dict:
    """
    Full evaluation of one model on one dataset's test set.
    Returns a flat results dict with all metrics + efficiency info.

    REVISION: adds bootstrap 95% CIs (compute_ci) and, if preds_store is a
    dict, stores {model_name: (y_pred, y_score)} for paired tests.
    Latency = mean wall-clock ms for ONE record (batch size 1), averaged
    over 1000 randomly chosen test records after a warm-up call.
    """
    y_pred = model.predict(X_test)

    # Get probability/score for ROC-AUC
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "anomaly_score"):
        y_score = model.anomaly_score(X_test)

    metrics = compute_metrics(y_test, y_pred, y_score)
    if compute_ci:
        from src.evaluation.uncertainty import bootstrap_ci
        metrics.update(bootstrap_ci(y_test, y_pred, y_score))
    if preds_store is not None:
        preds_store[model_name] = (np.asarray(y_pred), y_score)

    # Inference latency
    latency_ms = None
    if measure_latency:
        try:
            latency_ms = measure_inference_latency(model, X_test)
        except Exception as e:
            log.warning(f"Latency measurement failed for {model_name}: {e}")

    result = {
        "model":        model_name,
        "dataset":      dataset_name,
        "train_time_s": train_time_s,
        "latency_ms":   latency_ms,
        "model_size_mb":model_size_mb,
        "n_test":       int(len(y_test)),
        **metrics,
    }

    log.info(
        f"[{dataset_name}] {model_name:20s} | "
        f"F1={metrics['f1']:.4f} | "
        f"MCC={metrics['mcc']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"FPR={metrics['fpr']:.4f}"
    )
    return result
