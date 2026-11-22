"""
uncertainty.py  (camera-ready revision)
---------------------------------------
Confidence intervals and paired tests on the held-out TEST predictions
(Reviewer 3, comments 4 and 8):

  bootstrap_ci()           percentile bootstrap 95% CIs for MCC, F1, recall,
                           FPR (and ROC-AUC with fewer resamples)
  paired_bootstrap_diff()  95% CI of MCC(A) - MCC(B) on the SAME resamples
  mcnemar()                exact/continuity-corrected McNemar test on the
                           discordant test predictions of two detectors
  tune_threshold_mcc()     decision threshold maximising validation MCC
"""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from configs.config import BOOTSTRAP_N, BOOTSTRAP_N_AUC, RANDOM_SEED


def _counts(y, p):
    tp = np.sum((y == 1) & (p == 1)); tn = np.sum((y == 0) & (p == 0))
    fp = np.sum((y == 0) & (p == 1)); fn = np.sum((y == 1) & (p == 0))
    return float(tp), float(tn), float(fp), float(fn)


def _mcc(tp, tn, fp, fn):
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else (tp * tn - fp * fn) / den


def _point(y, p):
    tp, tn, fp, fn = _counts(y, p)
    f1 = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    fpr = 0.0 if (fp + tn) == 0 else fp / (fp + tn)
    return {"mcc": _mcc(tp, tn, fp, fn), "f1": f1, "recall": rec, "fpr": fpr}


def bootstrap_ci(y_true, y_pred, y_score=None, n=BOOTSTRAP_N,
                 n_auc=BOOTSTRAP_N_AUC, seed=RANDOM_SEED) -> dict:
    """Returns {metric_ci_low, metric_ci_high} for mcc, f1, recall, fpr, roc_auc."""
    y = np.asarray(y_true).astype(np.int8)
    p = np.asarray(y_pred).astype(np.int8)
    rng = np.random.default_rng(seed)
    N = len(y)
    vals = {k: [] for k in ("mcc", "f1", "recall", "fpr")}
    # 4-way cell code lets one bincount give all confusion counts per resample
    cell = (y * 2 + p).astype(np.int64)        # 0=tn 1=fp 2=fn 3=tp
    for _ in range(n):
        c = np.bincount(cell[rng.integers(0, N, N)], minlength=4)
        tn, fp, fn, tp = map(float, c)
        vals["mcc"].append(_mcc(tp, tn, fp, fn))
        vals["f1"].append(0.0 if (2*tp+fp+fn) == 0 else 2*tp/(2*tp+fp+fn))
        vals["recall"].append(0.0 if (tp+fn) == 0 else tp/(tp+fn))
        vals["fpr"].append(0.0 if (fp+tn) == 0 else fp/(fp+tn))
    out = {}
    for k, v in vals.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[f"{k}_ci_low"], out[f"{k}_ci_high"] = round(float(lo), 4), round(float(hi), 4)
    if y_score is not None:
        s = np.asarray(y_score, dtype=float)
        aucs = []
        for _ in range(n_auc):
            idx = rng.integers(0, N, N)
            yi = y[idx]
            if yi.min() == yi.max():
                continue
            aucs.append(roc_auc_score(yi, s[idx]))
        if aucs:
            lo, hi = np.percentile(aucs, [2.5, 97.5])
            out["roc_auc_ci_low"], out["roc_auc_ci_high"] = round(float(lo), 4), round(float(hi), 4)
    return out


def paired_bootstrap_diff(y_true, pred_a, pred_b, n=BOOTSTRAP_N,
                          seed=RANDOM_SEED) -> dict:
    """95% CI of MCC(A) - MCC(B), both evaluated on identical resamples."""
    y = np.asarray(y_true).astype(np.int8)
    a = np.asarray(pred_a).astype(np.int8)
    b = np.asarray(pred_b).astype(np.int8)
    rng = np.random.default_rng(seed)
    N = len(y)
    ca, cb = (y * 2 + a).astype(np.int64), (y * 2 + b).astype(np.int64)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        tn, fp, fn, tp = map(float, np.bincount(ca[idx], minlength=4))
        ma = _mcc(tp, tn, fp, fn)
        tn, fp, fn, tp = map(float, np.bincount(cb[idx], minlength=4))
        diffs.append(ma - _mcc(tp, tn, fp, fn))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = _point(y, a)["mcc"] - _point(y, b)["mcc"]
    return {"mcc_diff": round(float(point), 4),
            "mcc_diff_ci_low": round(float(lo), 4),
            "mcc_diff_ci_high": round(float(hi), 4)}


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """McNemar test on discordant pairs (b01: A right/B wrong, b10: reverse)."""
    y = np.asarray(y_true); a = np.asarray(pred_a); b = np.asarray(pred_b)
    ra, rb = (a == y), (b == y)
    n01 = int(np.sum(ra & ~rb)); n10 = int(np.sum(~ra & rb))
    n = n01 + n10
    if n == 0:
        return {"mcnemar_a_only_correct": 0, "mcnemar_b_only_correct": 0,
                "mcnemar_stat": 0.0, "mcnemar_p": 1.0}
    if n < 25:     # exact binomial version for small counts
        p = float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))
        stat = float(min(n01, n10))
    else:          # chi-square with continuity correction
        stat = (abs(n01 - n10) - 1) ** 2 / n
        p = float(stats.chi2.sf(stat, 1))
    return {"mcnemar_a_only_correct": n01, "mcnemar_b_only_correct": n10,
            "mcnemar_stat": round(float(stat), 4), "mcnemar_p": p}


def tune_threshold_mcc(y_val, score_val, grid=None) -> tuple:
    """Threshold (on a [0,1] score) maximising validation MCC. Coarse grid on
    purpose (a fine sweep over-fits the validation split)."""
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2) if grid is None else grid
    y = np.asarray(y_val); s = np.asarray(score_val)
    best_t, best_m = 0.5, -2.0
    for t in grid:
        p = (s >= t).astype(int)
        if p.min() == p.max():
            continue
        m = _point(y, p)["mcc"]
        if m > best_m:
            best_t, best_m = float(t), m
    return best_t, best_m
