"""
statistical_tests.py  (camera-ready revision)
----------------------------------------------
Paired comparison of the top-2 supervised classifiers per dataset
(ranked by VALIDATION MCC).

WHAT A "PAIR" IS (Reviewer 1, comment 2): both classifiers are trained on
exactly the same training rows and scored on exactly the same held-out rows
of the same split; one pair = (score_A, score_B) on one split. Scores are
MCC (primary) and F1 (secondary).

FULLY IN-FOLD PROTOCOL (Reviewer 3, comment 2): for every split, the
feature pipeline is refitted on that split's training part only:
    correlation filter -> RF-importance selection -> MinMax scaling
    -> SMOTE -> classifier (with the hyperparameters found during tuning)
The held-out part is only transformed. Data: a stratified sample of
STATS_POOL_N rows of the cleaned, NOT oversampled training pool.

Tests reported:
  * naive paired t-test and Wilcoxon on 5-fold CV pairs (kept for
    comparison; the t-test is known to be anti-conservative, and with 5
    pairs Wilcoxon cannot go below p = 0.0625)
  * Dietterich's 5x2cv paired t-test, repeated over STATS_SEEDS
    (Reviewer 3, comment 6) -- per-seed results are saved, and the summary
    reports how many seeds are significant and whether the sign is stable.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE

from utils.logger import get_logger
from configs.config import (CV_FOLDS, RANDOM_SEED, OUT_RESULTS,
                            STATS_POOL_N, STATS_SEEDS)

log = get_logger(__name__)


class FoldFeatureEngineer(BaseEstimator, TransformerMixin):
    """Correlation filter + RF importance + MinMax, refitted per split."""

    def __init__(self, feature_names=None, dataset_name="fold"):
        self.feature_names = feature_names
        self.dataset_name = dataset_name

    def fit(self, X, y):
        from src.preprocessing.feature_engineering import FeatureEngineer
        self.fe_ = FeatureEngineer(self.dataset_name + "_fold")
        self.fe_.fit_transform(pd.DataFrame(X, columns=self.feature_names),
                               pd.Series(y))
        return self

    def transform(self, X):
        return self.fe_.transform(pd.DataFrame(X, columns=self.feature_names))


def _fit_score_pair(est_a, est_b, fe_names, ds, X_tr, y_tr, X_te, y_te, seed):
    """Fit FE (once) + SMOTE + both models on (X_tr, y_tr); score on X_te."""
    fe = FoldFeatureEngineer(fe_names, ds).fit(X_tr, y_tr)
    Xtr, Xte = fe.transform(X_tr), fe.transform(X_te)
    Xs, ys = SMOTE(random_state=seed).fit_resample(Xtr, y_tr)
    out = {}
    for tag, est in (("a", est_a), ("b", est_b)):
        m = clone(est).fit(Xs, ys)
        p = m.predict(Xte)
        out[f"mcc_{tag}"] = matthews_corrcoef(y_te, p)
        out[f"f1_{tag}"] = f1_score(y_te, p, zero_division=0)
    return out


def _five_fold_pairs(est_a, est_b, names, ds, X, y):
    rows = []
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for k, (tr, te) in enumerate(skf.split(X, y)):
        r = _fit_score_pair(est_a, est_b, names, ds, X[tr], y[tr], X[te], y[te],
                            RANDOM_SEED)
        r["fold"] = k
        rows.append(r)
    return pd.DataFrame(rows)


def _five_by_two(est_a, est_b, names, ds, X, y, seed):
    """Dietterich (1998) 5x2cv paired t-test for MCC and F1."""
    rng = np.random.default_rng(seed)
    diffs = {"mcc": [], "f1": []}
    for i in range(5):
        s_i = int(rng.integers(0, 2**31 - 1))
        idx_A, idx_B = train_test_split(np.arange(len(X)), test_size=0.5,
                                        stratify=y, random_state=s_i)
        r1 = _fit_score_pair(est_a, est_b, names, ds, X[idx_A], y[idx_A],
                             X[idx_B], y[idx_B], s_i)
        r2 = _fit_score_pair(est_a, est_b, names, ds, X[idx_B], y[idx_B],
                             X[idx_A], y[idx_A], s_i)
        for m in diffs:
            diffs[m].append((r1[f"{m}_a"] - r1[f"{m}_b"], r2[f"{m}_a"] - r2[f"{m}_b"]))
    out = {"seed": seed}
    for m, d in diffs.items():
        d = np.array(d)                     # shape (5, 2)
        p_bar = d.mean(axis=1)
        s2 = ((d - p_bar[:, None]) ** 2).sum(axis=1)
        denom = np.sqrt(s2.mean())
        t = 0.0 if denom == 0 else d[0, 0] / denom
        p = 2 * (1 - stats.t.cdf(abs(t), 5))
        out[f"{m}_t"] = round(float(t), 4)
        out[f"{m}_p"] = round(float(p), 6)
        out[f"{m}_mean_diff"] = round(float(d.mean()), 5)
    return out


def run_significance_test(dataset_name, classifiers, X_raw, y, raw_names):
    ranked = sorted(classifiers.values(),
                    key=lambda c: (c.val_mcc if c.val_mcc is not None else -2),
                    reverse=True)
    if len(ranked) < 2:
        log.warning(f"[{dataset_name}] Fewer than 2 classifiers -- skipping")
        return None, None
    a, b = ranked[:2]
    log.info(f"[{dataset_name}] Top-2 by validation MCC: {a.name} "
             f"({a.val_mcc}) vs {b.name} ({b.val_mcc})")

    X_raw, y = np.asarray(X_raw), np.asarray(y)
    if len(y) > STATS_POOL_N:
        idx, _ = train_test_split(np.arange(len(y)), train_size=STATS_POOL_N,
                                  stratify=y, random_state=RANDOM_SEED)
        X_raw, y = X_raw[idx], y[idx]
    log.info(f"[{dataset_name}] Stats pool: {len(y):,} rows (in-fold FE + SMOTE)")

    est_a, est_b = a.estimator, b.estimator

    folds = _five_fold_pairs(est_a, est_b, raw_names, dataset_name, X_raw, y)
    res = {"dataset": dataset_name, "model_a": a.name, "model_b": b.name,
           "model_a_val_mcc": a.val_mcc, "model_b_val_mcc": b.val_mcc,
           "pool_rows": int(len(y)), "pair_definition":
           "same training rows and same held-out rows per split"}
    for m in ("mcc", "f1"):
        sa, sb = folds[f"{m}_a"].values, folds[f"{m}_b"].values
        res[f"cv5_{m}_a_mean"] = round(float(sa.mean()), 4)
        res[f"cv5_{m}_b_mean"] = round(float(sb.mean()), 4)
        t, p = stats.ttest_rel(sa, sb)
        res[f"cv5_{m}_naive_t"] = round(float(t), 4) if np.isfinite(t) else None
        res[f"cv5_{m}_naive_p"] = round(float(p), 6) if np.isfinite(p) else None
        try:
            w, pw = stats.wilcoxon(sa, sb)
            res[f"cv5_{m}_wilcoxon_p"] = round(float(pw), 6)
        except ValueError:
            res[f"cv5_{m}_wilcoxon_p"] = None
    log.info(f"[{dataset_name}] 5-fold pairs:\n{folds.round(4).to_string()}")

    seed_rows = []
    for sd in STATS_SEEDS:
        r = _five_by_two(est_a, est_b, raw_names, dataset_name, X_raw, y, sd)
        r.update({"dataset": dataset_name, "model_a": a.name, "model_b": b.name})
        seed_rows.append(r)
        log.info(f"[{dataset_name}] 5x2cv seed={sd}: MCC t={r['mcc_t']} "
                 f"p={r['mcc_p']} | F1 t={r['f1_t']} p={r['f1_p']}")
    seeds = pd.DataFrame(seed_rows)
    for m in ("mcc", "f1"):
        p = seeds[f"{m}_p"].values
        t = seeds[f"{m}_t"].values
        res[f"x52_{m}_p_seed42"] = float(seeds.loc[seeds.seed == STATS_SEEDS[0], f"{m}_p"].iloc[0])
        res[f"x52_{m}_p_median"] = round(float(np.median(p)), 6)
        res[f"x52_{m}_p_min"] = round(float(p.min()), 6)
        res[f"x52_{m}_p_max"] = round(float(p.max()), 6)
        res[f"x52_{m}_n_significant"] = int((p < 0.05).sum())
        res[f"x52_{m}_n_seeds"] = int(len(p))
        res[f"x52_{m}_sign_consistent"] = bool((t > 0).all() or (t < 0).all())
    log.info(f"[{dataset_name}] 5x2cv (MCC) significant in "
             f"{res['x52_mcc_n_significant']}/{res['x52_mcc_n_seeds']} seeds; "
             f"median p={res['x52_mcc_p_median']}; sign consistent="
             f"{res['x52_mcc_sign_consistent']}")
    return res, seeds


def _merge_save(df, path, dataset_names):
    if os.path.exists(path):
        old = pd.read_csv(path)
        if "dataset" in old.columns:
            old = old[~old["dataset"].isin(dataset_names)]
            df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def run_all_significance_tests(dataset_names, pipelines, trainings) -> pd.DataFrame:
    os.makedirs(OUT_RESULTS, exist_ok=True)
    rows, seed_frames = [], []
    for ds in dataset_names:
        pre = pipelines[ds]
        res, seeds = run_significance_test(
            ds, trainings[ds]["classifiers"], pre["X_train_raw"],
            pre["y_train"], pre["raw_feature_names"])
        if res:
            rows.append(res)
            seed_frames.append(seeds)
    df = pd.DataFrame(rows)
    _merge_save(df, os.path.join(OUT_RESULTS, "statistical_significance_results.csv"),
                dataset_names)
    if seed_frames:
        _merge_save(pd.concat(seed_frames, ignore_index=True),
                    os.path.join(OUT_RESULTS, "statistical_significance_per_seed.csv"),
                    dataset_names)
    log.info("Statistical significance results saved")
    return df
