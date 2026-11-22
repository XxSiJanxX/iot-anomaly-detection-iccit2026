"""
cross_dataset_eval.py
---------------------
Runs all 6 cross-dataset generalization experiments.

Handles two cases:
  1. Common features found → train RF on src common features, predict on tgt
  2. No common features    → record N/A result (complete domain shift finding)

The N-BaIoT vs UNSW/CICIoT pairs will likely have 0 common features because
N-BaIoT uses statistical packet features (MI_dir_L5_weight etc.) while
UNSW and CICIoT use flow-level features (dur, sbytes, rate etc.).
This IS a primary finding — it quantifies complete domain shift.

UNSW vs CICIoT may share some common flow features (rate, protocol, flags).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef

from src.evaluation.metrics import compute_metrics
from src.evaluation.uncertainty import bootstrap_ci
from utils.logger import get_logger
from configs.config import OUT_RESULTS, RANDOM_SEED

log = get_logger(__name__)


def run_all_cross_dataset(
    pipelines: dict,
    classifiers: dict,
    vae_models: dict,
    cross_data: dict,
) -> pd.DataFrame:
    """
    Run all 6 cross-dataset experiments.
    Returns DataFrame with one row per (src, tgt) pair.
    """
    os.makedirs(OUT_RESULTS, exist_ok=True)
    results = []
    controls = []        # within-source 2-feature control (Reviewer 3, c. 4)
    shap_rows = []       # SHAP of the restricted-feature transfer models (c. 9)
    mapping_rows = []    # feature-mapping table (Reviewer 1 c. 1, Reviewer 3 c. 3)

    for (src, tgt), cross in cross_data.items():

        # ── Case 1: No common features ─────────────────────────────
        if cross is None or cross.get("skipped", False):
            n_common  = cross["n_common"]  if cross else 0
            src_total = cross["src_total"] if cross else 0
            tgt_total = cross["tgt_total"] if cross else 0

            log.info(f"\n{'='*60}")
            log.info(f"Cross-dataset: {src} -> {tgt} | "
                     f"SKIPPED — 0 common features "
                     f"(src: {src_total}, tgt: {tgt_total})")
            log.info("This is a primary finding: complete domain shift "
                     "between statistical (N-BaIoT) and flow-level features.")

            # Record N/A row for paper table
            results.append({
                "src_dataset": src,
                "tgt_dataset": tgt,
                "model":       "N/A",
                "f1":          None,
                "mcc":         None,
                "roc_auc":     None,
                "accuracy":    None,
                "recall":      None,
                "fpr":         None,
                "threshold":   None,
                "n_common_features":  0,
                "src_total_features": src_total,
                "tgt_total_features": tgt_total,
                "note": "no_common_features_complete_domain_shift",
            })
            continue

        # ── Case 2: Common features available ──────────────────────
        X_src   = cross["X_src_common"]
        X_tgt   = cross["X_tgt_common"]
        y_src   = cross.get("y_src", cross.get("y_train", []))
        y_tgt   = cross["y_tgt"]
        n_common  = cross["n_common"]
        src_total = cross["src_total"]
        tgt_total = cross["tgt_total"]

        log.info(f"\n{'='*60}")
        log.info(f"Cross-dataset: {src} -> {tgt} | "
                 f"common features: {n_common} "
                 f"(src: {src_total}, tgt: {tgt_total})")

        # Subsample src if large
        if len(X_src) > 200_000:
            rng = np.random.default_rng(RANDOM_SEED)
            idx = rng.choice(len(X_src), 200_000, replace=False)
            X_src_fit = X_src[idx]
            y_src_fit = y_src[idx] if hasattr(y_src, '__len__') else y_src
        else:
            X_src_fit, y_src_fit = X_src, y_src

        # Train RF on src common features, evaluate on tgt.
        # oob_score=True lets us tune the decision threshold on the RF's
        # own out-of-bag predictions -- an unbiased estimate of how it
        # generalizes, with no leakage and no separate validation split
        # needed. This matters here specifically because cross-dataset
        # source sets can be severely imbalanced (e.g. ciciot2023's is
        # ~96% attack): a bare .predict() at the implicit 0.5 threshold
        # lets the model default to "always predict majority class" and
        # still look deceptively good on F1, exactly the failure mode
        # already diagnosed and fixed for the VAE's threshold (Bug 15) --
        # this is the same category of mistake, applied to a classifier.
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            n_jobs=-1, random_state=RANDOM_SEED, oob_score=True,
        )
        try:
            rf.fit(X_src_fit, y_src_fit)

            oob_probs = rf.oob_decision_function_[:, 1]
            # A handful of samples can end up with no OOB prediction at all
            # (included in every tree's bootstrap by chance) -- oob_probs
            # is NaN for those; exclude them from threshold selection.
            valid = ~np.isnan(oob_probs)
            threshold, using_oob = 0.5, False
            if valid.sum() > 0:
                y_src_arr = np.asarray(y_src_fit)[valid]
                oob_valid = oob_probs[valid]
                best_t, best_mcc = 0.5, -1.0
                # Coarse grid deliberately, not a fine sweep -- a coarser
                # grid can't land on an ultra-precise, noise-fitting
                # threshold the way an exhaustive search could.
                for t in np.linspace(0.05, 0.95, 19):
                    preds = (oob_valid > t).astype(int)
                    if preds.sum() == 0 or preds.sum() == len(preds):
                        continue
                    mcc = matthews_corrcoef(y_src_arr, preds)
                    if mcc > best_mcc:
                        best_mcc, best_t = mcc, t
                if best_mcc > -1.0:
                    threshold, using_oob = best_t, True

            tgt_probs = rf.predict_proba(X_tgt)[:, 1]
            y_pred = (tgt_probs > threshold).astype(int)
            metrics = compute_metrics(y_tgt, y_pred, y_score=tgt_probs)
            ci = bootstrap_ci(y_tgt, y_pred, tgt_probs)

            for m in cross.get("mapping", []):
                mapping_rows.append({"src_dataset": src, "tgt_dataset": tgt,
                                     "canonical": m["canonical"],
                                     "src_columns": "+".join(m["src_cols"]),
                                     "tgt_columns": "+".join(m["tgt_cols"])})

            # CONTROL: same 2-feature model on the SOURCE's own test set.
            # If it is weak here too, the ceiling is the feature set; if it
            # is strong here but weak on the target, it is distribution shift.
            if cross.get("X_src_test_common") is not None:
                Xs_te, ys_te = cross["X_src_test_common"], cross["y_src_test"]
                ps = rf.predict_proba(Xs_te)[:, 1]
                pr = (ps > threshold).astype(int)
                cm = compute_metrics(ys_te, pr, y_score=ps)
                controls.append({"src_dataset": src, "evaluated_on": f"{src} (own test)",
                                 "setting": "within_source_2feature_control",
                                 "n_test": int(len(ys_te)),
                                 **{k: cm[k] for k in ("f1", "mcc", "roc_auc",
                                                       "recall", "fpr")},
                                 **bootstrap_ci(ys_te, pr, ps)})
                log.info(f"  CONTROL [{src}->{src}, same 2 features] "
                         f"MCC={cm['mcc']:.4f} AUC={cm['roc_auc']}")

            # SHAP of this restricted-feature transfer model
            try:
                import shap
                rng_s = np.random.default_rng(RANDOM_SEED)
                expl = shap.TreeExplainer(rf)
                for where, Xw in (("source_test", cross.get("X_src_test_common")),
                                  ("target", X_tgt)):
                    if Xw is None:
                        continue
                    idx_s = rng_s.choice(len(Xw), min(2000, len(Xw)), replace=False)
                    sv = expl.shap_values(Xw[idx_s], check_additivity=False)
                    if isinstance(sv, list):
                        sv = sv[1]
                    elif getattr(sv, "ndim", 2) == 3:
                        sv = sv[:, :, 1]
                    for j, name in enumerate(cross.get("common_canonical", [])):
                        shap_rows.append({"src_dataset": src, "tgt_dataset": tgt,
                                          "evaluated_on": where, "feature": name,
                                          "mean_abs_shap": round(float(np.abs(sv[:, j]).mean()), 6)})
            except Exception as e:
                log.warning(f"  SHAP for transfer model failed: {e}")

            log.info(f"  [{src}->{tgt}] Threshold: {threshold:.3f} "
                     f"({'OOB-tuned' if using_oob else 'default 0.5, OOB tuning unavailable'})")

            results.append({
                "src_dataset": src,
                "tgt_dataset": tgt,
                "model":       "RandomForest",
                "f1":          metrics["f1"],
                "mcc":         metrics["mcc"],
                "roc_auc":     metrics["roc_auc"],
                "accuracy":    metrics["accuracy"],
                "recall":      metrics["recall"],
                "fpr":         metrics["fpr"],
                "n_test":      int(len(y_tgt)),
                **ci,
                "threshold":   round(float(threshold), 4),
                "n_common_features":  n_common,
                "src_total_features": src_total,
                "tgt_total_features": tgt_total,
                "note": f"common_features: {cross.get('common_canonical', [])}",
            })
            log.info(f"  RF [{src}->{tgt}] "
                     f"F1={metrics['f1']:.4f} MCC={metrics['mcc']:.4f} "
                     f"ROC-AUC={metrics['roc_auc']}")

        except Exception as e:
            log.error(f"  RF [{src}->{tgt}] FAILED: {e}")
            results.append({
                "src_dataset": src, "tgt_dataset": tgt,
                "model": "RandomForest",
                "f1": None, "mcc": None, "roc_auc": None,
                "accuracy": None, "recall": None, "fpr": None,
                "threshold": None,
                "n_common_features": n_common,
                "src_total_features": src_total,
                "tgt_total_features": tgt_total,
                "note": f"error: {e}",
            })

    df = pd.DataFrame(results)
    out_path = os.path.join(OUT_RESULTS, "cross_dataset_results.csv")
    df.to_csv(out_path, index=False)
    log.info(f"Cross-dataset results saved -> {out_path}")
    for name, rows in (("cross_dataset_controls.csv", controls),
                       ("cross_dataset_transfer_shap.csv", shap_rows),
                       ("cross_dataset_feature_mapping.csv", mapping_rows)):
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(OUT_RESULTS, name), index=False)
            log.info(f"Saved -> {name}")
    return df
