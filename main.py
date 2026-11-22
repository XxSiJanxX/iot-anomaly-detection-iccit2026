"""
main.py
-------
Master runner for the full experiment pipeline.

Usage:
    python main.py --dataset nbaiot --stage preprocess
    python main.py --dataset nbaiot --stage train
    python main.py --dataset nbaiot --stage evaluate
    python main.py --dataset nbaiot --stage cross
    python main.py --dataset nbaiot --stage shap
    python main.py --dataset nbaiot --stage all
    python main.py --dataset all --stage all
"""

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd
from itertools import permutations

from utils.seed import set_global_seed
from utils.logger import get_logger
from configs.config import (
    DATASET_NAMES, DATASETS, OUT_MODELS, OUT_RESULTS, RANDOM_SEED
)

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _ckpt_path(dataset: str, stage: str) -> str:
    os.makedirs(OUT_MODELS, exist_ok=True)
    return os.path.join(OUT_MODELS, f"{dataset}_{stage}_checkpoint.pkl")


def _save_checkpoint(dataset: str, stage: str, data: dict):
    path = _ckpt_path(dataset, stage)
    joblib.dump(data, path)
    log.info(f"Checkpoint saved: {path}")


def _load_checkpoint(dataset: str, stage: str) -> dict:
    path = _ckpt_path(dataset, stage)
    if not os.path.exists(path):
        log.error(
            f"Checkpoint not found: {path}\n"
            f"Run --stage preprocess first for dataset '{dataset}'."
        )
        sys.exit(1)
    log.info(f"Loading checkpoint: {path}")
    return joblib.load(path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="IoT Anomaly Detection Pipeline")
    parser.add_argument("--dataset", default="all",
                        choices=DATASET_NAMES + ["all"])
    parser.add_argument("--stage", default="all",
                        choices=["preprocess", "train", "evaluate",
                                 "cross", "stats", "shap", "all"])
    parser.add_argument("--subsample-svc-knn", type=int, default=200_000)
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — PREPROCESS
# ──────────────────────────────────────────────────────────────────────────────

def stage_preprocess(dataset_names: list):
    from src.preprocessing.data_loader import load_dataset
    from src.preprocessing.prepare import prepare_dataset, save_summary
    from src.preprocessing.pipeline_supervised import run_supervised_pipeline
    from src.preprocessing.pipeline_vae import run_vae_pipeline

    os.makedirs(OUT_MODELS, exist_ok=True)

    for ds in dataset_names:
        log.info(f"\n{'='*60}\nPreprocessing: {ds}\n{'='*60}")
        pool_n = DATASETS[ds].get("subsample")

        df = load_dataset(ds)
        # REVISION: cap + harmonise + DE-DUPLICATE ONCE, BEFORE the split,
        # and feed the SAME frame to both branches.
        df, summary = prepare_dataset(df, ds)

        sup_out = run_supervised_pipeline(df, ds, pool_n=pool_n)
        vae_out = run_vae_pipeline(df, ds, cleaner=sup_out["cleaner"])
        del df

        # Both branches must score the very same test rows (hybrids need it)
        if not np.array_equal(np.asarray(sup_out["y_test"]),
                              np.asarray(vae_out["y_test"])):
            raise RuntimeError(f"[{ds}] supervised/VAE test rows are not aligned")

        summary.update(sup_out["split_stats"])
        summary["vae_benign_train_rows"] = vae_out["n_benign_train"]
        summary["vae_n_features"] = vae_out["n_features"]
        save_summary(ds, summary)

        joblib.dump({
            "X_test":        sup_out["X_test"],
            "y_test":        sup_out["y_test"],
            "feature_names": sup_out["feature_names"],
        }, os.path.join(OUT_MODELS, f"{ds}_test_arrays.pkl"))
        joblib.dump(sup_out["cleaner"],
                    os.path.join(OUT_MODELS, f"{ds}_cleaner.pkl"))
        sup_out["engineer"].save(
                    os.path.join(OUT_MODELS, f"{ds}_engineer.pkl"))

        ckpt = {
            "X_train":       sup_out["X_train"],      # pre-SMOTE pool (scaled)
            "y_train":       sup_out["y_train"],
            "X_val":         sup_out["X_val"],
            "y_val":         sup_out["y_val"],
            "X_test":        sup_out["X_test"],
            "y_test":        sup_out["y_test"],
            "feature_names": sup_out["feature_names"],
            "X_train_raw":       sup_out["X_train_raw"],
            "raw_feature_names": sup_out["raw_feature_names"],
            "X_train_pre":       sup_out["X_train_pre"],
            "X_test_pre":        sup_out["X_test_pre"],
            "y_train_pre":       sup_out["y_train_pre"],
            "pre_feature_names": sup_out["pre_feature_names"],
            "vae_X_train_benign": vae_out["X_train_benign"],
            "vae_X_val":          vae_out["X_val"],
            "vae_y_val":          vae_out["y_val"],
            "vae_X_test":         vae_out["X_test"],
            "vae_y_test":         vae_out["y_test"],
            "vae_feature_names":  vae_out["feature_names"],
            "vae_n_features":     vae_out["n_features"],
        }
        _save_checkpoint(ds, "preprocess", ckpt)
        log.info(f"[{ds}] Preprocessing complete")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — TRAIN
# ──────────────────────────────────────────────────────────────────────────────

def stage_train(dataset_names: list, subsample_svc_knn: int):
    from src.models.supervised_classifiers import train_all_classifiers
    from src.models.vae import train_vae
    from src.models.hybrid_ensemble import build_ensembles

    for ds in dataset_names:
        log.info(f"\n{'='*60}\nTraining: {ds}\n{'='*60}")
        ckpt = _load_checkpoint(ds, "preprocess")

        # Train supervised classifiers
        clfs = train_all_classifiers(
            X_train=ckpt["X_train"],
            y_train=ckpt["y_train"],
            dataset_name=ds,
            subsample_n=subsample_svc_knn,
        )

        # Train VAE
        vae = train_vae(
            X_train_benign=ckpt["vae_X_train_benign"],
            X_val=ckpt["vae_X_val"],
            y_val=ckpt["vae_y_val"],
            dataset_name=ds,
        )

        # REVISION: rank classifiers by VALIDATION-set MCC (held-out, real
        # rows, no SMOTE) -- consistent with the paper's metric.
        from sklearn.metrics import matthews_corrcoef
        for name, clf in clfs.items():
            clf.val_mcc = round(float(matthews_corrcoef(
                ckpt["y_val"], clf.predict(ckpt["X_val"]))), 4)
            clf.save(os.path.join(OUT_MODELS, f"{ds}_{name}.pkl"))
            log.info(f"[{ds}] {name}: validation MCC = {clf.val_mcc}")
        best_clf = max(clfs.values(), key=lambda c: c.val_mcc)
        log.info(f"[{ds}] Best supervised (validation MCC): {best_clf.name} "
                 f"({best_clf.val_mcc:.4f})")

        joblib.dump(best_clf,
                    os.path.join(OUT_MODELS, f"{ds}_best_supervised.pkl"))

        # Build ensembles (supervised and VAE use different feature spaces)
        ensembles = build_ensembles(
            best_sup_clf=best_clf,
            trained_vae=vae,
            X_val_sup=ckpt["X_val"],
            y_val=ckpt["y_val"],
            X_val_vae=ckpt["vae_X_val"],
            dataset_name=ds,
        )

        vae_dir = os.path.join(OUT_MODELS, f"{ds}_VAE")
        ensembles["or"].save(
            os.path.join(OUT_MODELS, f"{ds}_hybrid_or.pkl"), vae_dir)
        ensembles["weighted"].save(
            os.path.join(OUT_MODELS, f"{ds}_hybrid_weighted.pkl"), vae_dir)

        # Save train checkpoint
        # VAE excluded from checkpoint — already saved to outputs/models/{ds}_VAE/
        # Ensembles excluded — already saved via ensemble.save()
        train_ckpt = {
            "classifiers":   clfs,
            "best_clf_name": best_clf.name,
            "vae_dir":       os.path.join(OUT_MODELS, f"{ds}_VAE"),
            "vae_input_dim": vae.input_dim,
        }
        _save_checkpoint(ds, "train", train_ckpt)

        # VAE settings / thresholds for the paper (Reviewer 3, comment 5)
        vmeta = {"dataset": ds, **vae.meta(),
                 "reconstruction_error": "per-record mean binary cross-entropy "
                                         "+ beta * KL (z_mean decoding)",
                 "architecture": "128-64-[32 latent]-64-128, ReLU, dropout 0.2, "
                                 "L2 1e-3, Adam 1e-3, batch 256, <=100 epochs, "
                                 "early stopping (patience 10) on benign val loss",
                 "ensemble_sup_weight": ensembles["weighted"].sup_weight,
                 "ensemble_val_mcc": getattr(ensembles["weighted"], "val_mcc", None)}
        vpath = os.path.join(OUT_RESULTS, "vae_summary.csv")
        os.makedirs(OUT_RESULTS, exist_ok=True)
        vdf = pd.DataFrame([vmeta])
        if os.path.exists(vpath):
            old = pd.read_csv(vpath)
            vdf = pd.concat([old[old["dataset"] != ds], vdf], ignore_index=True)
        vdf.to_csv(vpath, index=False)
        log.info(f"[{ds}] Training complete")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3 — EVALUATE
# ──────────────────────────────────────────────────────────────────────────────

def stage_evaluate(dataset_names: list):
    from src.evaluation.metrics import evaluate_model
    from src.evaluation.ablation import run_ablation
    from src.evaluation.visualizations import (
        plot_metric_comparison, plot_roc_curves, compute_roc,
        plot_confusion_matrix, plot_vae_reconstruction_dist,
        plot_vae_loss_curves, plot_efficiency, plot_ablation,
    )

    os.makedirs(OUT_RESULTS, exist_ok=True)
    all_results = []

    for ds in dataset_names:
        log.info(f"\n{'='*60}\nEvaluation: {ds}\n{'='*60}")
        pre  = _load_checkpoint(ds, "preprocess")
        trn  = _load_checkpoint(ds, "train")

        X_test   = pre["X_test"]
        y_test   = pre["y_test"]
        clfs        = trn["classifiers"]
        preds_store = {}
        # Reload VAE from its saved directory
        from src.models.vae import TrainedVAE
        from src.models.hybrid_ensemble import HybridEnsemble
        vae = TrainedVAE.load(trn["vae_dir"], trn["vae_input_dim"])
        ensembles = {
            "or":       HybridEnsemble.load(os.path.join(OUT_MODELS, f"{ds}_hybrid_or.pkl")),
            "weighted": HybridEnsemble.load(os.path.join(OUT_MODELS, f"{ds}_hybrid_weighted.pkl")),
        }

        # Evaluate classifiers
        for name, clf in clfs.items():
            res = evaluate_model(
                model=clf, X_test=X_test, y_test=y_test,
                model_name=name, dataset_name=ds,
                train_time_s=clf.train_time_s,
                model_size_mb=clf.model_size_mb,
                preds_store=preds_store,
            )
            res["val_mcc"] = clf.val_mcc
            all_results.append(res)

        # Evaluate VAE
        vae_X_test = pre["vae_X_test"]
        vae_y_test = pre["vae_y_test"]
        vae_res = evaluate_model(
            model=vae, X_test=vae_X_test, y_test=vae_y_test,
            model_name="VAE", dataset_name=ds,
            train_time_s=vae.train_time_s,
            model_size_mb=vae.model_size_mb,
        )
        all_results.append(vae_res)

        # Evaluate ensembles
        # Ensembles need both supervised (21) and VAE (54) feature spaces
        # Wrap each ensemble with a thin adapter so evaluate_model works generically
        class _EnsembleAdapter:
            """Adapter that injects X_vae so evaluate_model can call .predict(X)."""
            def __init__(self, ens, X_vae_full):
                self._ens = ens
                self._X_vae = X_vae_full
                # REVISION: footprint = constituent models (was 0.0)
                self.train_time_s  = ens.train_time_s
                self.model_size_mb = ens.model_size_mb
            def predict(self, X_sup):
                return self._ens.predict(X_sup, self._X_vae)
            def predict_proba(self, X_sup):
                return self._ens.predict_proba(X_sup, self._X_vae)

        for ens_name, ens in ensembles.items():
            adapter = _EnsembleAdapter(ens, vae_X_test)
            ens_res = evaluate_model(
                model=adapter, X_test=X_test, y_test=y_test,
                model_name=f"Hybrid_{ens_name.upper()}", dataset_name=ds,
                train_time_s=adapter.train_time_s,
                model_size_mb=adapter.model_size_mb,
            )
            all_results.append(ens_res)

        # Ablation (+ tuned-threshold baseline, label-free VAE, paired tests)
        best_clf = clfs[trn["best_clf_name"]]
        abl_df = run_ablation(
            best_sup_clf=best_clf, trained_vae=vae,
            ensemble_or=_EnsembleAdapter(ensembles["or"], vae_X_test),
            ensemble_weighted=_EnsembleAdapter(ensembles["weighted"], vae_X_test),
            X_test=X_test, y_test=y_test, dataset_name=ds,
            X_test_vae=vae_X_test, X_val=pre["X_val"], y_val=pre["y_val"],
        )
        for _, r in abl_df[abl_df["ablation_config"].isin(
                ["A2_Supervised_TunedThreshold", "B2_VAE_LabelFreeThreshold"])].iterrows():
            row = r.to_dict()
            row["model"] = ("BestSupervised_TunedThreshold"
                            if row["ablation_config"].startswith("A2")
                            else "VAE_LabelFreeThreshold")
            all_results.append(row)

        # Plots
        results_df = pd.DataFrame(all_results)
        plot_metric_comparison(results_df, ds)

        roc_data = {}
        for name, clf in clfs.items():
            fpr, tpr, auc_val = compute_roc(clf, X_test, y_test)
            roc_data[name] = (fpr, tpr, auc_val)
        plot_roc_curves(roc_data, ds)

        y_pred_best = best_clf.predict(X_test)
        plot_confusion_matrix(y_test, y_pred_best, best_clf.name, ds)

        if vae_X_test.shape[1] == vae.input_dim:
            plot_vae_reconstruction_dist(
                vae, vae_X_test, vae_y_test, ds, threshold=vae.threshold)

        plot_vae_loss_curves(vae.history, ds, vae.best_beta)
        plot_efficiency(results_df[results_df["dataset"] == ds], ds)
        plot_ablation(abl_df, ds)

        log.info(f"[{ds}] Evaluation complete")

    full_df = pd.DataFrame(all_results)
    all_results_path = os.path.join(OUT_RESULTS, "all_results.csv")
    if os.path.exists(all_results_path):
        # Merge with existing results instead of overwriting: keep every
        # previously-saved dataset's rows except the ones we just
        # regenerated (those are replaced by the fresh run in full_df).
        existing_df = pd.read_csv(all_results_path)
        existing_df = existing_df[~existing_df["dataset"].isin(dataset_names)]
        full_df = pd.concat([existing_df, full_df], ignore_index=True)
    full_df.to_csv(all_results_path, index=False)
    log.info(f"Full results saved -> {OUT_RESULTS}/all_results.csv "
             f"({full_df['dataset'].nunique()} dataset(s) now in file)")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4 — CROSS-DATASET
# ──────────────────────────────────────────────────────────────────────────────

def stage_cross_dataset(dataset_names: list):
    from src.preprocessing.cross_dataset import CrossDatasetPreparer
    from src.evaluation.cross_dataset_eval import run_all_cross_dataset
    from src.evaluation.visualizations import plot_cross_dataset_heatmap, plot_cross_dataset_bars

    if len(dataset_names) < 2:
        log.warning("Need at least 2 datasets for cross-dataset experiments.")
        return

    # Load all checkpoints
    pipelines   = {}
    classifiers = {}
    vaes        = {}
    for ds in dataset_names:
        pre = _load_checkpoint(ds, "preprocess")
        trn = _load_checkpoint(ds, "train")
        # Load engineer from its saved .pkl file
        from src.preprocessing.feature_engineering import FeatureEngineer
        engineer_path = os.path.join(OUT_MODELS, f"{ds}_engineer.pkl")
        engineer = FeatureEngineer.load(engineer_path)
        pre["engineer"] = engineer
        # Use post-selection feature names (pre_feature_names not saved separately)
        # The cross_dataset preparer will use these + canonical map
        # Use pre-selection arrays if available, else fall back to post-selection
        if "pre_feature_names" not in pre:
            pre["pre_feature_names"] = pre["feature_names"]
            pre["X_train_pre"]       = pre["X_train"]
            pre["y_train_pre"]       = pre["y_train"]
        if "X_test_pre" not in pre:
            pre["X_test_pre"] = pre["X_test"]
        pipelines[ds]   = pre
        classifiers[ds] = trn["classifiers"]
        from src.models.vae import TrainedVAE as _TrainedVAE
        vaes[ds] = _TrainedVAE.load(trn["vae_dir"], trn["vae_input_dim"])

    cross_data = {}
    for src, tgt in permutations(dataset_names, 2):
        preparer = CrossDatasetPreparer(src, tgt)

        # Use the target's pre-selection test array directly from its
        # already-loaded preprocess checkpoint (pipelines[tgt], loaded
        # above). This previously looked for a separate
        # "{tgt}_pre_arrays.pkl" file that no code path in this project
        # ever wrote -- os.path.exists() was always False, so every
        # target dataset silently fell back to its much smaller final
        # RF-selected feature set (e.g. 15 or 6 features) instead of its
        # full pre-selection pool (32 or 12), regardless of which dataset
        # it was. That's why cross-dataset matching kept coming back with
        # 0 common features even for pairs the canonical maps were
        # specifically designed to overlap on.
        tgt_feature_names = pipelines[tgt]["pre_feature_names"]
        tgt_df = pd.DataFrame(
            pipelines[tgt]["X_test_pre"], columns=tgt_feature_names)
        tgt_df["label"] = pipelines[tgt]["y_test"][:len(tgt_df)]

        src_test_df = pd.DataFrame(pipelines[src]["X_test_pre"],
                                   columns=pipelines[src]["pre_feature_names"])
        src_test_df["label"] = pipelines[src]["y_test"]
        result = preparer.prepare(
            src_pipeline_output=pipelines[src],
            tgt_df_clean=tgt_df,
            tgt_feature_names=tgt_feature_names,
            src_test_df=src_test_df,
        )
        cross_data[(src, tgt)] = result

    cross_df = run_all_cross_dataset(
        pipelines=pipelines,
        classifiers=classifiers,
        vae_models=vaes,
        cross_data=cross_data,
    )

    if cross_df is not None and not cross_df.empty:
        for metric in ["f1", "mcc"]:
            plot_cross_dataset_heatmap(cross_df, metric=metric)
        plot_cross_dataset_bars(cross_df)
    else:
        log.warning("Cross-dataset DataFrame empty — skipping heatmaps")

    log.info("Cross-dataset experiments complete")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4b — STATISTICAL SIGNIFICANCE TESTING
# ──────────────────────────────────────────────────────────────────────────────

def stage_statistical_tests(dataset_names: list, subsample_svc_knn: int):
    from src.evaluation.statistical_tests import run_all_significance_tests

    pipelines = {}
    trainings = {}
    for ds in dataset_names:
        pipelines[ds] = _load_checkpoint(ds, "preprocess")
        trainings[ds] = _load_checkpoint(ds, "train")

    run_all_significance_tests(dataset_names, pipelines, trainings)
    log.info("Statistical significance testing complete")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 5 — SHAP
# ──────────────────────────────────────────────────────────────────────────────

def stage_shap(dataset_names: list):
    from src.explainability.shap_analysis import (
        run_shap_analysis, analyze_cross_dataset_shap_overlap
    )

    shap_tree_models = ["RandomForest", "XGBoost", "DecisionTree"]
    importance_dicts = {}

    for ds in dataset_names:
        pre = _load_checkpoint(ds, "preprocess")
        trn = _load_checkpoint(ds, "train")

        X_test   = pre["X_test"]
        features = pre["feature_names"]
        clfs     = trn["classifiers"]
        # (VAE not needed for SHAP)

        # REVISION: best tree model by validation MCC; SHAP on the bare
        # classifier (the SMOTE step of the pipeline is fit-time only)
        best_tree_clf = None
        best_m = -2.0
        for name in shap_tree_models:
            clf = clfs.get(name)
            if clf and (clf.val_mcc or -2.0) > best_m:
                best_m = clf.val_mcc
                best_tree_clf = clf

        if best_tree_clf is None:
            log.warning(f"[{ds}] No tree-based model found for SHAP")
            continue

        imp_df = run_shap_analysis(
            model=best_tree_clf.estimator,
            X_test=X_test,
            feature_names=features,
            dataset_name=ds,
            model_name=best_tree_clf.name,
        )
        importance_dicts[ds] = imp_df

    if len(importance_dicts) > 1:
        analyze_cross_dataset_shap_overlap(importance_dicts)

    log.info("SHAP analysis complete")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_global_seed(RANDOM_SEED)

    dataset_names = DATASET_NAMES if args.dataset == "all" else [args.dataset]
    run_all = args.stage == "all"

    log.info(f"Pipeline start | datasets: {dataset_names} | stage: {args.stage}")
    try:
        from utils.run_info import save_run_info
        save_run_info()
    except Exception as e:
        log.warning(f"run_info not saved: {e}")

    if run_all or args.stage == "preprocess":
        stage_preprocess(dataset_names)
        if args.stage == "preprocess":
            log.info("Stage complete. Exiting.")
            return

    if run_all or args.stage == "train":
        stage_train(dataset_names, args.subsample_svc_knn)
        if args.stage == "train":
            log.info("Stage complete. Exiting.")
            return

    if run_all or args.stage == "evaluate":
        stage_evaluate(dataset_names)
        if args.stage == "evaluate":
            return

    if run_all or args.stage == "cross":
        stage_cross_dataset(dataset_names)
        if args.stage == "cross":
            return

    if run_all or args.stage == "stats":
        stage_statistical_tests(dataset_names, args.subsample_svc_knn)
        if args.stage == "stats":
            return

    if run_all or args.stage == "shap":
        stage_shap(dataset_names)

    log.info("\n" + "="*60)
    log.info("FULL PIPELINE COMPLETE")
    log.info(f"Results -> {OUT_RESULTS}")
    log.info("="*60)


if __name__ == "__main__":
    main()
