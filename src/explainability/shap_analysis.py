"""
shap_analysis.py
----------------
SHAP TreeExplainer analysis for the best-performing supervised classifier.
Produces:
  - Summary plot (beeswarm) for top N features
  - Feature importance bar chart
  - Cross-dataset feature overlap analysis
Saves plots to outputs/plots/ and importance scores to outputs/results/
"""

import os
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.logger import get_logger
from configs.config import SHAP_TOP_N_FEATURES, SHAP_BACKGROUND_SAMPLES, OUT_PLOTS, OUT_RESULTS

log = get_logger(__name__)
os.makedirs(OUT_PLOTS, exist_ok=True)
os.makedirs(OUT_RESULTS, exist_ok=True)


def run_shap_analysis(
    model,
    X_test: np.ndarray,
    feature_names: list,
    dataset_name: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Run SHAP TreeExplainer on a tree-based model.

    Parameters
    ----------
    model        : fitted sklearn/XGBoost model (Random Forest or XGBoost)
    X_test       : test features (numpy array)
    feature_names: list of feature name strings
    dataset_name : for file naming and logging
    model_name   : 'RandomForest' or 'XGBoost'

    Returns
    -------
    DataFrame with feature names and mean |SHAP| values (sorted descending)
    """
    log.info(f"[{dataset_name}] Running SHAP for {model_name}...")

    # Background sample for explainer
    bg_size = min(SHAP_BACKGROUND_SAMPLES, len(X_test))
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_test), bg_size, replace=False)
    background = X_test[bg_idx]

    # TreeExplainer - fast for RF and XGBoost
    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
    )

    # Compute SHAP values on a subset for speed
    explain_size = min(2000, len(X_test))
    ex_idx = rng.choice(len(X_test), explain_size, replace=False)
    X_explain = X_test[ex_idx]

    shap_values = explainer.shap_values(X_explain, check_additivity=False)

    # shap_values' shape/type for a binary classifier varies by SHAP
    # version AND by which model produced it -- this isn't one fixed API,
    # it's three different conventions depending on what's underneath:
    #   1. list of 2 arrays, one per class (older SHAP convention)
    #   2. 3D ndarray, shape (n_samples, n_features, n_classes) -- seen
    #      with RandomForestClassifier, since sklearn's predict_proba
    #      always has 2 columns even for binary classification
    #   3. plain 2D ndarray, shape (n_samples, n_features) -- seen with
    #      XGBoost's binary classifier, which has a single raw-margin
    #      output and no class axis at all
    # The original code only handled cases 1 and 3, which is exactly why
    # this ran fine when XGBoost was the best model (nbaiot) but crashed
    # with "Per-column arrays must each be 1-dimensional" the moment
    # RandomForest was best instead (ciciot2023) -- RF's 3D output fell
    # through to the `else` branch unchanged, and .mean(axis=0) on a 3D
    # array leaves a 2D (n_features, n_classes) result, which pandas
    # can't accept as a single DataFrame column.
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]           # class 1 = attack
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]     # class 1 = attack
    else:
        shap_vals = shap_values              # already (n_samples, n_features)

    # Mean absolute SHAP per feature
    mean_abs = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    top_df = importance_df.head(SHAP_TOP_N_FEATURES)

    # -- Plot 1: Summary beeswarm -----------------------------------
    top_features = top_df["feature"].tolist()
    top_idx = [feature_names.index(f) for f in top_features]
    shap_top = shap_vals[:, top_idx]
    X_top = X_explain[:, top_idx]

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_top, X_top,
        feature_names=top_features,
        show=False, plot_type="dot",
        max_display=SHAP_TOP_N_FEATURES,
        color_bar_label="Feature value",
    )
    plt.title(f"SHAP Summary - {model_name} / {dataset_name.upper()}")
    plt.tight_layout()
    beeswarm_path = os.path.join(OUT_PLOTS, f"{dataset_name}_{model_name}_shap_beeswarm.png")
    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"SHAP beeswarm saved -> {beeswarm_path}")

    # -- Plot 2: Feature importance bar ----------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df["feature"][::-1], top_df["mean_abs_shap"][::-1],
            color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Feature Importance (Top {SHAP_TOP_N_FEATURES}) - "
                 f"{model_name} / {dataset_name.upper()}")
    ax.grid(axis="x", alpha=0.3)
    bar_path = os.path.join(OUT_PLOTS, f"{dataset_name}_{model_name}_shap_bar.png")
    fig.tight_layout()
    fig.savefig(bar_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"SHAP bar chart saved -> {bar_path}")

    # -- Save importance scores -------------------------------------
    csv_path = os.path.join(OUT_RESULTS, f"{dataset_name}_{model_name}_shap_importance.csv")
    importance_df.to_csv(csv_path, index=False)
    log.info(f"SHAP importance saved -> {csv_path}")

    return importance_df


def analyze_cross_dataset_shap_overlap(
    importance_dicts: dict,   # {dataset_name: importance_df}
    top_n: int = SHAP_TOP_N_FEATURES,
) -> pd.DataFrame:
    """
    Analyse which features appear in the top-N SHAP rankings across datasets.
    Returns a DataFrame showing overlap.
    """
    top_features = {
        ds: set(df.head(top_n)["feature"].tolist())
        for ds, df in importance_dicts.items()
    }
    datasets = list(top_features.keys())

    overlap_data = []
    for ds in datasets:
        row = {"dataset": ds}
        for other_ds in datasets:
            if ds == other_ds:
                row[other_ds] = top_n
            else:
                overlap = len(top_features[ds] & top_features[other_ds])
                row[other_ds] = overlap
        overlap_data.append(row)

    overlap_df = pd.DataFrame(overlap_data).set_index("dataset")
    log.info(f"SHAP top-{top_n} feature overlap across datasets:\n{overlap_df}")

    # Plot overlap heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    import seaborn as sns
    sns.heatmap(overlap_df.astype(float), annot=True, fmt=".0f",
                cmap="Blues", vmin=0, vmax=top_n, ax=ax)
    ax.set_title(f"SHAP Top-{top_n} Feature Overlap Across Datasets")
    out_path = os.path.join(OUT_PLOTS, "shap_cross_dataset_overlap.png")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"SHAP overlap heatmap saved -> {out_path}")

    csv_path = os.path.join(OUT_RESULTS, "shap_cross_dataset_overlap.csv")
    overlap_df.to_csv(csv_path)
    return overlap_df
