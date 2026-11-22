"""
visualizations.py
-----------------
All plots defined in Section 9.1 of the proposal.

Plots produced:
  1. Grouped bar chart - Accuracy, Precision, Recall, F1, MCC per model/dataset
  2. ROC curves - all models per dataset
  3. Ablation bar chart - F1, Recall, FPR for 4 configs
  4. Cross-dataset heatmap - F1 and MCC (the central result figure)
  5. Confusion matrices - best supervised + VAE + best hybrid
  6. VAE reconstruction error distributions - benign vs anomalous
  7. SHAP summary plots - (called from shap_analysis.py)
  8. VAE training/validation ELBO loss curves
  9. Efficiency bar chart - training time and inference latency
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for VS Code / headless
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, matthews_corrcoef

from utils.logger import get_logger
from configs.config import OUT_PLOTS, DATASET_NAMES

log = get_logger(__name__)
os.makedirs(OUT_PLOTS, exist_ok=True)

# -- Style ---------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi":     150,
    # REVISION: larger fonts so labels stay readable at IEEE column width
    "font.size":      14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.facecolor": "white",
})
PALETTE = sns.color_palette("tab10")


# ------------------------------------------------------------------------------
# 1. GROUPED BAR CHART
# ------------------------------------------------------------------------------
def plot_metric_comparison(results_df: pd.DataFrame, dataset_name: str):
    """Grouped bar chart: Accuracy, Precision, Recall, F1, MCC per model."""
    metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
    df = results_df[results_df["dataset"] == dataset_name].copy()

    x = np.arange(len(df))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df[metric], width, label=metric.upper(),
               color=PALETTE[i])

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison - {dataset_name.upper()}")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    _save(fig, f"{dataset_name}_metric_comparison.png")


# ------------------------------------------------------------------------------
# 2. ROC CURVES
# ------------------------------------------------------------------------------
def plot_roc_curves(roc_data: dict, dataset_name: str):
    """
    roc_data: {model_name: (fpr_array, tpr_array, auc_score)}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    for i, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, lw=2, color=PALETTE[i],
                label=f"{name} (AUC={auc_val:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves - {dataset_name.upper()}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, f"{dataset_name}_roc_curves.png")


def compute_roc(model, X_test, y_test):
    """Returns (fpr, tpr, auc_score) for a model."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)
    elif hasattr(model, "anomaly_score"):
        scores = model.anomaly_score(X_test)
    else:
        scores = model.predict(X_test).astype(float)

    fpr, tpr, _ = roc_curve(y_test, scores)
    return fpr, tpr, auc(fpr, tpr)


# ------------------------------------------------------------------------------
# 3. ABLATION BAR CHART
# ------------------------------------------------------------------------------
def plot_ablation(ablation_df: pd.DataFrame, dataset_name: str):
    """Bar chart: F1, Recall, FPR for 4 ablation configs."""
    metrics = ["f1", "recall", "fpr"]
    df = ablation_df.copy()

    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df[metric], width, label=metric.upper(),
               color=PALETTE[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(df["ablation_config"], rotation=10, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Ablation Study - {dataset_name.upper()}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, f"{dataset_name}_ablation.png")


# ------------------------------------------------------------------------------
# 4. CROSS-DATASET HEATMAP (central result figure)
# ------------------------------------------------------------------------------
def plot_cross_dataset_heatmap(cross_df: pd.DataFrame, metric: str = "f1"):
    """
    Heatmap: rows = source dataset, cols = target dataset.
    Handles None values (no common features) with N/A annotation.
    """
    if cross_df.empty:
        log.warning("Cannot plot cross-dataset heatmap: empty dataframe")
        return

    datasets = DATASET_NAMES
    pivot = pd.DataFrame(index=datasets, columns=datasets, dtype=float)

    for _, row in cross_df.iterrows():
        src = row.get("src_dataset")
        tgt = row.get("tgt_dataset")
        val = row.get(metric)
        if src in datasets and tgt in datasets:
            try:
                pivot.loc[src, tgt] = float(val)
            except (TypeError, ValueError):
                pivot.loc[src, tgt] = np.nan

    for d in datasets:
        pivot.loc[d, d] = np.nan

    # Annotation matrix
    annot = pivot.copy().astype(object)
    for r in datasets:
        for c in datasets:
            if r == c:
                annot.loc[r, c] = "-"
            elif pd.isna(pivot.loc[r, c]):
                annot.loc[r, c] = "N/A"
            else:
                annot.loc[r, c] = f"{pivot.loc[r, c]:.3f}"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot.astype(float),
        annot=annot, fmt="",
        cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 12},
        mask=pivot.isna(),
    )
    ax.set_facecolor("#eeeeee")
    ax.set_title(
        f"Cross-Dataset Generalization — {metric.upper()}\n"
        "(rows=train, cols=test | N/A=no common features)"
    )
    ax.set_xlabel("Test Dataset")
    ax.set_ylabel("Train Dataset")
    _save(fig, f"cross_dataset_{metric}.png")


def plot_cross_dataset_bars(cross_df: pd.DataFrame, mcc_gap_flag: float = 0.5):
    """
    Grouped bar chart of F1 / MCC / AUC for every source->target pair that
    actually produced a numeric result (i.e. pairs with common features).
    Pairs with zero common features (e.g. N-BaIoT vs. either flow-based
    dataset) are excluded here since they have no bars to draw; they are
    already reported as "no common features" in Table VI / the heatmap.

    Where F1 is high but MCC is near zero, an annotation is added calling
    out the gap -- this is the paper's own point (F1 can look deceptively
    good under severe class imbalance while MCC reveals chance-level
    performance), made visible directly on the figure instead of only in
    prose.
    """
    if cross_df.empty:
        log.warning("Cannot plot cross-dataset bars: empty dataframe")
        return

    rows = cross_df.dropna(subset=["mcc"]).copy()
    if rows.empty:
        log.warning("Cross-dataset bars: no pairs with numeric results")
        return

    rows["pair"] = rows["src_dataset"] + " \u2192 " + rows["tgt_dataset"]
    metrics = ["f1", "mcc", "roc_auc"]
    labels = ["F1", "MCC", "AUC"]
    colors = [PALETTE[0], PALETTE[3], PALETTE[2]]

    x = np.arange(len(rows))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (m, lbl, c) in enumerate(zip(metrics, labels, colors)):
        vals = rows[m].astype(float).values
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=lbl, color=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    for i, (_, r) in enumerate(rows.iterrows()):
        if r["f1"] - r["mcc"] > mcc_gap_flag:
            ax.annotate("F1 misleading:\nMCC \u2248 chance",
                        xy=(x[i] - width, r["f1"]),
                        xytext=(x[i] - width, min(r["f1"] + 0.22, 1.28)),
                        ha="center", fontsize=8, color="#333333",
                        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(rows["pair"], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.35)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Cross-Dataset Transfer: F1 vs. MCC vs. AUC", fontweight="bold")
    ax.legend(loc="upper right", ncol=3, frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.2, linestyle=":")
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, "cross_dataset_bars.png")


# ------------------------------------------------------------------------------
# 5. CONFUSION MATRICES
# ------------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, model_name: str, dataset_name: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name} / {dataset_name.upper()}")
    _save(fig, f"{dataset_name}_{model_name}_confusion.png")


# ------------------------------------------------------------------------------
# 6. VAE RECONSTRUCTION ERROR DISTRIBUTION
# ------------------------------------------------------------------------------
def plot_vae_reconstruction_dist(vae_model, X_test, y_test,
                                  dataset_name: str, threshold: float = None):
    """VAE reconstruction-error distribution: benign vs. attack.

    Restyled for print: semi-transparent histogram + smooth KDE overlay
    (so the overlap region the paper's narrative depends on is easy to
    see at a glance), a cleaner two-color palette, and an inline MCC
    annotation that ties the visual separation directly to the reported
    metric. Falls back to histogram-only if a class has too few / too
    uniform values for a stable KDE fit (e.g. a near-degenerate class).
    """
    scores = vae_model.anomaly_score(X_test)
    benign  = scores[y_test == 0]
    attack  = scores[y_test == 1]

    normal_color = "#2E86AB"   # teal-blue
    attack_color = "#E4572E"   # warm coral-red

    fig, ax = plt.subplots(figsize=(9, 5))

    lo, hi = float(np.min(scores)), float(np.max(scores))
    pad = (hi - lo) * 0.02 if hi > lo else 1e-6
    bins = np.linspace(lo - pad, hi + pad, 80)

    for data, color, name in [(benign, normal_color, "Normal"),
                               (attack, attack_color, "Attack")]:
        ax.hist(data, bins=bins, alpha=0.30, color=color, density=True,
                label="_nolegend_")
        try:
            if len(data) > 1 and np.std(data) > 1e-12:
                sns.kdeplot(data, ax=ax, color=color, lw=2.4, fill=True,
                            alpha=0.25, label=name, clip=(lo, hi))
            else:
                raise ValueError("degenerate distribution")
        except Exception:
            ax.hist(data, bins=bins, alpha=0.6, color=color,
                    density=True, label=name)

    if threshold is not None:
        ax.axvline(threshold, color="#333333", linestyle="--", lw=1.8,
                    label=f"Threshold={threshold:.4f}")
        try:
            preds = (scores > threshold).astype(int)
            mcc = matthews_corrcoef(y_test, preds)
            ax.text(0.97, 0.95, f"MCC = {mcc:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="#333333", alpha=0.9))
        except Exception:
            pass

    ax.set_xlabel("ELBO Reconstruction Loss")
    ax.set_ylabel("Density")
    ax.set_title(f"VAE Reconstruction Error Distribution \u2014 {dataset_name.upper()}",
                 fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.2, linestyle=":")
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, f"{dataset_name}_vae_reconstruction_dist.png")


# ------------------------------------------------------------------------------
# 8. VAE TRAINING LOSS CURVES
# ------------------------------------------------------------------------------
def plot_vae_loss_curves(history: dict, dataset_name: str, beta: float):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.get("loss", []),     label="Train ELBO", color="steelblue")
    ax.plot(history.get("val_loss", []), label="Val ELBO",   color="firebrick",
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO Loss")
    ax.set_title(f"VAE Training Curves - {dataset_name.upper()} (β={beta})")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, f"{dataset_name}_vae_loss_curves.png")


# ------------------------------------------------------------------------------
# 9. EFFICIENCY BAR CHART
# ------------------------------------------------------------------------------
def plot_efficiency(results_df: pd.DataFrame, dataset_name: str):
    """Side-by-side bars: training time (s) and inference latency (ms)."""
    df = results_df[results_df["dataset"] == dataset_name].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(df["model"], df["train_time_s"], color=PALETTE[:len(df)])
    axes[0].set_title("Training Time (s)")
    axes[0].set_ylabel("Seconds")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(df["model"], df["latency_ms"], color=PALETTE[:len(df)])
    axes[1].set_title("Inference Latency (ms/sample)")
    axes[1].set_ylabel("Milliseconds")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle(f"Computational Efficiency - {dataset_name.upper()}")
    plt.tight_layout()
    _save(fig, f"{dataset_name}_efficiency.png")


# ------------------------------------------------------------------------------
# HELPER
# ------------------------------------------------------------------------------
def _save(fig, filename: str):
    path = os.path.join(OUT_PLOTS, filename)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot saved -> {path}")
