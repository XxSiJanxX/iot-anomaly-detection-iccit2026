"""Redraw the three VAE score-distribution plots (Fig. 2) from the saved
models and test arrays. No training or experiment is rerun.

Run from the project root:  python regen_vae_plots.py
Output: outputs_revision/plots/<dataset>_vae_reconstruction_dist.png
"""
import os
import joblib

from configs.config import OUT_MODELS
from src.models.vae import TrainedVAE
from src.evaluation.visualizations import plot_vae_reconstruction_dist

for ds in ["nbaiot", "ciciot2023", "iot23"]:
    pre = joblib.load(os.path.join(OUT_MODELS, f"{ds}_preprocess_checkpoint.pkl"))
    X, y = pre["vae_X_test"], pre["vae_y_test"]
    del pre
    vae = TrainedVAE.load(os.path.join(OUT_MODELS, f"{ds}_VAE"), X.shape[1])
    plot_vae_reconstruction_dist(vae, X, y, ds, threshold=vae.threshold)
    print(f"{ds}: done")
