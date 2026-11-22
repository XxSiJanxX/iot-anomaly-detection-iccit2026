"""
refresh_fig2.py
----------------
Regenerates ONLY the three VAE reconstruction-error plots (Fig. 2) using the
restyled plot_vae_reconstruction_dist() in src/evaluation/visualizations.py.

Loads already-saved checkpoints and VAE weights -- no retraining, no full
re-run of --stage evaluate needed. Takes seconds per dataset instead of
CICIoT2023's ~70-minute training cost.

Usage:
    python refresh_fig2.py
    python refresh_fig2.py --dataset nbaiot
"""

import argparse
import sys

from configs.config import DATASET_NAMES, OUT_MODELS
from utils.logger import get_logger

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                         choices=DATASET_NAMES + ["all"])
    args = parser.parse_args()
    dataset_names = DATASET_NAMES if args.dataset == "all" else [args.dataset]

    import joblib
    import os
    from src.models.vae import TrainedVAE
    from src.evaluation.visualizations import plot_vae_reconstruction_dist

    for ds in dataset_names:
        pre_path = os.path.join(OUT_MODELS, f"{ds}_preprocess_checkpoint.pkl")
        trn_path = os.path.join(OUT_MODELS, f"{ds}_train_checkpoint.pkl")
        if not (os.path.exists(pre_path) and os.path.exists(trn_path)):
            log.warning(f"[{ds}] Missing checkpoint(s), skipping "
                        f"({pre_path} / {trn_path})")
            continue

        pre = joblib.load(pre_path)
        trn = joblib.load(trn_path)

        vae_X_test = pre["vae_X_test"]
        vae_y_test = pre["vae_y_test"]
        vae = TrainedVAE.load(trn["vae_dir"], trn["vae_input_dim"])

        plot_vae_reconstruction_dist(
            vae, vae_X_test, vae_y_test, ds, threshold=vae.threshold)
        log.info(f"[{ds}] Fig. 2 panel refreshed -> "
                 f"outputs/plots/{ds}_vae_reconstruction_dist.png")


if __name__ == "__main__":
    sys.exit(main())
