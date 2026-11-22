"""
Diagnostic script: inspect actual VAE score distributions on val vs test
for N-BaIoT, to find the real source of the val/test discrepancy.

Run this ON THE USER'S MACHINE since checkpoints are local to their disk.
"""
import sys
sys.path.insert(0, "D:/iot_anomaly_detection")

import joblib
import numpy as np
import os

OUT_MODELS = "D:/iot_anomaly_detection/outputs/models"

# Load preprocess checkpoint
pre = joblib.load(os.path.join(OUT_MODELS, "nbaiot_preprocess_checkpoint.pkl"))

vae_X_val   = pre["vae_X_val"]
vae_y_val   = pre["vae_y_val"]
vae_X_test  = pre["vae_X_test"]
vae_y_test  = pre["vae_y_test"]

print(f"vae_X_val shape:  {vae_X_val.shape}")
print(f"vae_X_test shape: {vae_X_test.shape}")
print(f"vae_y_val attack rate:  {vae_y_val.mean():.4f}")
print(f"vae_y_test attack rate: {vae_y_test.mean():.4f}")
print()

# Check if X_val and X_test have similar value ranges
print("=== Feature value range comparison ===")
print(f"X_val  min={vae_X_val.min():.4f} max={vae_X_val.max():.4f} mean={vae_X_val.mean():.4f}")
print(f"X_test min={vae_X_test.min():.4f} max={vae_X_test.max():.4f} mean={vae_X_test.mean():.4f}")
print()

# Load the trained VAE
from src.models.vae import TrainedVAE
vae = TrainedVAE.load(os.path.join(OUT_MODELS, "nbaiot_VAE"), vae_X_val.shape[1])

print(f"Loaded VAE threshold: {vae.threshold}")
print()

# Compute scores on val
val_scores = vae.anomaly_score(vae_X_val)
print("=== VAL anomaly scores ===")
print(f"  Normal (y=0): min={val_scores[vae_y_val==0].min():.4f} "
      f"max={val_scores[vae_y_val==0].max():.4f} "
      f"mean={val_scores[vae_y_val==0].mean():.4f}")
print(f"  Attack (y=1): min={val_scores[vae_y_val==1].min():.4f} "
      f"max={val_scores[vae_y_val==1].max():.4f} "
      f"mean={val_scores[vae_y_val==1].mean():.4f}")
print()

# Compute scores on test
test_scores = vae.anomaly_score(vae_X_test)
print("=== TEST anomaly scores ===")
print(f"  Normal (y=0): min={test_scores[vae_y_test==0].min():.4f} "
      f"max={test_scores[vae_y_test==0].max():.4f} "
      f"mean={test_scores[vae_y_test==0].mean():.4f}")
print(f"  Attack (y=1): min={test_scores[vae_y_test==1].min():.4f} "
      f"max={test_scores[vae_y_test==1].max():.4f} "
      f"mean={test_scores[vae_y_test==1].mean():.4f}")
print()

# Check how many normal test samples exceed threshold
normal_test_scores = test_scores[vae_y_test == 0]
above_thresh = (normal_test_scores > vae.threshold).sum()
print(f"Normal test samples ABOVE threshold ({vae.threshold:.4f}): "
      f"{above_thresh} / {len(normal_test_scores)} "
      f"({above_thresh/len(normal_test_scores)*100:.2f}%)")

normal_val_scores = val_scores[vae_y_val == 0]
above_thresh_val = (normal_val_scores > vae.threshold).sum()
print(f"Normal val samples ABOVE threshold ({vae.threshold:.4f}): "
      f"{above_thresh_val} / {len(normal_val_scores)} "
      f"({above_thresh_val/len(normal_val_scores)*100:.2f}%)")

# Call anomaly_score TWICE on the same val data to check determinism
print()
print("=== Determinism check: calling anomaly_score twice on same val data ===")
scores_run1 = vae.anomaly_score(vae_X_val[:1000])
scores_run2 = vae.anomaly_score(vae_X_val[:1000])
max_diff = np.abs(scores_run1 - scores_run2).max()
print(f"Max difference between two calls on same data: {max_diff:.8f}")
print(f"(Should be ~0.0 if deterministic, large if still stochastic)")
