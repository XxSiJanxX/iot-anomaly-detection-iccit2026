"""
check_overlap.py  --  answers Reviewer 3, comment #1 (and the protocol question)

For each dataset it reports, using the SAVED preprocess checkpoints (no retraining):
  1. How many test rows are exact copies of a training row (train/test leakage)
  2. Of those, how many carry a DIFFERENT label than the training copy
  3. Duplicate rate inside the test set itself
And for the cross-dataset "protocol" feature:
  4. What values CICIoT2023's protocol_type actually holds
  5. How IoT-23's proto strings were label-encoded

Run from the project root:   python check_overlap.py
"""
import os
import joblib
import numpy as np
import pandas as pd

MODELS = os.path.join("outputs", "models")
DATASETS = ["nbaiot", "ciciot2023", "iot23"]


def row_hashes(X):
    """One 64-bit hash per row (exact-match identity on raw feature values)."""
    return pd.util.hash_pandas_object(pd.DataFrame(X), index=False).values


print("=" * 70)
print("PART A - train/test overlap (pre-RF-selection raw features)")
print("=" * 70)

for ds in DATASETS:
    path = os.path.join(MODELS, f"{ds}_preprocess_checkpoint.pkl")
    if not os.path.exists(path):
        print(f"\n[{ds}] checkpoint not found: {path}")
        continue
    print(f"\n[{ds}] loading {path} ...")
    ck = joblib.load(path)

    X_tr, y_tr = np.asarray(ck["X_train_pre"]), np.asarray(ck["y_train_pre"])
    X_te, y_te = np.asarray(ck["X_test_pre"]),  np.asarray(ck["y_test"])
    print(f"  train rows (deduped, pre-SMOTE pool): {len(X_tr):,}")
    print(f"  test rows:                            {len(X_te):,}")

    h_tr, h_te = row_hashes(X_tr), row_hashes(X_te)

    # first label seen in train for each unique train vector
    tr_map = pd.Series(y_tr, index=h_tr)
    tr_map = tr_map[~tr_map.index.duplicated(keep="first")]

    in_train = np.isin(h_te, tr_map.index.values)
    n_in = int(in_train.sum())
    print(f"  test rows that also appear in train:  {n_in:,} "
          f"({n_in / len(X_te):.1%})")

    if n_in:
        tr_lab = tr_map.reindex(h_te[in_train]).values
        conflict = int((tr_lab != y_te[in_train]).sum())
        print(f"    ...with a DIFFERENT label than train: {conflict:,} "
              f"({conflict / n_in:.1%} of overlapping rows)")
        for lab, name in [(0, "benign"), (1, "attack")]:
            m = y_te == lab
            if m.sum():
                print(f"    overlap among test {name:6s}: "
                      f"{in_train[m].mean():.1%} of {int(m.sum()):,}")

    n_unique_te = len(np.unique(h_te))
    print(f"  unique rows inside test:              {n_unique_te:,} "
          f"(test duplicate rate {1 - n_unique_te / len(X_te):.1%})")

    if ds in ("nbaiot", "ciciot2023"):
        print("  NOTE: train pool here is the 500k pre-SMOTE subsample, so the "
              "true overlap with the full train split is HIGHER than shown.")

    del ck, X_tr, X_te

print("\n" + "=" * 70)
print("PART B - what the shared 'protocol' feature actually contains")
print("=" * 70)

ck = joblib.load(os.path.join(MODELS, "ciciot2023_preprocess_checkpoint.pkl"))
names = list(ck["pre_feature_names"])
col = next((c for c in names if "protocol" in c.lower()), None)
if col:
    vals = pd.Series(np.asarray(ck["X_train_pre"])[:, names.index(col)])
    print(f"\n[ciciot2023] column '{col}': {vals.nunique():,} distinct values")
    print("  most common values:")
    print(vals.round(3).value_counts().head(10).to_string())
else:
    print("\n[ciciot2023] no protocol column in pre_feature_names:", names)
del ck

cleaner = joblib.load(os.path.join(MODELS, "iot23_cleaner.pkl"))
enc = getattr(cleaner, "_cat_encoders", {}).get("proto")
if enc is not None:
    print("\n[iot23] 'proto' label encoding (string -> code):")
    for i, c in enumerate(enc.classes_):
        print(f"  {c!r:10s} -> {i}")
else:
    print("\n[iot23] no 'proto' encoder found in cleaner")

print("\nFor reference, IANA protocol numbers: icmp=1, tcp=6, udp=17")
