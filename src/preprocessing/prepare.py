"""
prepare.py  (camera-ready revision)
-----------------------------------
Steps that must happen ONCE on the full dataset, BEFORE the train/val/test
split, so that both the supervised and the VAE branch see exactly the same
rows (Reviewer 3, comment 1):

  1. cap_rows()          stratified cap to MAX_ROWS (CICIoT2023 only)
  2. harmonize()         dataset-specific value fixes needed for cross-dataset
                         feature equivalence (IoT-23 'proto' -> IANA number)
  3. deduplicate()       group identical feature vectors and keep ONE row per
                         vector, so an identical observation can never appear
                         in two partitions. Vectors seen with BOTH labels are
                         resolved to their majority label (ties dropped); the
                         number of such conflicts is reported.
  4. split_overlap()     verifies, after splitting, that no feature vector is
                         shared between train / val / test.

All counts are written to OUT_RESULTS/<dataset>_data_summary.json so they
can be reported in the paper's dataset table (Reviewer 1, comment 3).
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configs.config import MAX_ROWS, OUT_RESULTS, RANDOM_SEED
from utils.logger import get_logger

log = get_logger(__name__)

NON_FEATURE_COLS = ("label", "attack_cat")

# Zeek conn.log 'proto' strings -> IANA protocol numbers, i.e. the same
# encoding CICIoT2023's 'protocol_type' already uses (TCP=6, UDP=17, ICMP=1).
IANA_PROTOCOL = {"icmp": 1, "tcp": 6, "udp": 17, "gre": 47,
                 "ipv6-icmp": 58, "icmpv6": 58, "sctp": 132}


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def row_hashes(df: pd.DataFrame, cols: list) -> np.ndarray:
    """64-bit hash per row over `cols` (exact-match identity)."""
    return pd.util.hash_pandas_object(df[cols], index=False).values


# ------------------------------------------------------------------------------
def cap_rows(df: pd.DataFrame, dataset_name: str, stats: dict) -> pd.DataFrame:
    stats["rows_loaded"] = int(len(df))
    if len(df) <= MAX_ROWS:
        stats["rows_after_cap"] = int(len(df))
        return df
    log.info(f"[{dataset_name}] {len(df):,} rows > {MAX_ROWS:,} -- stratified cap")
    df, _ = train_test_split(df, train_size=MAX_ROWS, stratify=df["label"],
                             random_state=RANDOM_SEED)
    df = df.reset_index(drop=True)
    stats["rows_after_cap"] = int(len(df))
    return df


def harmonize(df: pd.DataFrame, dataset_name: str, stats: dict) -> pd.DataFrame:
    if dataset_name == "iot23" and "proto" in df.columns:
        raw = df["proto"].astype(str).str.strip().str.lower()
        mapped = raw.map(IANA_PROTOCOL)
        unknown = int(mapped.isna().sum())
        stats["iot23_proto_values"] = {k: int(v) for k, v in
                                       raw.value_counts().head(10).items()}
        stats["iot23_proto_unmapped_rows"] = unknown
        df["proto"] = mapped.astype("float32")   # unmapped -> NaN -> median fill
        log.info(f"[iot23] proto mapped to IANA numbers "
                 f"(tcp=6, udp=17, icmp=1); unmapped rows: {unknown:,}")
    return df


def deduplicate(df: pd.DataFrame, dataset_name: str, stats: dict) -> pd.DataFrame:
    cols = feature_columns(df)
    n_in = len(df)
    y = df["label"].values.astype(np.int8)
    h = row_hashes(df, cols)

    counts = (pd.DataFrame({"h": h, "y": y})
              .groupby(["h", "y"]).size().unstack(fill_value=0)
              .reindex(columns=[0, 1], fill_value=0))
    n_unique = len(counts)
    conflict = (counts[0] > 0) & (counts[1] > 0)
    majority = pd.Series(np.where(counts[1] > counts[0], 1,
                                  np.where(counts[0] > counts[1], 0, -1)),
                         index=counts.index)
    n_ties = int((majority == -1).sum())

    row_major = majority.reindex(h).values
    candidate = (row_major == y)                       # rows carrying the kept label
    keep = np.zeros(n_in, dtype=bool)
    cand_idx = np.flatnonzero(candidate)
    first_among_cand = ~pd.Series(h[cand_idx]).duplicated(keep="first").values
    keep[cand_idx[first_among_cand]] = True

    out = df.loc[keep].reset_index(drop=True)
    stats.update({
        "rows_before_dedup":          int(n_in),
        "unique_feature_vectors":     int(n_unique),
        "duplicate_rows_removed":     int(n_in - n_unique),
        "duplicate_rate":             round(1 - n_unique / n_in, 4) if n_in else 0.0,
        "conflicting_label_vectors":  int(conflict.sum()),
        "conflicting_label_rows":     int(counts.loc[conflict, [0, 1]].sum().sum()),
        "conflict_ties_dropped":      n_ties,
        "rows_after_dedup":           int(len(out)),
        "benign_after_dedup":         int((out["label"] == 0).sum()),
        "attack_after_dedup":         int((out["label"] == 1).sum()),
    })
    log.info(f"[{dataset_name}] De-dup BEFORE split: {n_in:,} -> {len(out):,} rows "
             f"({stats['duplicate_rate']:.1%} duplicates) | feature vectors with "
             f"conflicting labels: {stats['conflicting_label_vectors']:,} "
             f"({stats['conflicting_label_rows']:,} rows; resolved by majority, "
             f"{n_ties:,} ties dropped)")
    return out


def prepare_dataset(df: pd.DataFrame, dataset_name: str) -> tuple:
    """cap -> harmonize -> dedup. Returns (df, stats)."""
    stats = {"dataset": dataset_name}
    df = cap_rows(df, dataset_name, stats)
    df = harmonize(df, dataset_name, stats)
    df = deduplicate(df, dataset_name, stats)
    return df, stats


def split_overlap(train_df, val_df, test_df) -> dict:
    """Count feature vectors shared between partitions (should be 0)."""
    cols = feature_columns(train_df)
    h_tr = row_hashes(train_df, cols)
    h_va = row_hashes(val_df, cols)
    h_te = row_hashes(test_df, cols)
    return {
        "test_rows_in_train": int(np.isin(h_te, h_tr).sum()),
        "val_rows_in_train":  int(np.isin(h_va, h_tr).sum()),
        "test_rows_in_val":   int(np.isin(h_te, h_va).sum()),
    }


def save_summary(dataset_name: str, stats: dict):
    os.makedirs(OUT_RESULTS, exist_ok=True)
    path = os.path.join(OUT_RESULTS, f"{dataset_name}_data_summary.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2, default=lambda o: o.item()
                  if hasattr(o, "item") else str(o))
    log.info(f"[{dataset_name}] Data summary saved -> {path}")
