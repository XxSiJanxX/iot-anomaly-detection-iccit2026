"""
Diagnostic: check whether IoT-23's extreme duplication rate (~97%) is a
real dataset property or an artifact of our row-capping/sampling.

Loads ONE large file COMPLETELY (no cap), checks duplication rate on the
full file directly -- no preprocessing pipeline involved, just raw
pandas duplicate detection on the feature columns we actually use.

Run this ON THE USER'S MACHINE.
"""
import pandas as pd
import numpy as np
import os
import time

PATH = r"D:\iot_anomaly_detection\data\raw\iot23\dataset10.csv"
# dataset10.csv is 510MB, 3,394,338 lines -- large enough to be meaningful,
# small enough to load fully without excessive RAM risk (~2-3 min expected).

BASE_COLS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes",
    "history", "orig_pkts", "orig_ip_bytes", "resp_pkts",
    "resp_ip_bytes",
]
DROP_ID_COLS = ["ts", "uid", "id.orig_h", "id.orig_p",
               "id.resp_h", "id.resp_p", "local_orig", "local_resp"]

print(f"Loading FULL file (no cap): {PATH}")
print(f"File size: {os.path.getsize(PATH) / (1024**2):.1f} MB")
start = time.time()

raw = pd.read_csv(
    PATH, sep="\x00", header=None, skiprows=1,
    names=["_line"], dtype=str, engine="python",
    on_bad_lines="skip",
)["_line"]
print(f"Read {len(raw):,} lines in {time.time()-start:.1f}s")

split_df = raw.str.split(",", n=20, expand=True)
base_part = split_df.iloc[:, :20]
base_part.columns = BASE_COLS
tail_part = split_df.iloc[:, 20].astype(str)
tail_split = tail_part.str.split()

label_val = tail_split.apply(
    lambda toks: toks[-2] if isinstance(toks, list) and len(toks) >= 2
                else (toks[-1] if isinstance(toks, list) and len(toks) == 1 else "-")
)

df = base_part.copy()
df["label_raw"] = label_val.values
df.drop(columns=[c for c in DROP_ID_COLS if c in df.columns], inplace=True)

print(f"\nTotal rows in FULL file: {len(df):,}")

# Check duplicates on the FEATURE columns (excluding identifiers we already
# dropped) -- this mirrors what the cleaner.fit_transform() duplicate check
# actually does.
before = len(df)
df_dedup = df.drop_duplicates()
after = len(df_dedup)
dup_rate = (before - after) / before * 100

print(f"After drop_duplicates(): {after:,} rows")
print(f"Duplicate rate: {dup_rate:.2f}%")
print()

# Also check: what does the label distribution look like in the FULL file
# vs after dedup?
df["label_raw"] = df["label_raw"].astype(str).str.strip().str.lower()
df["label"] = (~df["label_raw"].isin(["benign", "-", "nan", "none", ""])).astype(int)
print(f"FULL file attack rate: {df['label'].mean():.2%}")

df_dedup_labeled = df.drop_duplicates(subset=[c for c in df.columns if c != "label_raw"])
print(f"Attack rate after dedup: {df_dedup_labeled['label'].mean():.2%}")

print()
print("=== Comparison to our 2M-row-capped sample's dedup rate ===")
print("(from actual pipeline run: 6,812,622 / 7,000,000 = 97.32% duplicate)")
print(f"(this full-file test:      {dup_rate:.2f}% duplicate)")
