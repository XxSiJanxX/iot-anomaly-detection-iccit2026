"""
Diagnostic v2: check duplication rate across MULTIPLE IoT-23 files with
different characteristics (not just one DDoS-heavy file), to see whether
extreme duplication is universal or concentrated in specific scenario types.

Tests a spread of file sizes/types based on what we already observed:
  - dataset10.csv : 510MB,  99.89% attack (heavy DDoS/flood scenario)
  - dataset12.csv : 8994MB, large scenario, attack rate unknown
  - dataset19.csv : 23.4MB, smaller scenario, attack rate unknown
  - dataset23.csv : 142.4MB, medium scenario, attack rate unknown

For the large files (12), we sample randomly rather than loading in full
to keep this diagnostic fast and memory-safe.
"""
import pandas as pd
import numpy as np
import os
import time

BASE_COLS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes",
    "history", "orig_pkts", "orig_ip_bytes", "resp_pkts",
    "resp_ip_bytes",
]
DROP_ID_COLS = ["ts", "uid", "id.orig_h", "id.orig_p",
               "id.resp_h", "id.resp_p", "local_orig", "local_resp"]

FILES_TO_TEST = [
    (r"D:\iot_anomaly_detection\data\raw\iot23\dataset19.csv", None),   # 23MB, full load
    (r"D:\iot_anomaly_detection\data\raw\iot23\dataset23.csv", None),   # 142MB, full load
    (r"D:\iot_anomaly_detection\data\raw\iot23\dataset17.csv", None),   # 2.9MB, full load
    (r"D:\iot_anomaly_detection\data\raw\iot23\dataset12.csv", 3_000_000),  # 9GB, sample 3M
]

def parse_file(path, nrows=None):
    read_kwargs = dict(
        sep="\x00", header=None, skiprows=1,
        names=["_line"], dtype=str, engine="python",
        on_bad_lines="skip",
    )
    if nrows:
        read_kwargs["nrows"] = nrows
    raw = pd.read_csv(path, **read_kwargs)["_line"]

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
    return df


print("="*70)
for path, nrows in FILES_TO_TEST:
    fname = os.path.basename(path)
    size_mb = os.path.getsize(path) / (1024**2)
    print(f"\nFile: {fname} ({size_mb:.1f} MB)" +
          (f" -- sampling first {nrows:,} rows" if nrows else " -- FULL load"))

    start = time.time()
    try:
        df = parse_file(path, nrows=nrows)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue
    elapsed = time.time() - start

    total = len(df)
    df["label_raw"] = df["label_raw"].astype(str).str.strip().str.lower()
    df["label"] = (~df["label_raw"].isin(["benign", "-", "nan", "none", ""])).astype(int)
    attack_rate = df["label"].mean()

    df_dedup = df.drop_duplicates(subset=[c for c in df.columns if c != "label_raw"])
    after = len(df_dedup)
    dup_rate = (total - after) / total * 100 if total > 0 else 0
    attack_rate_after = df_dedup["label"].mean() if after > 0 else 0

    print(f"  Loaded in {elapsed:.1f}s | Total rows: {total:,}")
    print(f"  Attack rate (before dedup): {attack_rate:.2%}")
    print(f"  Duplicate rate: {dup_rate:.2f}%")
    print(f"  Unique rows after dedup: {after:,}")
    print(f"  Attack rate (after dedup): {attack_rate_after:.2%}")

print("\n" + "="*70)
print("SUMMARY: compare duplicate rates across different scenario types")
print("to determine if 97%+ duplication is universal or concentrated")
print("in attack-flood-heavy files specifically.")
