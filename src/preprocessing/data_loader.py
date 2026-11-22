"""
data_loader.py
--------------
Loads raw CSV files for each dataset into a standardised DataFrame with:
  - lowercase column names
  - a unified binary 'label' column  (0 = normal, 1 = attack)

Memory-efficient: reads CSVs one at a time, assigns label immediately,
downcasts numeric dtypes, then concatenates once at the end.
"""

import os
import glob
import pandas as pd
import numpy as np
from utils.logger import get_logger
from configs.config import DATA_RAW, DATASETS, RANDOM_SEED

# Used for reproducible random sampling of rows from very large files
# (separate name for clarity at the call site, same underlying seed
# used throughout the project for consistency).
RANDOM_SEED_FOR_SAMPLING = RANDOM_SEED

log = get_logger(__name__)


# --------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------

def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _safe_read(path: str, label: int = None) -> pd.DataFrame | None:
    """
    Read one CSV safely:
      - normalise column names
      - downcast dtypes
      - assign label column efficiently (avoids fragmentation warning)
    """
    try:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_")
                      for c in df.columns]
        df = _downcast(df)
        if label is not None:
            # Assign via concat to avoid fragmentation
            label_series = pd.Series(
                np.full(len(df), label, dtype=np.int8), name="label"
            )
            df = pd.concat([df, label_series], axis=1)
        return df
    except Exception as e:
        log.warning(f"Skipping {path}: {e}")
        return None


# --------------------------------------------------------------
# INTERNAL LOADERS
# --------------------------------------------------------------

def _load_nbaiot() -> pd.DataFrame:
    """
    N-BaIoT:
      data/raw/nbaiot/<device>/<benign|gafgyt_attacks|mirai_attacks>/*.csv
      'benign' anywhere in path -> label 0, else -> label 1
    """
    raw_dir = os.path.join(DATA_RAW, DATASETS["nbaiot"]["raw_subdir"])
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"N-BaIoT directory not found: {raw_dir}"
        )

    csv_files = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
    # Exclude root-level descriptor CSVs (demonstrate_structure.csv etc.)
    csv_files = [f for f in csv_files
                 if os.path.dirname(f) != raw_dir]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")

    log.info(f"N-BaIoT: found {len(csv_files)} CSV files - loading...")

    frames = []
    for i, path in enumerate(csv_files, 1):
        parts = path.lower().replace("\\", "/").split("/")
        label = 0 if "benign" in parts else 1
        df = _safe_read(path, label=label)
        if df is not None:
            frames.append(df)
        if i % 10 == 0:
            log.info(f"  Loaded {i}/{len(csv_files)} files...")

    log.info("N-BaIoT: concatenating...")
    combined = pd.concat(frames, ignore_index=True, copy=False)
    log.info(f"N-BaIoT loaded: {len(combined):,} rows, {combined.shape[1]} cols | "
             f"Attack rate: {combined['label'].mean():.2%}")
    return combined


def _load_unsw() -> pd.DataFrame:
    """
    UNSW-NB15: 4 CSV part files, flat in data/raw/unsw/
    Label column: 'label' or 'Label' (0=normal, 1=attack, already binary)
    """
    raw_dir = os.path.join(DATA_RAW, DATASETS["unsw"]["raw_subdir"])
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"UNSW-NB15 directory not found: {raw_dir}")

    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")

    log.info(f"UNSW-NB15: found {len(csv_files)} CSV files - loading...")

    # Official UNSW-NB15 column names (49 columns, label is last)
    UNSW_COLS = [
        "srcip","sport","dstip","dsport","proto","state","dur","sbytes",
        "dbytes","sttl","dttl","sloss","dloss","service","sload","dload",
        "spkts","dpkts","swin","dwin","stcpb","dtcpb","smeansz","dmeansz",
        "trans_depth","res_bdy_len","sjit","djit","stime","ltime","sintpkt",
        "dintpkt","tcprtt","synack","ackdat","is_sm_ips_ports","ct_state_ttl",
        "ct_flw_http_mthd","is_ftp_login","ct_ftp_cmd","ct_srv_src","ct_srv_dst",
        "ct_dst_ltm","ct_src_ltm","ct_src_dport_ltm","ct_dst_sport_ltm",
        "ct_dst_src_ltm","attack_cat","label"
    ]

    frames = []
    for path in csv_files:
        try:
            # Peek at first cell to detect if header row exists
            sample = pd.read_csv(path, nrows=1, header=None)
            first_cell = str(sample.iloc[0, 0]).strip().lower()
            # If first cell is a known col name it has a header; if IP or numeric, no header
            has_header = first_cell in ("srcip", "dur", "proto", "state")

            if has_header:
                df = pd.read_csv(path, low_memory=False)
                df.columns = [str(c).strip().lower().replace(" ", "_")
                              for c in df.columns]
            else:
                ncols = sample.shape[1]
                cols = UNSW_COLS[:ncols] if ncols <= len(UNSW_COLS) else \
                       UNSW_COLS + [f"col_{i}" for i in range(ncols - len(UNSW_COLS))]
                df = pd.read_csv(path, low_memory=False, header=None, names=cols)

            df = _downcast(df)
            frames.append(df)
        except Exception as e:
            log.warning(f"Skipping {path}: {e}")

    log.info("UNSW-NB15: concatenating...")
    combined = pd.concat(frames, ignore_index=True, copy=False)

    # Standardise label column
    label_found = False
    for candidate in ["label", "Label", "class", "attack"]:
        if candidate in combined.columns:
            if candidate != "label":
                combined.rename(columns={candidate: "label"}, inplace=True)
            label_found = True
            break

    if not label_found:
        last_col = combined.columns[-1]
        log.warning(f"Label column not found, using last column: {last_col}")
        combined.rename(columns={last_col: "label"}, inplace=True)

    combined["label"] = (
        pd.to_numeric(combined["label"], errors="coerce")
        .fillna(0).astype(int)
    )
    combined["label"] = (combined["label"] > 0).astype(np.int8)

    log.info(f"UNSW-NB15 loaded: {len(combined):,} rows, {combined.shape[1]} cols | "
             f"Attack rate: {combined['label'].mean():.2%}")
    return combined


def _load_ciciot2023() -> pd.DataFrame:
    """
    CICIoT2023: CSVs have NO label column inside them.
    Labels are inferred from the PARENT FOLDER NAME:
      - folder contains 'benign' -> label 0
      - everything else          -> label 1
    Structure: data/raw/ciciot2023/CSV/<AttackFolder>/<file>.pcap.csv
    """
    raw_dir = os.path.join(DATA_RAW, DATASETS["ciciot2023"]["raw_subdir"])
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"CICIoT2023 directory not found: {raw_dir}")

    csv_files = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")

    log.info(f"CICIoT2023: found {len(csv_files)} CSV files - loading...")

    frames = []
    for i, path in enumerate(csv_files, 1):
        # Label comes from parent folder name
        parent_folder = os.path.basename(os.path.dirname(path)).lower()
        label = 0 if "benign" in parent_folder else 1
        df = _safe_read(path, label=label)
        if df is not None:
            frames.append(df)
        if i % 20 == 0:
            log.info(f"  Loaded {i}/{len(csv_files)} files...")

    log.info("CICIoT2023: concatenating...")
    combined = pd.concat(frames, ignore_index=True, copy=False)

    log.info(f"CICIoT2023 loaded: {len(combined):,} rows, {combined.shape[1]} cols | "
             f"Attack rate: {combined['label'].mean():.2%}")
    return combined


def _load_iot23() -> pd.DataFrame:
    """
    IoT-23 (Avast/Stratosphere Lab): 23 Zeek conn.log-derived CSVs.

    IMPORTANT FORMAT QUIRK: these CSVs use a HYBRID delimiter. The first
    21 fields (ts through tunnel_parents) are comma-separated, but the
    final 3 logical fields (tunnel_parents, label, detailed-label) are
    packed into what pandas sees as ONE comma-separated column, actually
    separated internally by TABS. This is a known artifact of how Zeek's
    conn.log gets converted to CSV (Zeek's native log format uses tabs;
    only a partial conversion to commas was done upstream).

    Columns used as features: duration, orig_bytes, resp_bytes, proto,
    service, conn_state, orig_pkts, orig_ip_bytes, resp_pkts, resp_ip_bytes,
    missed_bytes, history.
    Dropped (identifiers, not features): ts, uid, id.orig_h, id.orig_p,
    id.resp_h, id.resp_p, local_orig, local_resp, tunnel_parents.
    Label: 'label' column -- 'benign' -> 0, everything else (e.g. 'Malicious')
    -> 1. The 'detailed-label' column is kept as 'attack_cat' for reference,
    matching the same convention used for UNSW/CICIoT.
    """
    raw_dir = os.path.join(DATA_RAW, DATASETS["iot23"]["raw_subdir"])
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"IoT-23 directory not found: {raw_dir}")

    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")

    log.info(f"IoT-23: found {len(csv_files)} CSV files - loading...")

    BASE_COLS = [
        "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
        "proto", "service", "duration", "orig_bytes", "resp_bytes",
        "conn_state", "local_orig", "local_resp", "missed_bytes",
        "history", "orig_pkts", "orig_ip_bytes", "resp_pkts",
        "resp_ip_bytes",
    ]
    DROP_ID_COLS = ["ts", "uid", "id.orig_h", "id.orig_p",
                   "id.resp_h", "id.resp_p", "local_orig", "local_resp"]

    # Safety cap: some IoT-23 scenario files are extremely large
    # (dataset12.csv is ~9 GB). Reading + parsing files beyond this size
    # risks running out of RAM on typical machines. We cap the number of
    # lines read per file via nrows, using a representative random-ish
    # subset (first N rows after header) rather than skipping the file
    # entirely -- this preserves some signal from large scenarios while
    # keeping memory use bounded.
    MAX_ROWS_PER_FILE = 2_000_000

    frames = []
    for i, path in enumerate(csv_files, 1):
        file_size_mb = os.path.getsize(path) / (1024 ** 2)
        log.info(f"  [{i}/{len(csv_files)}] Starting {os.path.basename(path)} "
                 f"({file_size_mb:.1f} MB)...")
        try:
            # FAST VECTORIZED PARSING using pandas string operations instead
            # of a pure-Python per-line loop (which is extremely slow on
            # large IoT-23 scenario files -- some are hundreds of MB to
            # multiple GB).
            #
            # Step 1: read the whole file as a single-column text blob
            #         (one row per line), skipping the header.
            #
            # SAMPLING NOTE: for large files we cannot read every row into
            # memory, so we cap at MAX_ROWS_PER_FILE. Using nrows= (first
            # N rows only) would systematically bias toward whatever
            # traffic appears EARLY in the capture -- IoT-23 scenarios
            # often start with a benign baseline period before malware
            # activity ramps up, so reading only the first 2M rows of an
            # 8-10 GB file risks under-representing or entirely missing
            # attack traffic that appears later. Instead we count total
            # lines first, then take a UNIFORM RANDOM sample of row
            # indices across the whole file, preserving the file's true
            # benign/attack mix regardless of where each class is
            # concentrated within the capture.
            if file_size_mb > 200:
                # Count total lines cheaply first (single pass, no parsing)
                with open(path, "rb") as fcount:
                    total_lines = sum(1 for _ in fcount) - 1  # minus header
                log.info(f"    Large file ({file_size_mb:.0f} MB, "
                         f"{total_lines:,} lines) -- "
                         f"random-sampling {MAX_ROWS_PER_FILE:,} rows")

                if total_lines > MAX_ROWS_PER_FILE:
                    rng = np.random.default_rng(RANDOM_SEED_FOR_SAMPLING)
                    keep_idx = set(
                        rng.choice(total_lines, size=MAX_ROWS_PER_FILE,
                                  replace=False).tolist()
                    )
                    # Read line-by-line, keeping only sampled indices.
                    # This is a single sequential pass (fast) rather than
                    # random-access seeking, and avoids loading the full
                    # file into a DataFrame before sampling.
                    kept_lines = []
                    with open(path, "r", encoding="utf-8",
                             errors="replace") as fh:
                        next(fh)  # skip header
                        for line_idx, line in enumerate(fh):
                            if line_idx in keep_idx:
                                kept_lines.append(line)
                    raw = pd.Series(kept_lines, name="_line").str.rstrip("\n")
                else:
                    raw = pd.read_csv(
                        path, sep="\x00", header=None, skiprows=1,
                        names=["_line"], dtype=str, engine="python",
                        on_bad_lines="skip",
                    )["_line"]
            else:
                raw = pd.read_csv(
                    path, sep="\x00", header=None, skiprows=1,
                    names=["_line"], dtype=str, engine="python",
                    on_bad_lines="skip",
                )["_line"]
            log.info(f"    Read {len(raw):,} lines, parsing (vectorized)...")

            # Step 2: split each line on comma, max 20 splits, giving us
            # the first 20 fields plus one "tail" string containing
            # tunnel_parents + label + detailed-label (whitespace-separated).
            split_df = raw.str.split(",", n=20, expand=True)

            if split_df.shape[1] < 21:
                log.warning(f"    Unexpected column count "
                           f"({split_df.shape[1]}) in {path}, skipping")
                continue

            base_part = split_df.iloc[:, :20]
            base_part.columns = BASE_COLS
            tail_part = split_df.iloc[:, 20].astype(str)

            # Step 3: extract label and detailed-label from the tail
            # (last two whitespace-separated tokens; tunnel_parents is
            # whatever remains before them, which we don't need).
            tail_split = tail_part.str.split()
            label_val = tail_split.apply(
                lambda toks: toks[-2] if isinstance(toks, list) and len(toks) >= 2
                            else (toks[-1] if isinstance(toks, list) and len(toks) == 1
                                  else "-")
            )
            detail_val = tail_split.apply(
                lambda toks: toks[-1] if isinstance(toks, list) and len(toks) >= 2
                            else "-"
            )

            df = base_part.copy()
            df["label_raw"]  = label_val.values
            df["attack_cat"] = detail_val.values
            log.info(f"    Parsed {len(df):,} records, building DataFrame...")

            # Convert numeric columns
            numeric_cols = ["duration", "orig_bytes", "resp_bytes",
                            "missed_bytes", "orig_pkts", "orig_ip_bytes",
                            "resp_pkts", "resp_ip_bytes"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop identifier columns
            df.drop(columns=[c for c in DROP_ID_COLS if c in df.columns],
                    inplace=True)

            # Encode label: 'benign' -> 0, everything else -> 1
            df["label_raw"] = df["label_raw"].astype(str).str.strip().str.lower()
            df["label"] = (~df["label_raw"].isin(
                ["benign", "-", "nan", "none", ""]
            )).astype(np.int8)
            df.drop(columns=["label_raw"], inplace=True)

            df.columns = [c.strip().lower().replace(" ", "_").replace(".", "_")
                          for c in df.columns]
            df = _downcast(df)
            frames.append(df)
        except Exception as e:
            log.warning(f"Skipping {path}: {e}")

        if i % 5 == 0:
            log.info(f"  Loaded {i}/{len(csv_files)} files...")

    log.info("IoT-23: concatenating...")
    combined = pd.concat(frames, ignore_index=True, copy=False)

    log.info(f"IoT-23 loaded: {len(combined):,} rows, {combined.shape[1]} cols | "
             f"Attack rate: {combined['label'].mean():.2%}")
    return combined


# --------------------------------------------------------------
# PUBLIC ENTRY-POINT
# --------------------------------------------------------------

_LOADERS = {
    "nbaiot":     _load_nbaiot,
    "unsw":       _load_unsw,
    "ciciot2023": _load_ciciot2023,
    "iot23":      _load_iot23,
}


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a raw dataset by name.

    Parameters
    ----------
    name : str  - one of 'nbaiot', 'ciciot2023', 'iot23'

    Returns
    -------
    pd.DataFrame with lowercase columns and a binary 'label' column.
    """
    name = name.lower()
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(_LOADERS)}"
        )
    log.info(f"Loading dataset: {name}")
    return _LOADERS[name]()
