"""
cross_dataset.py
----------------
Prepares data for cross-dataset generalization experiments.

DATASET COMBINATION: N-BaIoT + CICIoT2023 + IoT-23 (all genuinely IoT-focused).
UNSW-NB15 was deliberately excluded -- it is a general enterprise network
intrusion dataset (synthetic IXIA PerfectStorm traffic), not IoT-specific,
and shares almost no feature overlap with the other two regardless of
canonical mapping effort. IoT-23 (Avast/Stratosphere Lab) replaces it:
it is flow-based (Zeek conn.log derived) like CICIoT2023, genuinely IoT
malware traffic, and shares real semantic feature overlap with CICIoT2023
(duration, byte counts, packet counts, protocol) -- giving a much better
basis for genuine cross-dataset generalization experiments.

APPROACH:
  N-BaIoT uses statistical packet-aggregate features (MI_dir_L5_weight etc.)
  with no direct semantic equivalent in flow-based datasets -- cross-dataset
  pairs involving N-BaIoT are expected to show 0 or near-0 common features,
  which we report explicitly as a documented limitation (consistent with
  established literature on this dataset's feature methodology), not as a
  novel discovery.

  CICIoT2023 and IoT-23 are both flow/Zeek-derived, so we expect genuine,
  non-trivial feature overlap (duration, byte/packet counts, protocol) --
  this is the pair where real cross-dataset generalization can be tested.

  For each (src, tgt) pair with N common features:
    1. Extract the common feature columns from each dataset's pre-selection
       arrays (stored in the preprocessing checkpoint)
    2. Train a simple RF on the source common features
    3. Predict on target common features
    4. Report F1, MCC
"""

import os
import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# SEMANTIC FEATURE MAPPING
# Maps raw column names to canonical names for cross-dataset alignment.
# Only features with clear semantic equivalence across datasets are mapped.
# ──────────────────────────────────────────────────────────────────────────────

# CICIoT2023 raw → canonical
CICIOT_CANONICAL = {
    "flow_duration":    "flow_duration",
    "rate":             "pkt_rate",
    "fin_flag_number":  "fin_flag",
    "syn_flag_number":  "syn_flag",
    "rst_flag_number":  "rst_flag",
    "psh_flag_number":  "psh_flag",
    "ack_flag_number":  "ack_flag",
    "protocol_type":    "protocol",
    "header_length":    "header_length",
    "time_to_live":     "src_ttl",
    "ack_count":        "ack_count",
    "syn_count":        "syn_count",
    "fin_count":        "fin_count",
    "rst_count":        "rst_count",
    "tot_sum":          "total_bytes",   # total bytes in the flow, pre-summed
}

# IoT-23 raw → canonical (Zeek conn.log fields)
# These map naturally onto the same canonical names used for CICIoT2023's
# flow-level features, since both ultimately describe network flow
# characteristics (duration, byte/packet counts, protocol).
IOT23_CANONICAL = {
    "duration":      "flow_duration",
    "orig_bytes":    "src_bytes",
    "resp_bytes":    "dst_bytes",
    "orig_pkts":     "src_pkts",
    "resp_pkts":     "dst_pkts",
    "orig_ip_bytes": "src_bytes",      # alt granularity, same semantic family
    "resp_ip_bytes": "dst_bytes",
    "proto":         "protocol",
    "missed_bytes":  "missed_bytes",
}

# N-BaIoT: statistical packet features — no direct semantic equivalent
# in flow-based datasets (CICIoT2023, IoT-23). We report 0 common features
# and treat this as a documented limitation consistent with the established
# literature on N-BaIoT's distinct feature extraction methodology, not as
# a novel discovery.
NBAIOT_CANONICAL = {}   # intentionally empty — no mappable features


# ──────────────────────────────────────────────────────────────────────────────
# DERIVED (multi-column) CANONICAL FEATURES
# Some canonical concepts exist in one dataset only as a COMBINATION of raw
# columns, not a single renamed column -- e.g. IoT-23 (Zeek conn.log) reports
# flow bytes split by direction (orig_bytes, resp_bytes), while CICIoT2023
# reports the same underlying quantity pre-summed as a single "tot_sum"
# column (mapped directly above). Each entry here:
#   canonical_name -> (list of raw columns required, combine function)
# combine_fn takes the raw columns' values as separate positional 1-D arrays,
# in the same order as the column list, and returns the combined 1-D array.
# A derived entry is only used if the simpler 1:1 canonical map didn't
# already supply that canonical name, and only if ALL of its required raw
# columns are actually present in the dataset's feature list.
# ──────────────────────────────────────────────────────────────────────────────

CICIOT_DERIVED = {}   # tot_sum already covers this directly, see CICIOT_CANONICAL

# REVISION: CICIoT2023's tot_sum is the sum of PACKET lengths (headers
# included), so the closest Zeek quantity is the IP-level byte count
# orig_ip_bytes + resp_ip_bytes; payload-only orig_bytes + resp_bytes is the
# fallback. Each entry may list alternatives, tried in order.
# NOTE the remaining, unavoidable mismatch (documented in the paper):
# tot_sum is computed over a fixed packet window, Zeek counts over a whole
# connection.
def _add(a, b):
    return a + b


IOT23_DERIVED = {
    "total_bytes": [(["orig_ip_bytes", "resp_ip_bytes"], _add),
                    (["orig_bytes", "resp_bytes"], _add)],
}

NBAIOT_DERIVED = {}


def _get_canonical_map(dataset_name: str) -> dict:
    maps = {
        "nbaiot":     NBAIOT_CANONICAL,
        "ciciot2023": CICIOT_CANONICAL,
        "iot23":      IOT23_CANONICAL,
    }
    return maps.get(dataset_name, {})


def _get_derived_map(dataset_name: str) -> dict:
    maps = {
        "nbaiot":     NBAIOT_DERIVED,
        "ciciot2023": CICIOT_DERIVED,
        "iot23":      IOT23_DERIVED,
    }
    return maps.get(dataset_name, {})


def _canonical_candidates(dataset_name: str, features: list) -> dict:
    """
    Build canonical_name -> spec for every canonical concept reachable given
    the features actually present in this dataset's pre-selection pool.

    spec is one of:
      ("simple",  [raw_col],            None)
      ("derived", [raw_col1, raw_col2, ...], combine_fn)
    """
    features_set = set(features)
    out = {}

    # Simple 1:1 entries take priority
    for raw, can in _get_canonical_map(dataset_name).items():
        if raw in features_set and can not in out:
            out[can] = ("simple", [raw], None)

    # Derived entries fill in canonical concepts the simple map couldn't reach
    for can, spec in _get_derived_map(dataset_name).items():
        if can in out:
            continue
        alternatives = spec if isinstance(spec, list) else [spec]
        for raw_cols, fn in alternatives:
            if all(c in features_set for c in raw_cols):
                out[can] = ("derived", raw_cols, fn)
                break

    return out


def get_common_features(
    src_features: list,
    src_dataset: str,
    tgt_features: list,
    tgt_dataset: str,
) -> list:
    """
    Find common features between two datasets via canonical name mapping,
    including derived (multi-column) features where one side needs several
    raw columns combined to reach the same canonical concept the other side
    has as a single column. Falls back to direct name intersection if
    canonical mapping yields nothing.

    Returns a list of match dicts, each:
      {
        "canonical":   canonical name (for reporting),
        "src_kind":    "simple" | "derived",
        "src_cols":    [raw column name(s) in the source dataset],
        "src_combine": None or combine_fn for the source side,
        "tgt_kind":    "simple" | "derived",
        "tgt_cols":    [raw column name(s) in the target dataset],
        "tgt_combine": None or combine_fn for the target side,
      }
    """
    src_candidates = _canonical_candidates(src_dataset, src_features)
    tgt_candidates = _canonical_candidates(tgt_dataset, tgt_features)

    matches = []
    for can, (src_kind, src_cols, src_combine) in src_candidates.items():
        tgt_spec = tgt_candidates.get(can)
        if tgt_spec is None:
            continue
        tgt_kind, tgt_cols, tgt_combine = tgt_spec
        matches.append({
            "canonical":   can,
            "src_kind":    src_kind, "src_cols": src_cols, "src_combine": src_combine,
            "tgt_kind":    tgt_kind, "tgt_cols": tgt_cols, "tgt_combine": tgt_combine,
        })

    # Direct name fallback (only if canonical mapping found nothing)
    if not matches:
        direct = sorted(set(src_features) & set(tgt_features))
        if direct:
            for col in direct:
                matches.append({
                    "canonical":   col,
                    "src_kind": "simple", "src_cols": [col], "src_combine": None,
                    "tgt_kind": "simple", "tgt_cols": [col], "tgt_combine": None,
                })
            log.info(f"[{src_dataset}->{tgt_dataset}] "
                     f"Using {len(direct)} direct name matches")

    log.info(f"Cross-dataset [{src_dataset}->{tgt_dataset}]: "
             f"{len(matches)} common features "
             f"(src: {len(src_features)}, tgt: {len(tgt_features)})")
    return matches


class CrossDatasetPreparer:
    """
    Aligns two datasets for cross-generalization experiments.
    Uses the FULL pre-selection feature arrays (not post-RF-selection)
    so that the canonical map has more features to work with.
    """

    def __init__(self, src_name: str, tgt_name: str):
        self.src_name = src_name
        self.tgt_name = tgt_name

    def prepare(
        self,
        src_pipeline_output: dict,
        tgt_df_clean: pd.DataFrame,
        tgt_feature_names: list,
        src_test_df: pd.DataFrame = None,
    ) -> dict:
        """
        Build cross-dataset aligned arrays.

        src_pipeline_output must contain:
          - "feature_names"     : post-selection feature names (for reference)
          - "X_train"           : post-selection training array
          - "y_train"           : training labels
          - "pre_feature_names" : FULL feature names before RF selection (optional)
          - "X_train_pre"       : FULL training array before RF selection (optional)

        If pre_* keys are not present, falls back to post-selection arrays.
        """
        # Prefer pre-selection feature names for better cross-dataset matching
        if "pre_feature_names" in src_pipeline_output:
            src_features = src_pipeline_output["pre_feature_names"]
            X_src_full   = src_pipeline_output["X_train_pre"]
            log.info(f"[{self.src_name}] Using pre-selection features "
                     f"({len(src_features)} features)")
        else:
            src_features = src_pipeline_output["feature_names"]
            X_src_full   = src_pipeline_output["X_train"]
            log.info(f"[{self.src_name}] Using post-selection features "
                     f"({len(src_features)} features) — no pre-selection saved")

        matches = get_common_features(
            src_features, self.src_name,
            tgt_feature_names, self.tgt_name,
        )

        if not matches:
            log.warning(
                f"No common features for [{self.src_name}->{self.tgt_name}]. "
                f"This is a primary finding: complete domain shift between "
                f"{self.src_name} (statistical features) and "
                f"{self.tgt_name} (flow features). "
                f"Reporting as 0-feature cross-dataset result."
            )
            return {
                "skipped":   True,
                "reason":    "no_common_features",
                "src_total": len(src_features),
                "tgt_total": len(tgt_feature_names),
                "n_common":  0,
                "src_dataset": self.src_name,
                "tgt_dataset": self.tgt_name,
            }

        src_idx_of = {c: i for i, c in enumerate(src_features)}

        def _build(kind, cols, combine, *, from_array=None, from_df=None):
            """Build one common-feature column, simple or derived, from
            either a numpy array (source, indexed by src_idx_of) or a
            DataFrame (target, indexed by column name directly)."""
            if from_array is not None:
                parts = [from_array[:, src_idx_of[c]] for c in cols]
            else:
                parts = [from_df[c].values for c in cols]
            if kind == "simple":
                return parts[0]
            return combine(*parts)

        src_columns, tgt_columns, canonical_names = [], [], []
        for m in matches:
            src_columns.append(_build(
                m["src_kind"], m["src_cols"], m["src_combine"],
                from_array=X_src_full))
            tgt_columns.append(_build(
                m["tgt_kind"], m["tgt_cols"], m["tgt_combine"],
                from_df=tgt_df_clean))
            canonical_names.append(m["canonical"])

        X_src = np.column_stack(src_columns).astype(float)
        X_tgt_raw = np.column_stack(tgt_columns).astype(float)

        # Source's own held-out test set on the same common features --
        # used for the within-dataset 2-feature CONTROL (Reviewer 3, c. 4)
        X_src_test_raw, y_src_test = None, None
        if src_test_df is not None:
            X_src_test_raw = np.column_stack([
                _build(m["src_kind"], m["src_cols"], m["src_combine"],
                       from_df=src_test_df) for m in matches]).astype(float)
            y_src_test = src_test_df["label"].values

        # Scale target with source statistics (min-max using source training data)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(X_src)
        X_tgt = scaler.transform(X_tgt_raw)
        X_src = scaler.transform(X_src)
        X_src_test = (scaler.transform(X_src_test_raw)
                      if X_src_test_raw is not None else None)

        y_tgt = tgt_df_clean["label"].values
        y_src = src_pipeline_output.get("y_train_pre", src_pipeline_output["y_train"])

        log.info(f"[{self.src_name}->{self.tgt_name}] "
                 f"Common features: {len(canonical_names)} | "
                 f"src shape: {X_src.shape} | tgt shape: {X_tgt.shape}")

        return {
            "skipped":        False,
            "X_src_common":   X_src,
            "y_src":          y_src,
            "y_train":        y_src,
            "X_tgt_common":   X_tgt,
            "y_tgt":          y_tgt,
            "X_src_test_common": X_src_test,
            "y_src_test":     y_src_test,
            "mapping": [{"canonical": m["canonical"],
                         "src_cols": m["src_cols"], "tgt_cols": m["tgt_cols"],
                         "src_kind": m["src_kind"], "tgt_kind": m["tgt_kind"]}
                        for m in matches],
            "common_canonical": canonical_names,
            "src_cols":       [",".join(m["src_cols"]) for m in matches],
            "tgt_cols":       [",".join(m["tgt_cols"]) for m in matches],
            "n_common":       len(canonical_names),
            "src_total":      len(src_features),
            "tgt_total":      len(tgt_feature_names),
        }
