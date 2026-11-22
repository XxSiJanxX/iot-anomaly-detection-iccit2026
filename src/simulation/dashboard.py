"""
dashboard.py
------------
Streamlit real-time anomaly detection dashboard.

Run with:
    streamlit run src/simulation/dashboard.py

Features:
  - Dataset selector (nbaiot / unsw / ciciot2023)
  - Model selector (best classifier / VAE / hybrid)
  - Live detection feed (records streaming one-by-one)
  - Running accuracy, precision, recall, F1
  - Alert log table (flagged attack records)
  - Confidence score sparkline
  - Replay / pause controls
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from configs.config import OUT_MODELS, DATASET_NAMES, SIM_SLEEP_MS
from src.simulation.stream_pipeline import StreamSimulator
from utils.logger import get_logger

log = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT Anomaly Detection — Live Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ IoT Anomaly Detection — Real-Time Simulation")
st.markdown("*Hybrid VAE + Supervised ML | ICCIT 2025 Research Demo*")

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    dataset_choice = st.selectbox("Dataset", DATASET_NAMES)

    # Scoped to "Best Supervised" only. The VAE / Hybrid options need a
    # second, separate VAE-space feature array alongside the supervised
    # one -- stream_pipeline.py's single-array interface has no way to
    # supply that, so those three options would either crash mid-stream
    # or silently score the wrong feature space. Rather than leave a
    # dropdown option that fails live, the demo is scoped to the option
    # that is fully correct: the best-performing supervised classifier
    # per dataset. See HANDOFF/paper notes for the underlying reason.
    model_type = st.selectbox(
        "Model",
        ["Best Supervised"]
    )

    speed_ms = st.slider("Stream speed (ms / record)", 10, 500, SIM_SLEEP_MS, 10)
    n_records = st.slider("Records to stream", 100, 5000, 500, 100)

    start_btn  = st.button("▶ Start Simulation", use_container_width=True)
    reset_btn  = st.button("⟳ Reset",            use_container_width=True)

    st.divider()
    st.caption("Adjust speed before starting.")

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(dataset: str, mtype: str):
    """Load the appropriate model from disk."""
    model_map = {
        "Best Supervised": f"{dataset}_best_supervised.pkl",
        "VAE":             None,   # VAE loaded separately
        "Hybrid (Weighted)": f"{dataset}_hybrid_weighted.pkl",
        "Hybrid (OR)":       f"{dataset}_hybrid_or.pkl",
    }

    if mtype == "VAE":
        from src.models.vae import TrainedVAE
        vae_dir = os.path.join(OUT_MODELS, f"{dataset}_VAE")
        if not os.path.isdir(vae_dir):
            return None, f"VAE model not found at {vae_dir}"
        # Need input_dim — try to read from meta
        meta_path = os.path.join(vae_dir, "vae_meta.pkl")
        if not os.path.exists(meta_path):
            return None, "VAE meta file not found"
        meta = joblib.load(meta_path)
        vae  = TrainedVAE.load(vae_dir, meta["input_dim"])
        return vae, None
    else:
        fname = model_map.get(mtype)
        path  = os.path.join(OUT_MODELS, fname) if fname else None
        if path is None or not os.path.exists(path):
            return None, f"Model file not found: {path}"
        return joblib.load(path), None


@st.cache_resource(show_spinner="Loading test data…")
def load_test_data(dataset: str):
    """Load pre-processed test arrays saved during main pipeline run."""
    arrays_path = os.path.join(OUT_MODELS, f"{dataset}_test_arrays.pkl")
    if not os.path.exists(arrays_path):
        return None, None, None, f"Test arrays not found: {arrays_path}"
    data = joblib.load(arrays_path)
    return data["X_test"], data["y_test"], data["feature_names"], None


# ──────────────────────────────────────────────────────────────────────────────
# LAYOUT PLACEHOLDERS
# ──────────────────────────────────────────────────────────────────────────────
col_metrics, col_feed = st.columns([1, 2])

with col_metrics:
    st.subheader("📊 Live Metrics")
    metric_accuracy  = st.empty()
    metric_f1        = st.empty()
    metric_recall    = st.empty()
    metric_alerts    = st.empty()
    st.divider()
    st.subheader("⚠️ Alert Log")
    alert_table      = st.empty()

with col_feed:
    st.subheader("📡 Live Detection Feed")
    feed_table       = st.empty()
    st.divider()
    st.subheader("📈 Confidence Score History")
    score_chart      = st.empty()

# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION LOOP
# ──────────────────────────────────────────────────────────────────────────────
if reset_btn:
    st.rerun()

if start_btn:
    # Load model and data
    model, model_err = load_model(dataset_choice, model_type)
    X_test, y_test, feature_names, data_err = load_test_data(dataset_choice)

    if model_err:
        st.error(f"Model load error: {model_err}")
        st.stop()
    if data_err:
        st.error(f"Data load error: {data_err}")
        st.stop()

    # Subsample for demo
    X_demo = X_test[:n_records]
    y_demo = y_test[:n_records]

    simulator = StreamSimulator(
        model=model,
        X_test=X_demo,
        y_test=y_demo,
        feature_names=feature_names or [],
        sleep_ms=speed_ms,
    )

    all_records   = []
    alert_records = []
    confidence_history = []
    y_true_so_far = []
    y_pred_so_far = []

    progress_bar = st.progress(0, text="Streaming…")

    for record in simulator.stream():
        all_records.append(record)
        confidence_history.append(record["confidence"])
        y_true_so_far.append(record["true_label"])
        y_pred_so_far.append(record["prediction"])

        if record["is_alert"]:
            alert_records.append(record)

        # Update every 10 records
        if len(all_records) % 10 == 0 or len(all_records) == n_records:
            yt = np.array(y_true_so_far)
            yp = np.array(y_pred_so_far)

            acc = accuracy_score(yt, yp)
            f1  = f1_score(yt, yp, zero_division=0)
            rec = recall_score(yt, yp, zero_division=0)

            metric_accuracy.metric("Accuracy",  f"{acc:.3f}")
            metric_f1.metric("F1 Score",         f"{f1:.3f}")
            metric_recall.metric("Recall",        f"{rec:.3f}")
            metric_alerts.metric("Total Alerts",  len(alert_records))

            # Feed table (last 20)
            feed_df = pd.DataFrame(all_records[-20:])[
                ["timestamp", "prediction", "confidence", "true_label", "correct"]
            ]
            feed_df["prediction"] = feed_df["prediction"].map({0: "✅ Normal", 1: "🚨 Attack"})
            feed_df["correct"]    = feed_df["correct"].map({True: "✔", False: "✘"})
            feed_table.dataframe(feed_df, use_container_width=True)

            # Alert log (last 10)
            if alert_records:
                alerts_df = pd.DataFrame(alert_records[-10:])[
                    ["timestamp", "confidence", "true_label"]
                ]
                alert_table.dataframe(alerts_df, use_container_width=True)

            # Confidence sparkline
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.plot(confidence_history[-200:], color="firebrick", lw=1)
            ax.axhline(0.5, color="grey", linestyle="--", lw=0.8)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Record index")
            ax.set_ylabel("Score")
            ax.set_title("Confidence Score (last 200 records)")
            fig.tight_layout()
            score_chart.pyplot(fig)
            plt.close(fig)

            progress_bar.progress(len(all_records) / n_records,
                                   text=f"Processed {len(all_records)} / {n_records} records")

    simulator.save_alert_log()
    progress_bar.progress(1.0, text="✅ Simulation complete")
    st.success(
        f"Done! {n_records} records processed | "
        f"{len(alert_records)} alerts | "
        f"Alert rate: {len(alert_records)/n_records:.1%}"
    )
