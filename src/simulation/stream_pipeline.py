"""
stream_pipeline.py
------------------
Simulates real-time IoT traffic by streaming test records one-by-one
through the best hybrid model.

Yields per-record dicts with:
  - timestamp
  - prediction (0/1)
  - confidence score
  - true label (for live accuracy tracking)
  - feature values (for display)
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime
from utils.logger import get_logger
from configs.config import SIM_BATCH_SIZE, SIM_SLEEP_MS, SIM_ALERT_LOG
import os

log = get_logger(__name__)


class StreamSimulator:
    """
    Streams test records through a trained model in real time.

    Parameters
    ----------
    model        : any model with .predict() and .predict_proba()
    X_test       : numpy array of scaled test features
    y_test       : true labels (for accuracy tracking)
    feature_names: list of feature name strings
    sleep_ms     : pause between records in ms (controls speed)
    """

    def __init__(self, model, X_test: np.ndarray, y_test: np.ndarray,
                 feature_names: list, sleep_ms: int = SIM_SLEEP_MS):
        self.model         = model
        self.X_test        = X_test
        self.y_test        = y_test
        self.feature_names = feature_names
        self.sleep_ms      = sleep_ms
        self._alerts       = []

    def stream(self):
        """
        Generator: yields one result dict per record.
        Use in a for-loop or collect into a list.
        """
        for i in range(len(self.X_test)):
            x = self.X_test[i:i+1]
            y_true = int(self.y_test[i])

            # Predict
            pred = int(self.model.predict(x)[0])

            # Confidence score
            if hasattr(self.model, "predict_proba"):
                score = float(self.model.predict_proba(x)[0])
            elif hasattr(self.model, "anomaly_score"):
                score = float(self.model.anomaly_score(x)[0])
            else:
                score = float(pred)

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            record = {
                "index":      i,
                "timestamp":  ts,
                "prediction": pred,
                "confidence": round(score, 4),
                "true_label": y_true,
                "is_alert":   pred == 1,
                "correct":    pred == y_true,
            }

            # Attach top-5 feature values for display
            for j, fname in enumerate(self.feature_names[:5]):
                record[fname] = round(float(x[0, j]), 4)

            if pred == 1:
                self._alerts.append(record)

            time.sleep(self.sleep_ms / 1000.0)
            yield record

    def save_alert_log(self):
        """Persist all alerts to CSV."""
        if not self._alerts:
            log.info("No alerts to save.")
            return
        os.makedirs(os.path.dirname(SIM_ALERT_LOG), exist_ok=True)
        df = pd.DataFrame(self._alerts)
        df.to_csv(SIM_ALERT_LOG, index=False)
        log.info(f"Alert log saved → {SIM_ALERT_LOG} ({len(df)} alerts)")

    def summary(self) -> dict:
        """Return cumulative stats after streaming."""
        total   = len(self.X_test)
        alerts  = len(self._alerts)
        return {
            "total_records": total,
            "total_alerts":  alerts,
            "alert_rate":    round(alerts / total, 4) if total > 0 else 0,
        }
