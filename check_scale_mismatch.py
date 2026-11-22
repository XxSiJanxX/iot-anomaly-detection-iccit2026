import joblib
import numpy as np

ciciot = joblib.load(r"D:\iot_anomaly_detection\outputs\models\ciciot2023_preprocess_checkpoint.pkl")
iot23 = joblib.load(r"D:\iot_anomaly_detection\outputs\models\iot23_preprocess_checkpoint.pkl")

def stats(name, arr):
    print(f"{name}: min={arr.min():.2f}  max={arr.max():.2f}  "
          f"median={np.median(arr):.2f}  mean={arr.mean():.2f}  "
          f"p95={np.percentile(arr, 95):.2f}  p99={np.percentile(arr, 99):.2f}")

ciciot_idx = ciciot["pre_feature_names"].index("tot_sum")
ciciot_tot_sum = ciciot["X_train_pre"][:, ciciot_idx]
stats("CICIoT2023 tot_sum (raw, pre-scaling)", ciciot_tot_sum)

iot23_ob_idx = iot23["pre_feature_names"].index("orig_bytes")
iot23_rb_idx = iot23["pre_feature_names"].index("resp_bytes")
iot23_total_bytes = iot23["X_train_pre"][:, iot23_ob_idx] + iot23["X_train_pre"][:, iot23_rb_idx]
stats("IoT-23 orig_bytes+resp_bytes (raw, pre-scaling)", iot23_total_bytes)

# Simulate what CrossDatasetPreparer actually does: fit MinMax on source,
# apply to target, and see what fraction of target values get clipped.
from sklearn.preprocessing import MinMaxScaler

print("\n--- Simulating ciciot2023 -> iot23 direction ---")
scaler = MinMaxScaler()
scaler.fit(ciciot_tot_sum.reshape(-1, 1))
scaled_tgt = scaler.transform(iot23_total_bytes.reshape(-1, 1)).ravel()
stats("iot23 total_bytes AFTER scaling with ciciot2023's fitted scaler", scaled_tgt)
pct_at_1 = (scaled_tgt >= 0.999).mean() * 100
pct_at_0 = (scaled_tgt <= 0.001).mean() * 100
print(f"% of target values clipped near 1.0: {pct_at_1:.1f}%   near 0.0: {pct_at_0:.1f}%")

print("\n--- Simulating iot23 -> ciciot2023 direction ---")
scaler2 = MinMaxScaler()
scaler2.fit(iot23_total_bytes.reshape(-1, 1))
scaled_tgt2 = scaler2.transform(ciciot_tot_sum.reshape(-1, 1)).ravel()
stats("ciciot2023 tot_sum AFTER scaling with iot23's fitted scaler", scaled_tgt2)
pct_at_1b = (scaled_tgt2 >= 0.999).mean() * 100
pct_at_0b = (scaled_tgt2 <= 0.001).mean() * 100
print(f"% of target values clipped near 1.0: {pct_at_1b:.1f}%   near 0.0: {pct_at_0b:.1f}%")
