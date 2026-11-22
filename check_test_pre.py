import joblib

ckpt = joblib.load(r"D:\iot_anomaly_detection\outputs\models\nbaiot_preprocess_checkpoint.pkl")

print("Keys in checkpoint:", sorted(ckpt.keys()))
print()
if "X_test_pre" in ckpt:
    print(f"X_test_pre IS present. Shape: {ckpt['X_test_pre'].shape}")
    print(f"pre_feature_names count: {len(ckpt['pre_feature_names'])}")
    print(f"X_test_pre column count matches pre_feature_names: "
          f"{ckpt['X_test_pre'].shape[1] == len(ckpt['pre_feature_names'])}")
else:
    print("X_test_pre is MISSING -- the fix did not take effect on this checkpoint.")
