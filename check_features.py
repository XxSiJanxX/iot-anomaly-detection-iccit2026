import joblib

pre = joblib.load(r"D:\iot_anomaly_detection\outputs\models\ciciot2023_preprocess_checkpoint.pkl")
print("CICIoT2023 pre-selection features:", pre["pre_feature_names"])

eng = joblib.load(r"D:\iot_anomaly_detection\outputs\models\ciciot2023_engineer.pkl")
print("\nDropped by correlation filter:", eng._dropped_corr)

pre23 = joblib.load(r"D:\iot_anomaly_detection\outputs\models\iot23_preprocess_checkpoint.pkl")
print("\nIoT-23 pre-selection features:", pre23["pre_feature_names"])
