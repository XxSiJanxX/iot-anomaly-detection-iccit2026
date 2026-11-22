import joblib
ckpt = joblib.load(r"D:\iot_anomaly_detection\outputs\models\ciciot2023_preprocess_checkpoint.pkl")
idx = ckpt["pre_feature_names"].index("number")
vals = ckpt["X_train_pre"][:, idx]
print(f"min={vals.min()}, max={vals.max()}, are_all_integers={(vals == vals.round()).all()}")