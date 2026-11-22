"""Records hardware + library versions for the paper's reproducibility
section (Reviewer 3, comments 7 and 10). Written to OUT_RESULTS/run_info.json."""
import json
import os
import platform
import sys
from datetime import datetime

from configs.config import OUT_RESULTS, RANDOM_SEED


def save_run_info():
    info = {"timestamp": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "random_seed": RANDOM_SEED}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    for mod in ("numpy", "pandas", "sklearn", "imblearn", "xgboost",
                "tensorflow", "shap", "scipy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import tensorflow as tf
        info["tf_gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        pass
    os.makedirs(OUT_RESULTS, exist_ok=True)
    with open(os.path.join(OUT_RESULTS, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info
