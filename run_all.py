"""
run_all.py -- resumable runner for the full revised pipeline.

    python run_all.py            run (or continue) everything
    python run_all.py --status   show which steps are done

Each step runs `python main.py --dataset X --stage Y` in a fresh process.
When a step finishes successfully a marker file is written to
outputs_revision/run_state/. After a power cut, just run the same command
again: finished steps are skipped and the interrupted step restarts.
Inside the "train" step, each finished classifier is saved and re-loaded,
so a power cut there loses at most the model that was being trained.

To redo a step, delete its marker file (e.g. run_state/iot23_train.done)
and the step's outputs.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from configs.config import OUT_STATE, OUT_LOGS  # noqa: E402

DATASETS = ["iot23", "ciciot2023", "nbaiot"]   # smallest first

STEPS = (
    [(ds, "preprocess") for ds in DATASETS]
    + [(ds, "train") for ds in DATASETS]
    + [(ds, "evaluate") for ds in DATASETS]
    + [("all", "cross")]
    + [(ds, "shap") for ds in DATASETS]
    + [(ds, "stats") for ds in DATASETS]
)


def marker(ds, stage):
    return os.path.join(OUT_STATE, f"{ds}_{stage}.done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_STATE, exist_ok=True)
    os.makedirs(OUT_LOGS, exist_ok=True)

    if args.status:
        for ds, st in STEPS:
            print(f"  [{'x' if os.path.exists(marker(ds, st)) else ' '}] {ds:11s} {st}")
        return

    t0 = time.time()
    for i, (ds, st) in enumerate(STEPS, 1):
        if os.path.exists(marker(ds, st)):
            print(f"[{i}/{len(STEPS)}] {ds} {st}: already done, skipping")
            continue
        print(f"\n[{i}/{len(STEPS)}] {datetime.now():%H:%M:%S}  {ds} {st} ...", flush=True)
        s = time.time()
        rc = subprocess.call([sys.executable, "main.py", "--dataset", ds,
                              "--stage", st], cwd=HERE)
        if rc != 0:
            print(f"\nSTEP FAILED: {ds} {st} (exit code {rc}).\n"
                  f"See {os.path.join(OUT_LOGS, 'project.log')}. Fix the issue "
                  f"and run `python run_all.py` again to continue from here.")
            sys.exit(rc)
        with open(marker(ds, st), "w") as f:
            f.write(f"{datetime.now().isoformat()} {time.time() - s:.0f}s\n")
        print(f"[{i}/{len(STEPS)}] {ds} {st}: done in {(time.time() - s) / 60:.1f} min",
              flush=True)

    print(f"\nALL STEPS COMPLETE in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
