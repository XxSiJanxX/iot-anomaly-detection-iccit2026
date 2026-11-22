# Hybrid IoT Anomaly Detection: Cross-Dataset Generalization, VAE and Explainable AI

Code and result files for the IEEE ICCIT 2026 paper:

> T. I. Sijan, M. M. Arpita, and S. Saha, "Hybrid IoT Anomaly Detection with Cross-Dataset Generalization, Variational Autoencoder, and Explainable AI," *Proc. ICCIT 2026*.

The pipeline compares five supervised classifiers (Random Forest, XGBoost, Decision Tree, LinearSVC, k-NN), a VAE anomaly detector and two hybrid ensembles. It runs on three IoT datasets (N-BaIoT, CICIoT2023, IoT-23) under a leakage-controlled protocol:

- Duplicates are removed on the full dataset **before** the train/validation/test split.
- SMOTE is refitted inside every cross-validation fold.
- Feature selection and scaling are fitted on training data only.
- Every test metric carries a bootstrap 95% confidence interval.

## Main results (test set, MCC)

| | N-BaIoT | CICIoT2023 | IoT-23 |
|---|---|---|---|
| XGBoost (best supervised) | 0.997 | 0.818 | 0.948 |
| VAE (calibrated threshold) | 0.942 | 0.690 | 0.096 |
| Hybrid (weighted) | 0.997 | 0.821 | 0.951 |
| XGBoost, tuned threshold | 0.998 | 0.822 | 0.951 |

Full results, including confidence intervals, paired tests, cross-dataset transfer and significance tests, are in [`outputs_revision/results/`](outputs_revision/results/). [`REVISION_CHANGES.md`](REVISION_CHANGES.md) lists every methodological change made for the camera-ready version.

## Data

The datasets are public and not included in this repository. Download them and place them as follows:

```
data/raw/
├── nbaiot/       one folder per device, each with benign/, gafgyt_attacks/, mirai_attacks/ CSVs
├── ciciot2023/
│   └── CSV/      one folder per traffic class (Benign_Final/, DDoS-SYN_Flood/, ...)
└── iot23/        dataset1.csv ... dataset23.csv (Zeek conn.log, labelled)
```

- N-BaIoT: <https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot>
- CICIoT2023: <https://www.unb.ca/cic/datasets/iotdataset-2023.html>
- IoT-23: <https://www.stratosphereips.org/datasets-iot23>

## Setup

Tested with Python 3.11.9 on Windows 11 (CPU only, 16 GB RAM).

```
pip install -r requirements.txt
```

`requirements-lock.txt` pins the exact library versions used for the reported results.

## Running

```
python run_all.py            # full pipeline, about 4 hours on the hardware above
python run_all.py --status   # which steps are finished
```

`run_all.py` runs preprocessing, training, evaluation, cross-dataset transfer, SHAP and significance tests in order. It records each finished step, so after an interruption you can run it again and it continues where it stopped. Single stages can also be run directly:

```
python main.py --dataset iot23 --stage preprocess
python main.py --dataset all --stage cross
```

Stages: `preprocess`, `train`, `evaluate`, `cross`, `shap`, `stats`. All settings live in `configs/config.py`. Outputs go to `outputs_revision/` (override with the `IOT_OUTPUT_DIR` environment variable).

## Repository layout

```
configs/config.py            all hyperparameters, seeds and paths
src/preprocessing/           loading, de-duplication before splitting, cleaning, feature engineering, cross-dataset mapping
src/models/                  supervised classifiers (SMOTE inside CV), VAE, hybrid ensembles
src/evaluation/              metrics + bootstrap CIs, ablation, transfer, significance tests, plots
src/explainability/          SHAP analysis
main.py, run_all.py          pipeline entry points
check_overlap.py             measures train/test overlap in saved splits
outputs_revision/results/    all result CSVs and per-dataset data summaries
outputs_revision/plots/      all figures
```

## Reproducibility

- Split seed: 42. Significance-test seeds: 42, 7, 123, 2024, 31337.
- `outputs_revision/results/run_info.json` records the hardware and library versions of the reported run.
- `outputs_revision/results/<dataset>_data_summary.json` records the row counts, duplicates removed and label conflicts for each dataset.
