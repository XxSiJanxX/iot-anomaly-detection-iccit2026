# Camera-ready revision: code changes

All results of the revised pipeline go to `outputs_revision/`. The original
`outputs/` folder is not touched.

## How to run

```
cd /d D:\iot_anomaly_detection
python run_all.py            # runs everything; run it again after a power cut
python run_all.py --status   # shows which steps are done
```

`run_all.py` runs 16 steps (preprocess, train, evaluate per dataset; cross;
SHAP and stats per dataset). It records each finished step in
`outputs_revision/run_state/`. After an interruption it skips finished steps.
Inside `train`, each finished classifier is saved and re-loaded, so a power cut
loses at most the model that was being trained.

## Changes, mapped to the reviews

| Change | Files | Review |
|---|---|---|
| Duplicates removed on the full dataset **before** the split. Feature vectors seen with both labels are resolved by majority (ties dropped), and counts are logged. The split is then checked for zero shared vectors. | `src/preprocessing/prepare.py` (new), `main.py`, `pipeline_supervised.py` | R3.1 |
| The VAE branch uses the same de-duplicated frame and the same split. Previously it skipped de-duplication. | `pipeline_vae.py` | R3.1, R1.3 |
| Rows are no longer dropped from val/test by their true label. Outlier filtering applies to training rows only. | `cleaner.py` | R3.1 |
| SMOTE runs inside every CV fold (imblearn `Pipeline`). The training pool stays un-oversampled, and the search optimises MCC. | `pipeline_supervised.py`, `supervised_classifiers.py` | R3.2 |
| The significance tests refit correlation filter, RF selection, scaling and SMOTE inside each split. They run 5x2cv over 5 seeds, report MCC and F1, state what a "pair" is, and save a per-seed CSV. Results for each dataset are merged instead of overwritten. | `statistical_tests.py` | R1.2, R3.2, R3.6 |
| The best supervised model is chosen by **validation MCC**, used consistently for the ensemble, SHAP and the stats test. | `main.py` | consistency |
| IoT-23 `proto` is mapped to IANA numbers (tcp=6, udp=17, icmp=1), as in CICIoT2023. Before, it was alphabetical codes 0/1/2. | `prepare.py` | R1.1, R3.3 |
| `total_bytes` is mapped to `orig_ip_bytes+resp_ip_bytes` (headers included, like `tot_sum`), with payload bytes as a fallback. The mapping table is saved. | `cross_dataset.py`, `cross_dataset_eval.py` | R3.3 |
| Cross-dataset results now include bootstrap 95% CIs, a **within-source 2-feature control** (same model and the same 2 features on the source's own test set), and SHAP of the 2-feature transfer models. | `cross_dataset_eval.py` | R1.1, R3.4, R3.9 |
| **Bug fix: VAE anomaly score.** A double `reduce_mean` collapsed the reconstruction error into one batch-level number, so records differed only by the KL term. The score is now a per-record BCE plus beta*KL. | `vae.py` | R3.5 |
| VAE early stopping uses benign validation rows only. The chosen percentile, val MCC, epochs, benign count and thresholds are saved to `vae_summary.csv`. A **label-free threshold** (p99 of benign training scores) is added. | `vae.py`, `main.py` | R3.5 |
| VAE model size is measured from the weights file (it was always 0.0). Ensemble size and training time are the sums of their components. | `vae.py`, `hybrid_ensemble.py`, `main.py` | R3.7 |
| Hybrid: VAE scores are normalised by the benign-validation empirical CDF. The old min-max made the VAE contribute nearly nothing, which is why hybrid AUC equalled XGBoost's. Weights are tuned by val MCC, and the grid includes 1.0 (supervised only). | `hybrid_ensemble.py` | R3.8 |
| Ablation adds a **supervised model with a validation-tuned threshold** and a **label-free VAE**. Paired tests (McNemar and a paired-bootstrap CI of the MCC difference) go to `<ds>_paired_tests.csv`. | `ablation.py`, `uncertainty.py` (new) | R3.8 |
| Every test-set result carries bootstrap 95% CIs (MCC, F1, recall, FPR, AUC). | `metrics.py`, `uncertainty.py` | R3.4, R3.8 |
| Dataset counts (rows loaded, duplicates, conflicts, per-split class counts, feature counts) are saved to `<ds>_data_summary.json`. | `prepare.py`, `main.py` | R1.3, R3.10 |
| Hardware and library versions are saved to `run_info.json`. | `utils/run_info.py` (new) | R3.7, R3.10 |
| Larger plot fonts. | `visualizations.py`, `shap_analysis.py` | R3.10 |

## Settings worth knowing (`configs/config.py`)

- `STATS_SEEDS` (5 seeds) and `STATS_POOL_N` (100k rows) control the stats
  runtime. Reduce them if time is short.
- `IOT_OUTPUT_DIR` environment variable: overrides the output folder.
