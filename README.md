# ParkinSense: Four-Channel EEG Classification Baseline

## Overview

ParkinSense is a NYAS Junior Academy team project exploring an affordable, reusable EEG headband concept for potential neurological screening, with agricultural workers as an intended community. This notebook evaluates a narrower question: can features from four awake resting EEG channels distinguish participants already diagnosed with Parkinson's disease (PD) from controls?

This is an exploratory computational feasibility study. The experiment does not validate early detection, pesticide-related risk, overnight screening, or recordings from the actual headband. It is not a clinical diagnostic tool.

## Dataset and cohort

- Source: OpenNeuro ds008768, version 1.0.0, *Resting-State EEG in Parkinson's Disease and Healthy Controls*.
- Dataset: https://openneuro.org/datasets/ds008768/versions/1.0.0
- NEMAR serving-copy DOI: https://doi.org/10.82901/nemar.on008768
- Signal download base: https://nemar.s3.us-east-2.amazonaws.com/on008768/zarr/
- Condition: awake, eyes-open resting EEG.
- The dataset describes 358 recordings from 312 unique participants; some people have repeated sessions.
- This analysis uses one first-session recording per selected participant: 267 people, comprising 160 PD participants and 107 controls.
- Labels: 0 = control, 1 = PD.

The participant selection and five fold assignments were reused from the earlier project analysis and are explicitly listed in the notebook. That earlier selection retained the original 500 Hz acquisition group: 36 original 25,000 Hz participants were excluded because all belonged to the PD group, creating an acquisition-rate confound. Nine additional participants failed the earlier quality/duration requirements, leaving 267. The Colab reproduction begins with this fixed cohort; it does not independently repeat that initial cohort-selection audit. All 267 selected participants passed the current four-channel feature-availability checks.

The notebook downloads the dequantized 250 Hz NEMAR serving representation of the original 500 Hz recordings. It does not directly process the original BrainVision files. The download URLs are live and their contents could change; the notebook does not enforce file hashes or pin the remote data to an immutable revision.

### Dataset attribution

Brooke Yeager, Arturo Espinoza, Rachel Cole, Christina Weber, Ergun Uc, James Cavanagh and Nandakumar Narayanan (2026). *Resting-State EEG in Parkinson's Disease and Healthy Controls*, v1.0.0. Dataset authors retain credit for data collection and release. The NEMAR dataset metadata lists CC0; consult the source record for current license and citation details. The team did not collect these EEG recordings.

## Channels and features

The four signal channels are Fz, C3, Cz and C4. Pz is the source recording reference and is not loaded as a separate signal channel. Ground is separate in the proposed hardware and is not a classifier input. Matching scalp sites does not establish that the physical wearable produces equivalent signals.

For each participant, the notebook:

1. Decodes stored signals into microvolts using the supplied scale and offset.
2. Divides the full recording into non-overlapping four-second windows, discarding an incomplete trailing window.
3. Checks each channel's windows independently: all values must be finite, peak-to-peak amplitude must be below 250 microvolts, and standard deviation must exceed 0.5 microvolts.
4. Requires at least ten accepted windows on each selected channel.
5. Computes detrended Welch spectra and takes the median spectrum over that channel's accepted windows.
6. Extracts relative delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), and beta (13-30 Hz) power, theta/alpha ratio, and alpha peak frequency.

Relative power is normalized by the sum of the four listed bands. Four channels with six features each produce 24 input features per participant. Quality checks reject obvious problems, but do not guarantee removal of all eye, muscle or electrode artifacts. These are candidate statistical features, not established disease-specific diagnostic biomarkers.

## Model and evaluation

The completed notebook uses a custom regularized Linear Discriminant Analysis (LDA) classifier with covariance shrinkage 0.2 and equal class priors. It does not use Random Forest or Logistic Regression for this baseline.

Evaluation uses the five saved participant folds. Each round trains on 213 or 214 people and tests on the remaining 54 or 53. Each participant receives one held-out prediction. Standardization and LDA parameters are fitted using only training participants. IDs, fold numbers and clinical scores are not classifier inputs.

The classification threshold is fixed at 0.5 on sigmoid-transformed LDA scores. These scores are not calibrated probabilities of having PD. There is no threshold tuning in this notebook.

The cohort and folds have supported earlier project development, so these are exploratory internal-validation results rather than a pristine final test. This notebook does not rerun the alternative-electrode search or external-cohort experiments from the earlier project package.

## Results

| Metric | Held-out result |
|---|---:|
| Accuracy | 68.5% |
| Balanced accuracy | 69.3% |
| Sensitivity | 65.6% |
| Specificity | 72.9% |
| ROC AUC | 0.738 |
| Coverage within selected cohort | 100.0% |

Accuracy is 183 correct predictions out of 267. Balanced accuracy averages sensitivity and specificity, giving the two classes equal importance. Predicting PD for everyone would yield 59.9% accuracy but 50% balanced accuracy. ROC AUC measures ranking across thresholds, not classification accuracy.

### Confusion matrix

| Actual group | Predicted control | Predicted PD |
|---|---:|---:|
| Control | 78 | 29 |
| PD | 55 | 105 |

### Fold results

| Fold | Test participants | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| 1 | 54 | 64.8% | 68.9% |
| 2 | 54 | 70.4% | 70.7% |
| 3 | 53 | 66.0% | 65.3% |
| 4 | 53 | 67.9% | 65.3% |
| 5 | 53 | 73.6% | 75.7% |

Overall metrics are calculated from the pooled held-out predictions, not by selecting the best fold. Coverage means everyone in the selected cohort received a score, not that every classification was correct.

After evaluation, a separate final LDA model is fitted on all 267 participants for future testing. The reported held-out metrics describe the cross-validation procedure, not an independent evaluation of that final fitted model.

## Files

| File | Purpose |
|---|---|
| `Parkinsense.ipynb` | Executed Colab notebook containing downloading, extraction, training, checks and graphs |
| `requirements.txt` | Pinned Python package versions; the same installation command is included in the notebook |
| `cohort.csv` | Selected participant IDs, labels and saved folds |
| `channel_quality.csv` | Accepted and total four-second windows per participant and channel |
| `features.csv` | Extracted 24-feature table, with IDs, labels and folds |
| `held_out_predictions.csv` | Held-out scores and predicted labels for each participant |
| `fold_metrics.csv` | Performance for each outer fold |
| `metrics.json` | Overall metrics and confusion counts |
| `baseline_results.png` | Confusion matrix and ROC curve |
| `four_channel_lda.joblib` | Final fitted LDA parameter dictionary and configuration |

The run also creates `features.npz` and `download_errors.json`. These are reproducible intermediates and need not be included in the submission. Downloaded EEG files are not bundled.

## Run in Google Colab

1. Open https://colab.research.google.com/ and upload `Parkinsense.ipynb`.
2. Use a standard CPU runtime; a GPU is not required.
3. Run the cells in order, starting with package installation. If Colab requests a runtime restart after installation, restart and rerun the import and subsequent cells.
4. The notebook recreates the cohort, downloads recordings, extracts features, fits the five evaluation models, checks metrics, produces graphs and fits the final model.
5. Download the output files before ending the session. Data are stored under `/content/ParkinSense_rebuild`, which is temporary. No Google Drive mount is required.

A fresh runtime needs internet access and sufficient temporary disk space for the recordings. Downloads may take time. If a download fails, rerun the download cell; completed local recordings are reused. After a runtime reset, temporary downloads and outputs must be regenerated unless separately saved.

The final notebook cell downloads the metrics, held-out predictions, fold metrics, graph and fitted model. To download the additional tables, run:

```python
from google.colab import files

for name in ["cohort.csv", "channel_quality.csv", "features.csv"]:
    files.download(str(root / name))
```

Save the executed notebook through File > Download > Download .ipynb.

To run outside Colab, install `requirements.txt` and adapt the Colab-specific download cells and `/content` path. The notebook is primarily intended for Colab.

## Using the saved model

`four_channel_lda.joblib` contains a dictionary of parameters, not a standalone scikit-learn estimator. Use the notebook's `predict_lda` function with `bundle["estimator"]` after applying the identical feature extraction and channel order. For one feature vector, the input must have shape `(1, 24)`.

Raw EEG cannot be passed directly to this file. Sampling, units, reference and feature definitions must match. The current preprocessing-quality thresholds are defined in the notebook; not all are duplicated in the saved bundle. Only load joblib files from trusted sources.

## Limitations and interpretation

- These results concern already-diagnosed participants and awake resting recordings. Diagnosed does not mean every participant was late-stage.
- Sleep, early or prediagnostic detection, pesticide exposure classification and actual-device performance were not tested.
- The model missed 55 of 160 PD participants and incorrectly flagged 29 of 107 controls. It is not ready to issue medical screening recommendations.
- Four-channel features, finite sample size, artifacts and cohort differences may limit performance; these experiments do not isolate which factor causes the errors.
- We have not established that medication explains the accuracy or that increasing channel count will improve it.
- Current QC starts from a cohort previously filtered using the original channel/duration criteria. Results should not be interpreted as availability or performance in an unselected community sample.
- The notebook checks arithmetic and participant separation, but does not independently validate the inherited labels, exclusions or clinical diagnoses against source documentation.
- A new, untouched cohort and recordings from the physical prototype are needed before making broader performance claims.

## AI assistance and team contribution

ChatGPT assisted with code generation and adaptation, debugging guidance, interpretation and documentation. The student ran the notebook in Google Colab and inspected its saved results. This package should not be described as code written entirely without assistance. Project decisions, contributions and any additional AI use should be disclosed according to the applicable NYAS submission rules.
