# ParkinSense

# shrinkage-lda

> A small, robust Linear Discriminant Analysis classifier for binary tabular data —
> with covariance shrinkage, NaN-safe handling, and coverage-aware metrics.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/numpy-%E2%89%A51.20-013243.svg)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%E2%89%A51.0-F7931E.svg)](https://scikit-learn.org/)

---

## Why this exists

`sklearn.discriminant_analysis.LinearDiscriminantAnalysis` is great, but in
small-`n`, many-`p` settings (biomarkers, omics, clinical cohorts) its pooled
covariance estimate is unstable and can even be singular. This module keeps the
LDA idea but adds three pragmatic fixes:

- **Covariance shrinkage** toward a scaled identity — keeps `Σ` well-conditioned.
- **NaN/inf safety** — invalid rows are dropped at fit time and flagged as
  unscorable at predict time, instead of crashing or silently biasing results.
- **Coverage-aware metrics** — `calculate_metrics` reports *how many* samples
  were actually scorable, not just the accuracy on whatever survived.

The result is a ~90-line, dependency-light classifier you can audit and cite.

---

## Features

| | |
|---|---|
| ✅ Binary LDA with shrinkage | `Σ ← (1−λ)·Σ + λ·(tr Σ / p)·I + εI` |
| ✅ Per-feature standardization | zero-variance features handled safely |
| ✅ Non-finite value handling | dropped at fit, returned as `NaN` at predict |
| ✅ Sigmoid-transformed scores | `score > 0.5` ⇒ class 1 |
| ✅ Rich metrics | accuracy, balanced accuracy, sens/spec, ROC-AUC, coverage, confusion counts |
| ✅ No black boxes | plain NumPy + two SciPy/sklearn helpers |

---

## Installation

Clone the repo and import directly (no packaging required):

```bash
git clone https://github.com/<your-user>/shrinkage-lda.git
cd shrinkage-lda
```

**Requirements**

- Python ≥ 3.8
- `numpy`
- `scipy` (for `scipy.special.expit`)
- `scikit-learn` (for `roc_auc_score` only)

```bash
pip install numpy scipy scikit-learn
```

---

## Quick start

```python
import numpy as np
from sklearn.model_selection import train_test_split

from shrinkage_lda import fit_lda, predict_lda, calculate_metrics

# X: (n_samples, n_features) float array — NaNs allowed
# y: (n_samples,) int array of 0/1 labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model = fit_lda(X_train, y_train, shrinkage=0.2)
scores = predict_lda(model, X_test)          # scores in [0, 1], NaN = unscorable
metrics = calculate_metrics(y_test, scores)

print(metrics["balanced_accuracy"], metrics["roc_auc"], metrics["coverage"])
```

---

## API reference

### `fit_lda(X_train, y_train, shrinkage=0.2)`

Trains the classifier and returns a model `dict`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X_train` | `np.ndarray (n, p)` | — | Feature matrix. Rows with any non-finite value are dropped. |
| `y_train` | `np.ndarray (n,)` | — | Binary labels: `0` = control, `1` = case. |
| `shrinkage` | `float ∈ [0, 1]` | `0.2` | Shrinkage strength toward the scaled identity. `0` = off, `1` = fully isotropic. |

**Returns** — `dict` with keys:

| Key | Meaning |
|---|---|
| `mean` | Per-feature mean used for standardization |
| `scale` | Per-feature std (zero-variance features → `1`) |
| `weights` | Discriminant vector, `Σ⁻¹(μ₁ − μ₀)` |
| `bias` | Intercept, `−½(μ₀ + μ₁)ᵀw` |
| `shrinkage` | Value used (for provenance) |

**Raises** — `ValueError` if either class has < 2 usable samples.

---

### `predict_lda(model, X_test)`

| Parameter | Type | Description |
|---|---|---|
| `model` | `dict` | Output of `fit_lda` |
| `X_test` | `np.ndarray (m, p)` | Test features |

**Returns** — `np.ndarray (m,)` of scores in `[0, 1]`. Rows with non-finite values
are returned as `np.nan` (not scored).

---

### `calculate_metrics(target, scores)`

| Parameter | Type | Description |
|---|---|---|
| `target` | `np.ndarray` | Ground-truth `0/1` labels |
| `scores` | `np.ndarray` | Predicted scores (`NaN` = unscorable) |

**Returns** — `dict`:

| Key | Meaning |
|---|---|
| `n` | Total samples |
| `accuracy` | `(TP+TN)/n` (unscorable counted as errors) |
| `balanced_accuracy` | `(sensitivity + specificity) / 2` |
| `sensitivity` | `TP / (TP+FN)` |
| `specificity` | `TN / (TN+FP)` |
| `coverage` | Fraction of samples with a valid score |
| `scorable_accuracy` | Accuracy on scored samples only (or `None`) |
| `roc_auc` | ROC-AUC on scored samples (or `None` if only one class present) |
| `tn, fp, fn, tp` | Confusion-matrix counts |
| `no_result_control` | Control samples that were unscorable |
| `no_result_pd` | Case samples that were unscorable |

---

## How it works

1. **Drop invalid rows.** Any row of `X_train` containing `NaN`/`inf` is removed
   along with its label. A class-balance check guards against degenerate folds.
2. **Standardize.** Compute per-feature `mean` and `std`; features with
   `std < 1e-12` get `scale = 1` so we never divide by zero.
3. **Estimate the pooled covariance.**
   `Σ = RᵀR / (n − 2)`, where `R` are residuals around class means.
4. **Shrink.**
   `Σ ← (1−λ)·Σ + λ·(tr Σ / p)·I + 1e-8·I`.
   This is the Ledoit–Wolf-style "shrink to scaled identity" family — simple,
   fast, and effective when `n` is small relative to `p`.
5. **Solve for the discriminant.** `w = Σ⁻¹(μ₁ − μ₀)`, `b = −½(μ₀ + μ₁)ᵀw`.
6. **Score.** `score = σ(xᵀw + b)` via the logistic function.

The logistic transform is monotonic, so **ranking** (and therefore ROC-AUC)
is unchanged by it — but note the scores are **not calibrated probabilities**.

---

## Design notes & caveats

- **Shrinkage matters.** With `shrinkage=0` and correlated features, the
  covariance can be singular and `np.linalg.solve` will raise. The default
  `0.2` is a reasonable starting point; tune it on a validation split.
- **Scores are not calibrated.** `0.5` is a threshold, not a probability.
  If you need calibration, wrap in isotonic/Platt scaling.
- **Unscorable ≠ negative.** Use `coverage` and `no_result_*` to report how
  often the model *could not* produce an answer — especially important in
  clinical use cases.
- **Labels must be non-negative integers** (`np.bincount` is used internally).
- **Binary only.** Multiclass is intentionally out of scope.

---

## Suggested repo layout

```
shrinkage-lda/
├── README.md
├── LICENSE
├── requirements.txt
├── shrinkage_lda.py        # the three functions above
└── examples/
    └── quickstart.py
```

---

## Citation

If this code contributes to a publication, please cite the repository:

```bibtex
@software{shrinkage_lda,
  title  = {shrinkage-lda: A robust Linear Discriminant Analysis classifier},
  author = {<Your Name>},
  year   = {2025},
  url    = {https://github.com/<your-user>/shrinkage-lda}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
