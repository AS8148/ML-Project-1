# EPFL CS-433 Project 1: predicting coronary heart disease (MICHD)

Binary classification of coronary heart disease on the BRFSS 2015 survey data,
using only Python and NumPy.

## Setup

- Python 3.9 or newer with NumPy (and matplotlib for the figure of `experiments.py`).
- Extract the data so that `x_train.csv`, `y_train.csv` and `x_test.csv` are in
  `data/dataset/`:

  ```
  python -m zipfile -e data/dataset_to_release.zip data/
  ```

## Usage

```
python run.py          # trains the final model and writes submission.csv
python experiments.py  # experiments of the report: results/*.json, report/threshold.pdf
```

Paths are relative to the repository, so the scripts can be launched from any
folder. The first run parses the CSV files, which takes several minutes and
about 3-4 GB of RAM, and caches the parsed arrays in `data/dataset/raw.npz`
(1.1 GB); later runs load them in seconds. With the cache, `run.py` takes about
10 minutes on a laptop. `experiments.py` takes four to five hours; groups of
experiments can be run alone, e.g. `python experiments.py methods ablations`.
Each experiment also saves its cross-validation scores to `results/oof_*.npy`
(not tracked), for later analysis without retraining.

`run.py` uses fixed random seeds, so its output is reproducible up to
floating-point differences between machines. Our submission file has MD5
`4663fde19985460f68dab1ba78314935` (same with NumPy 1.23 and 2.1).

## Files

| File | Content |
|---|---|
| `implementations.py` | The six required methods (least squares, ridge, GD/SGD, logistic regression). |
| `features.py` | Loading, decoding of the survey answer codes, feature encoding and scaling. |
| `codebook.py` | Decoding of all the columns with the BRFSS 2015 codebook (input of the gradient boosting), written by our teammate Samuel Bachmann. |
| `gbdt.py` | Gradient boosted trees for the logistic loss (NumPy only). |
| `cv.py` | Stratified 5-fold cross-validation and the F1-optimal decision threshold. |
| `run.py` | Final model (gradient boosting) and the logistic regression baseline; writes `submission.csv`. |
| `experiments.py` | Cross-validation experiments of the report, by group. |
| `helpers.py` | Course helpers to load the data and write the submission. |
| `results/` | Cross-validation results (one JSON per experiment) and AIcrowd scores (`leaderboard_log.csv`). |

## Method in short

1. Decode the BRFSS answer codes ("don't know" and "refused" become missing, "none"
   becomes 0, frequencies are converted to one unit): for the gradient boosting,
   all 260 columns left after dropping administrative and redundant ones, with the
   tables of the CDC codebook (`codebook.py`); for the logistic regression, 96
   selected questions (`features.py`).
2. Final model: gradient boosted trees of depth 3 with learning rate 0.05, row
   and column subsampling 0.5, and every question binned into at most 32
   quantiles (missing values in their own bin). The trees are fitted on 80% of
   the training rows; the other 20% stops the boosting and sets the decision
   threshold that maximizes the F1 score.
3. Baseline: an L2-regularized logistic regression trained with gradient descent
   (`reg_logistic_regression` of `implementations.py`) on imputed, one-hot
   encoded and standardized features, with the F1-optimal threshold on the
   training predictions.

Cross-validated F1 (5 folds): 0.442 for the gradient boosting, 0.438 for the
logistic regression.
