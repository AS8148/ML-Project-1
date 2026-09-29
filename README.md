# EPFL CS-433 Project 1: predicting coronary heart disease (MICHD)

Binary classification of coronary heart disease on the BRFSS 2015 survey data,
using only Python and NumPy.

## Setup

- Python 3.9 or newer with NumPy.
- Extract the data so that `x_train.csv`, `y_train.csv` and `x_test.csv` are in
  `data/dataset/`:

  ```
  python -m zipfile -e data/dataset_to_release.zip data/
  ```

## Usage

```
python run.py          # trains the final model and writes submission.csv
python experiments.py  # cross-validation experiments, results in results/*.json
```

Paths are relative to the repository, so the scripts can be launched from any
folder. The first run parses the CSV files, which takes several minutes and
about 3-4 GB of RAM, and caches the parsed arrays in `data/dataset/raw.npz`
(1.1 GB); later runs load them in seconds. With the cache, `run.py` takes 2-4
minutes and `experiments.py` about 12 minutes on a laptop.

`run.py` uses no random numbers, so its output is reproducible up to
floating-point differences between machines. Our submission file has MD5
`2b80879eda4fa6b62904761b01f70b0a` (same with NumPy 1.23 and 2.1).

## Files

| File | Content |
|---|---|
| `implementations.py` | The six required methods (least squares, ridge, GD/SGD, logistic regression). |
| `features.py` | Loading, decoding of the survey answer codes, feature encoding and scaling. |
| `cv.py` | Stratified 5-fold cross-validation and the F1-optimal decision threshold. |
| `run.py` | Final model and its hyperparameters: trained on the whole training set, writes `submission.csv`. |
| `experiments.py` | Cross-validation experiments reported in the report. |
| `helpers.py` | Course helpers to load the data and write the submission. |

## Method in short

1. Decode the BRFSS answer codes of 96 selected questions ("don't know" and
   "refused" become missing, "none" becomes 0, frequencies are converted to one unit).
2. Impute medians and add missing-value indicators, one-hot encode nominal
   questions, flag whether the cardiac rehabilitation question was answered,
   and standardize, with all statistics fitted on the training rows only.
3. Train an L2-regularized logistic regression with gradient descent
   (`reg_logistic_regression` of `implementations.py`).
4. Predict a positive when the logit is at least the threshold that maximizes
   the F1 score on the training predictions.
