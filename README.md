# EPFL CS-433 Project 1: predicting coronary heart disease (MICHD)

Binary classification of coronary heart disease on the BRFSS 2015 survey data,
using only Python and NumPy.

## Setup

- Python 3.9 or newer with NumPy (and matplotlib for the figure of `experiments.py`).
- The data are not included in this repository. Download
  `data/dataset_to_release.zip` from the course repository
  (https://github.com/epfml/ML_course/tree/main/projects/project1) and extract
  `x_train.csv`, `y_train.csv` and `x_test.csv` into `data/dataset/`, for
  example, with the archive saved at the root of the repository:

  ```
  python -m zipfile -e dataset_to_release.zip data/
  ```

## Usage

```
python run.py          # trains the final model and writes submission.csv
python experiments.py  # experiments of the report: results/*.json, report/threshold.pdf
```

`python run.py` produces our selected submission (F1 0.454 on AIcrowd).

Paths are relative to the repository, so the scripts can be launched from any
folder. The first run parses the CSV files, which takes several minutes and
about 3-4 GB of RAM, and caches the parsed arrays in `data/dataset/raw.npz`
(1.1 GB); later runs load them in seconds. With the cache, `run.py` takes about
8.5 minutes on a laptop. `experiments.py` takes about five hours; groups of
experiments can be run alone, e.g. `python experiments.py methods ablations`.
Each experiment also saves its cross-validation scores to `results/oof_*.npy`
(not tracked), for later analysis without retraining. Five variants of Table II
of the report were tried and rejected, and their code is not part of this
submission: the average with a neural network, bagging of 3 models, 64 bins,
rate 0.025 and trees started from the logistic logits.

`run.py` uses fixed random seeds, so its output is reproducible up to
floating-point differences between machines. Our submission file has MD5
`c7e12cb2ebc684664ba47d805ec6dec6` (F1 0.454 on AIcrowd).

## Blend of Samuel Bachmann (`blend/`)

`blend/` holds a heavier model written by our teammate Samuel Bachmann: the
average of gradient boosted trees of depth 6 (300 trees, 4 seeds, weight 0.7)
and neural networks with two hidden layers (10 seeds, weight 0.3). It scored
F1 0.456 on AIcrowd, the best online score of the team, and 0.4464 +- 0.0040 in
its own 5-fold cross-validation (other folds and protocol than ours, see
`blend/tuning_results.md`). It is not our selected submission: it runs for
about one hour on a 10-core laptop for a small gain, and we kept the simpler
model of `run.py`. To reproduce it:

```
python blend/run_blend.py   # writes submission_blend.csv at the root of the repository
python blend/tuning.py      # its model selection, appended to blend/tuning_results.md
```

The folder is self-contained (its own copies of the helpers, the cleaning and
the codebook tables) and reads the same `data/dataset/` folder; its first run
caches the parsed CSV files in `data/dataset/cache.npz`.
`blend/implementations.py` is Samuel's copy of the six methods, used by
`blend/tuning.py`; the graded implementations are in the root
`implementations.py`.

## Files

| File | Content |
|---|---|
| `implementations.py` | The six required methods (least squares, ridge, GD/SGD, logistic regression). |
| `features.py` | Loading, decoding of the survey answer codes, feature encoding and scaling. |
| `codebook.py` | Decoding of the 260 non-administrative columns with the BRFSS 2015 codebook (input of the gradient boosting), written by our teammate Samuel Bachmann. |
| `gbdt.py` | Gradient boosted trees for the logistic loss (NumPy only). |
| `cv.py` | Stratified 5-fold cross-validation and the F1-optimal decision threshold. |
| `run.py` | Final model (gradient boosting) and the logistic regression baseline; writes `submission.csv`. |
| `experiments.py` | Cross-validation experiments of the report, by group. |
| `helpers.py` | Course helpers to load the data and write the submission. |
| `results/` | Cross-validation results (one JSON per experiment) and AIcrowd scores (`leaderboard_log.csv`). |
| `report/` | The report (`report.tex`, `report.pdf`), its bibliography and figure. |
| `blend/` | Samuel Bachmann's blend (`run_blend.py`), its modules and model selection (`tuning.py`, `tuning_results.md`). |

## Method in short

1. Decode the BRFSS answer codes ("don't know" and "refused" become missing, "none"
   becomes 0, frequencies are converted to one unit). The gradient boosting reads
   the 260 columns left after dropping administrative and redundant ones, decoded
   with the tables of the CDC codebook (`codebook.py`). The logistic regression
   reads 96 selected questions, decoded by `features.py`. The answer HAREHAB1
   (cardiac rehabilitation, asked only after a heart attack) is kept: it is
   answered by 0.24% of the training rows, all positive.
2. Final model: gradient boosted trees of depth 3 with learning rate 0.05, row
   and column subsampling 0.5, and every question binned into at most 32
   quantiles (missing values in their own bin). The trees are fitted on 80% of
   the training rows; the other 20% stops the boosting and sets the decision
   threshold that maximizes the F1 score.
3. Baseline: an L2-regularized logistic regression trained with gradient descent
   (`reg_logistic_regression` of `implementations.py`) on imputed, one-hot
   encoded and standardized features, with the F1-optimal threshold on the
   training predictions.

Cross-validated F1 (5 folds, seed 42): 0.444 for the gradient boosting (0.444
on the folds of seed 43), 0.438 for the logistic regression.
