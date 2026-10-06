# ML Project 1 — Predicting coronary heart disease (MICHD) from the BRFSS survey

EPFL CS-433 Machine Learning, Fall 2026. Only Python, NumPy and matplotlib (for plots) are used.

## Reproducing the submission

1. Unzip the data so that the CSV files are in `data/dataset/`:
   ```bash
   unzip data/dataset_to_release.zip -d data
   ```
2. Run:
   ```bash
   python run.py
   ```
   This prints the 5-fold cross-validated F1 of the final model and writes `sample-submission.csv` (ids and predictions in {-1, 1}), the file submitted to AIcrowd. The first run parses the CSV files (~30 s) and caches them in `data/dataset/cache.npz`.

## Files

| file | content |
|---|---|
| `implementations.py` | the 6 required methods (GD / SGD least squares, least squares, ridge, logistic and regularized logistic regression) |
| `codebook.py` | meaning of the answer codes of every column, taken from the BRFSS 2015 codebook (missing codes, "none" codes, recodings, unit-coded answers, nominal columns) |
| `data_cleaning.py` | data loading, column dropping, code decoding, and the fit/transform cleaning pipeline (imputation, clipping, one-hot, missing indicators, standardization) |
| `cross_validation.py` | stratified hold-out split, stratified k-fold CV, F1 / accuracy, threshold selection |
| `models.py` | Newton logistic regression (class-weighted, L2), one-hidden-layer neural network (Adam), feature expansion (squares, interactions) |
| `tuning.py` | model and hyper-parameter selection with 5-fold CV; results in `results/tuning_results.md` |
| `run.py` | final pipeline producing `sample-submission.csv` |
| `data_cleaning_and_validation.ipynb` | explained walkthrough of the cleaning and validation, with before/after plots |
| `tests/` | unit tests (`pytest tests/`) |

## Method

- **Cleaning**: administrative columns (dates, sampling weights, phone screening) and redundant raw answers are dropped; every answer code is decoded with the official codebook (`don't know` / `refused` → NaN, `none` → 0, non-ordered code sets recoded, unit-coded answers converted). Statistics (medians, clipping quantiles, means/stds) are fitted on training data only. Missing values are median-imputed and a 0/1 "was missing" indicator is added, because in this survey whether a question was asked is informative.
- **Validation**: stratified 5-fold cross-validation (≈9% positives). The decision threshold is chosen on the training folds to maximize F1, since the default 0.5 threshold gives a very low F1 on such imbalanced data.
- **Model**: average of the predicted probabilities of
  - an L2-regularized logistic regression (λ = 1e-4, Newton's method) on the cleaned features + squares of continuous features + pairwise products of the 20 features most correlated with the label, and
  - 5 neural networks (2 hidden ReLU layers of 128 and 64 units, dropout 0.2, weight decay 1e-4, Adam, 4 epochs) trained with different seeds.

  The decision threshold maximizes the F1 of the out-of-fold predictions (≈ 0.22).

## Main results (stratified 5-fold CV, F1 of the positive class)

All 40+ configurations are in `results/tuning_results.md` (produced by `python tuning.py`).

| configuration | CV F1 |
|---|---|
| logistic regression, columns with ≤ 50% missing values | 0.4259 ± 0.0047 |
| + keep all columns with missing-value indicators | 0.4372 ± 0.0053 |
| ridge regression (same features) | 0.4284 ± 0.0070 |
| class weights for the positives (2, 5, 10) | no gain (≤ 0.4372) |
| logistic + squares + 20 interactions | 0.4418 ± 0.0039 |
| neural network (128, 64), dropout 0.2, 1 seed / 5 seeds | 0.4371 / 0.4425 |
| **0.5 · logistic (expanded) + 0.5 · networks (5 seeds)** | **0.4430 ± 0.0052** |

Using the default threshold 0.5 instead of a tuned threshold divides the F1 by about two (≈ 0.22).
