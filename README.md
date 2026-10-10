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
   This takes about 1 hour on a 10-core laptop, prints the 5-fold cross-validated F1 of the final model and writes `sample-submission.csv` (ids and predictions in {-1, 1}), the file submitted to AIcrowd. The first run parses the CSV files (~30 s) and caches them in `data/dataset/cache.npz`.

## Files

| file | content |
|---|---|
| `implementations.py` | the 6 required methods (GD / SGD least squares, least squares, ridge, logistic and regularized logistic regression) |
| `codebook.py` | meaning of the answer codes of every column, taken from the BRFSS 2015 codebook (missing codes, "none" codes, recodings, unit-coded answers, nominal columns) |
| `data_cleaning.py` | data loading, column dropping, code decoding, and the fit/transform cleaning pipeline (imputation, clipping, one-hot, missing indicators, standardization) |
| `cross_validation.py` | stratified hold-out split, stratified k-fold CV, F1 / accuracy, threshold selection |
| `models.py` | Newton logistic regression (class-weighted, L2), fully-connected neural networks (dropout, Adam), feature expansion (squares, interactions), histogram gradient-boosted trees |
| `tuning.py` | model and hyper-parameter selection with 5-fold CV; results in `results/tuning_results.md` |
| `run.py` | final pipeline producing `sample-submission.csv` |
| `data_cleaning_and_validation.ipynb` | explained walkthrough of the cleaning and validation, with before/after plots |
| `tests/` | unit tests (`pytest tests/`) |

## Method

- **Cleaning**: administrative columns (dates, sampling weights, phone screening) and redundant raw answers are dropped; every answer code is decoded with the official codebook (`don't know` / `refused` → NaN, `none` → 0, non-ordered code sets recoded, unit-coded answers converted). Statistics (medians, clipping quantiles, means/stds) are fitted on training data only. Missing values are median-imputed and a 0/1 "was missing" indicator is added, because in this survey whether a question was asked is informative.
- **Validation**: stratified 5-fold cross-validation (≈9% positives). The decision threshold is chosen on the training folds to maximize F1, since the default 0.5 threshold gives a very low F1 on such imbalanced data.
- **Model**: weighted average of the predicted probabilities of
  - **gradient-boosted decision trees** (weight 0.7), implemented from scratch: histogram-based (≤ 64 bins per feature, missing values in their own bin), logistic loss with second-order (Newton) leaf values, depth 6, learning rate 0.05, 300 trees, 50% of the rows and columns per tree, min. hessian 10 per leaf, L2 = 1 on leaf values; trained on the decoded columns (no imputation needed) and averaged over 4 seeds;
  - **neural networks** (weight 0.3): 2 hidden ReLU layers of 128 and 64 units, dropout 0.2, weight decay 1e-4, Adam (lr 1e-3, batch 512, 4 epochs), on the cleaned and standardized features, averaged over 10 seeds.

  The decision threshold maximizes the F1 of the out-of-fold predictions (≈ 0.21).

## Main results (stratified 5-fold CV, F1 of the positive class)

All configurations are in `results/tuning_results.md` (produced by `python tuning.py`). Stages A–F choose the threshold on the training folds; stage G (trees and final blends) uses one threshold chosen on the out-of-fold predictions, as `run.py` does (the two protocols agree within 0.0005 on the logistic model).

| configuration | CV F1 |
|---|---|
| logistic regression, columns with ≤ 50% missing values | 0.4259 ± 0.0047 |
| + keep all columns with missing-value indicators | 0.4372 ± 0.0053 |
| ridge regression (same features) | 0.4284 ± 0.0070 |
| class weights for the positives (2, 5, 10) | no gain (≤ 0.4372) |
| logistic + squares + 20 interactions | 0.4414 ± 0.0037 |
| neural network (128, 64), dropout 0.2, 1 seed / 10 seeds | 0.437 / 0.4434 |
| gradient-boosted trees, depth 6, 300 trees, 1 seed / 4 seeds | 0.4446 / 0.4459 |
| previous submission: 0.5 · logistic + 0.5 · networks (5 seeds) | 0.4440 ± 0.0055 |
| **final: 0.7 · trees (4 seeds) + 0.3 · networks (10 seeds)** | **0.4464 ± 0.0040** |

The final blend is better than the previous submission on each of the 5 folds (+0.0024 on average). Things that did not help: depth 4 or 8 trees, a lower learning rate with more trees, a cosine learning-rate schedule or wider networks once several seeds are averaged, more than 4 tree seeds, giving weight to the logistic model in the blend.

Using the default threshold 0.5 instead of a tuned threshold divides the F1 by about two (≈ 0.22).
