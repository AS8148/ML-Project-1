"""Produce the competition submission (sample-submission.csv).

Usage:  python run.py

Pipeline (all hyper-parameters were selected with tuning.py, see
results/tuning_results.md):
  1. load the data, drop administrative/redundant columns, decode the answer
     codes with the BRFSS 2015 codebook (data_cleaning.py, codebook.py);
  2. stratified 5-fold cross-validation of the final model: prints the
     estimated F1 and chooses the decision threshold that maximizes the F1 of
     the out-of-fold predictions;
  3. fit the cleaner and the model on the whole training set, predict the test
     set, apply the threshold and write the submission in {-1, 1} format.

Everything is deterministic (fixed seeds), so running it again gives the same file.
"""

import time

import numpy as np

from cross_validation import best_threshold, f1_score, stratified_k_indices
from data_cleaning import (
    ADMIN_COLUMNS,
    REDUNDANT_COLUMNS,
    drop_columns,
    fit_cleaner,
    load_data,
    replace_missing_codes,
    transform,
)
from helpers import create_csv_submission
from tuning import blend, expanded, mlp, newton

OUTPUT = "sample-submission.csv"
K_FOLD = 5
SEED = 1

# Best configuration found by tuning.py
CLEANER = dict(max_nan_frac=1.0, add_nan_indicators=True, extra_nominal=("_STATE",))
KEEP_STATE = True
LOGISTIC = dict(lambda_=1e-4, pos_weight=1.0)
EXPANSION = dict(squares=True, n_interact=20)
MLP = dict(hidden=(128, 64), epochs=4, lr=1e-3, l2=1e-4, dropout=0.2)
N_MLP = 5
ALPHA = 0.5  # weight of the logistic regression in the blend


def build_model():
    """fit_predict(y_tr, tx_tr, tx_other, is_continuous) of the final model."""
    linear = expanded(newton(**LOGISTIC), **EXPANSION)
    if ALPHA >= 1:
        return linear
    return blend(linear, mlp(n_models=N_MLP, **MLP), ALPHA)


def main():
    t0 = time.time()
    x_train, x_test, y_pm, _, test_ids, columns = load_data("data/dataset")
    y = (y_pm == 1).astype(float)
    to_drop = [c for c in ADMIN_COLUMNS + REDUNDANT_COLUMNS]
    if KEEP_STATE:
        to_drop.remove("_STATE")
    x_train, cols = drop_columns(x_train, columns, to_drop)
    x_test, _ = drop_columns(x_test, columns, to_drop)
    x_train = replace_missing_codes(x_train, cols)
    x_test = replace_missing_codes(x_test, cols)
    fit_predict = build_model()

    # 1. Cross-validation: F1 estimate and out-of-fold scores for the threshold
    folds = stratified_k_indices(y, K_FOLD, SEED)
    oof = np.zeros(len(y))
    f1s = []
    for f in range(K_FOLD):
        val_idx = folds[f]
        tr_idx = np.concatenate([folds[i] for i in range(K_FOLD) if i != f])
        params = fit_cleaner(x_train[tr_idx], cols, **CLEANER)
        s_tr, s_val = fit_predict(
            y[tr_idx],
            transform(x_train[tr_idx], params),
            transform(x_train[val_idx], params),
            params["is_continuous"],
        )
        thr_fold, _ = best_threshold(y[tr_idx], s_tr)
        f1s.append(f1_score(y[val_idx], (s_val >= thr_fold).astype(int)))
        oof[val_idx] = s_val
        print(f"fold {f + 1}/{K_FOLD}: F1 = {f1s[-1]:.4f}  ({time.time() - t0:.0f}s)")
    print(f"CV F1 = {np.mean(f1s):.4f} +- {np.std(f1s):.4f}")
    candidates = np.linspace(0.05, 0.6, 221)
    threshold, f1_oof = best_threshold(y, oof, candidates)
    print(f"threshold = {threshold:.4f} (out-of-fold F1 {f1_oof:.4f})")

    # 2. Final model on the whole training set
    params = fit_cleaner(x_train, cols, **CLEANER)
    _, s_test = fit_predict(
        y,
        transform(x_train, params),
        transform(x_test, params),
        params["is_continuous"],
    )
    y_pred = np.where(s_test >= threshold, 1, -1)
    create_csv_submission(test_ids, y_pred, OUTPUT)
    print(
        f"wrote {OUTPUT}: {len(y_pred)} predictions, "
        f"{np.mean(y_pred == 1):.1%} positive ({time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
