"""Produce the competition submission (sample-submission.csv).

Usage:  python run.py            (~1 hour on a 10-core laptop)

Pipeline (all hyper-parameters were selected with tuning.py, see
results/tuning_results.md):
  1. load the data, drop administrative/redundant columns, decode the answer
     codes with the BRFSS 2015 codebook (data_cleaning.py, codebook.py);
  2. stratified 5-fold cross-validation of the final model: prints the
     estimated F1 and chooses the decision threshold that maximizes the F1 of
     the out-of-fold predictions;
  3. fit the model on the whole training set, predict the test set, apply the
     threshold and write the submission in {-1, 1} format.

The final model is a weighted average of the predicted probabilities of
  * a logistic regression (Newton) on the cleaned features + squares +
    pairwise interactions of the most label-correlated features,
  * neural networks (2 hidden layers, dropout) averaged over several seeds,
  * gradient-boosted decision trees on the decoded raw columns, averaged over
    several seeds.
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
from models import (
    apply_bins,
    apply_expansion,
    fit_bins,
    fit_expansion,
    logistic_newton,
    predict_gbdt,
    predict_mlp,
    sigmoid,
    train_gbdt,
    train_mlp,
)

OUTPUT = "sample-submission.csv"
K_FOLD = 5
SEED = 1

# Best configuration found by tuning.py
CLEANER = dict(max_nan_frac=1.0, add_nan_indicators=True, extra_nominal=("_STATE",))
LOGISTIC = dict(lambda_=1e-4)
EXPANSION = dict(squares=True, n_interact=20)
MLP = dict(hidden=(128, 64), epochs=4, lr=1e-3, l2=1e-4, dropout=0.2)
N_MLP = 10
GBDT = dict(max_depth=6, lr=0.05, min_child_weight=10, reg_lambda=1.0, n_trees=300)
GBDT_SAMPLING = dict(subsample=0.5, colsample=0.5)
N_GBDT = 4
# Weights of the blend (logistic regression gets no weight once trees are used)
WEIGHTS = {"logistic": 0.0, "mlp": 0.3, "gbdt": 0.7}


def predict_blend(x_tr, y_tr, x_other, cols):
    """Fit all members on (x_tr, y_tr) and return the blended probabilities on x_other.

    Args:
        x_tr: (N, D) decoded training features (NaN = missing).
        y_tr: (N,) labels in {0, 1}.
        x_other: (M, D) decoded features to predict.
        cols: (D,) column names.

    Returns:
        (M,) blended probabilities.
    """
    out = np.zeros(len(x_other))
    if WEIGHTS["logistic"] > 0 or WEIGHTS["mlp"] > 0:
        params = fit_cleaner(x_tr, cols, **CLEANER)
        tx_tr, tx_other = transform(x_tr, params), transform(x_other, params)
    if WEIGHTS["logistic"] > 0:
        exp = fit_expansion(tx_tr, y_tr, params["is_continuous"], **EXPANSION)
        w = logistic_newton(y_tr, apply_expansion(tx_tr, exp), **LOGISTIC)
        out += WEIGHTS["logistic"] * sigmoid(apply_expansion(tx_other, exp) @ w)
    if WEIGHTS["mlp"] > 0:
        for seed in range(N_MLP):
            net = train_mlp(y_tr, tx_tr, seed=seed, **MLP)
            out += WEIGHTS["mlp"] * predict_mlp(tx_other, net) / N_MLP
    if WEIGHTS["gbdt"] > 0:
        edges = fit_bins(x_tr)
        xb_tr, xb_other = apply_bins(x_tr, edges), apply_bins(x_other, edges)
        for seed in range(N_GBDT):
            model = train_gbdt(y_tr, xb_tr, seed=seed, **GBDT, **GBDT_SAMPLING)
            out += WEIGHTS["gbdt"] * predict_gbdt(xb_other, model) / N_GBDT
    return out


def main():
    t0 = time.time()
    x_train, x_test, y_pm, _, test_ids, columns = load_data("data/dataset")
    y = (y_pm == 1).astype(float)
    to_drop = [c for c in ADMIN_COLUMNS + REDUNDANT_COLUMNS if c != "_STATE"]
    x_train, cols = drop_columns(x_train, columns, to_drop)
    x_test, _ = drop_columns(x_test, columns, to_drop)
    x_train = replace_missing_codes(x_train, cols)
    x_test = replace_missing_codes(x_test, cols)

    # 1. Cross-validation: out-of-fold predictions, F1 estimate and threshold
    folds = stratified_k_indices(y, K_FOLD, SEED)
    oof = np.zeros(len(y))
    for f in range(K_FOLD):
        val_idx = folds[f]
        tr_idx = np.concatenate([folds[i] for i in range(K_FOLD) if i != f])
        oof[val_idx] = predict_blend(x_train[tr_idx], y[tr_idx], x_train[val_idx], cols)
        print(f"fold {f + 1}/{K_FOLD} done ({time.time() - t0:.0f}s)", flush=True)
    candidates = np.linspace(0.05, 0.6, 221)
    threshold, f1_oof = best_threshold(y, oof, candidates)
    f1s = [f1_score(y[fd], (oof[fd] >= threshold).astype(int)) for fd in folds]
    print(f"threshold = {threshold:.4f}")
    print(
        f"CV F1 = {np.mean(f1s):.4f} +- {np.std(f1s):.4f} (out-of-fold F1 {f1_oof:.4f})"
    )

    # 2. Final model on the whole training set
    s_test = predict_blend(x_train, y, x_test, cols)
    y_pred = np.where(s_test >= threshold, 1, -1)
    create_csv_submission(test_ids, y_pred, OUTPUT)
    print(
        f"wrote {OUTPUT}: {len(y_pred)} predictions, "
        f"{np.mean(y_pred == 1):.1%} positive ({time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
