"""Train the final model on the whole training set and write submission.csv."""

import os

import numpy as np

import cv
import features
import gbdt
from helpers import create_csv_submission
from implementations import reg_logistic_regression

DATA_DIR = os.path.join("data", "dataset")
# Logistic regression baseline. Its objective is the summed log-loss
# + LAMBDA / 2 * ||w||^2; in the course convention (mean log-loss + lambda_ * ||w||^2)
# this is lambda_ = LAMBDA / (2 N), so the same LAMBDA suits every training size.
LAMBDA = 10.0
GAMMA = 1.0
MAX_ITERS = 500
# Final model: gradient boosted trees (other parameters: defaults of gbdt.fit).
DEPTH = 3
RATE = 0.05


def logistic_scores(x_fit, y_fit, x_apply, lam=LAMBDA, **options):
    """Build features and fit on x_fit; return the logits of x_fit and x_apply.

    lam replaces LAMBDA and the options go to features.build_features (ablations).
    """
    tx_fit, tx_apply, _ = features.build_features(x_fit, x_apply, **options)
    lambda_ = lam / (2 * len(y_fit))
    w0 = np.zeros(tx_fit.shape[1])
    w, _ = reg_logistic_regression(y_fit, tx_fit, lambda_, w0, MAX_ITERS, GAMMA)
    return tx_fit @ w, tx_apply @ w


def gbdt_scores(x_fit, y_fit, x_apply, depth=DEPTH, rate=RATE):
    """Boost trees on 80% of x_fit, stopping early on the other 20%.

    The scores of x_fit are NaN outside that 20%, where the threshold is chosen.
    """
    held = cv.stratified_folds(y_fit, seed=1) == 0
    params = {"depth": depth, "rate": rate}
    model = gbdt.fit(x_fit[~held], y_fit[~held], x_fit[held], y_fit[held], **params)
    print(f"  {len(model[3])} trees")
    s_fit = np.full(len(y_fit), np.nan)
    s_fit[held] = gbdt.predict(model, x_fit[held])
    return s_fit, gbdt.predict(model, x_apply)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths are relative to here
    x_train, x_test, y, test_ids = features.load_data(DATA_DIR)
    s_train, s_test = gbdt_scores(x_train, y, x_test)
    held = ~np.isnan(s_train)
    threshold = cv.best_threshold(y[held], s_train[held])
    pred = np.where(s_test >= threshold, 1, -1)
    create_csv_submission(test_ids, pred, "submission.csv")
    print(f"threshold {threshold:.4f}, positive rate {np.mean(pred == 1):.4f}")


if __name__ == "__main__":
    main()
