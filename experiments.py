"""Cross-validation experiments behind the numbers of the report."""

import numpy as np

import cv
import features
from implementations import reg_logistic_regression

DATA_DIR = "data/dataset"
# L2 strength on the summed log-loss (penalty LAMBDA / 2 * ||w||^2), i.e. the
# course lambda_ = LAMBDA / (2 N), which stays comparable when N changes.
LAMBDA = 10.0
GAMMA = 1.0
MAX_ITERS = 500


def logistic_scores(x, y, train, test):
    """Fit the regularized logistic regression on train; return train and test logits."""
    tx_train, tx_test, _ = features.build_features(x[train], x[test])
    lambda_ = LAMBDA / (2 * len(train))
    w0 = np.zeros(tx_train.shape[1])
    w, _ = reg_logistic_regression(y[train], tx_train, lambda_, w0, MAX_ITERS, GAMMA)
    return tx_train @ w, tx_test @ w


def main():
    x, _, y, _ = features.load_data(DATA_DIR)
    folds = cv.stratified_folds(y)
    res, _ = cv.cross_validate(lambda tr, te: logistic_scores(x, y, tr, te), y, folds)
    print("logistic:", cv.summary(res))
    config = {"lambda": LAMBDA, "gamma": GAMMA, "max_iters": MAX_ITERS}
    cv.save_result("logistic", config, res)


if __name__ == "__main__":
    main()
