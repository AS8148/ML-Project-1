"""Train the final model on the whole training set and write submission.csv."""

import os

import numpy as np

import features
from cv import best_threshold
from helpers import create_csv_submission
from implementations import reg_logistic_regression

DATA_DIR = os.path.join("data", "dataset")
# The objective is the summed log-loss + LAMBDA / 2 * ||w||^2. In the course
# convention (mean log-loss + lambda_ * ||w||^2) this is lambda_ = LAMBDA / (2 N),
# so the same LAMBDA suits the CV folds and the full training set.
LAMBDA = 10.0
GAMMA = 1.0
MAX_ITERS = 500


def logistic_scores(x_fit, y_fit, x_apply, lam=LAMBDA, **options):
    """Build features and fit on x_fit; return the logits of x_fit and x_apply.

    The options are passed to features.build_features (for the ablation study).
    """
    tx_fit, tx_apply, _ = features.build_features(x_fit, x_apply, **options)
    lambda_ = lam / (2 * len(y_fit))
    w0 = np.zeros(tx_fit.shape[1])
    w, _ = reg_logistic_regression(y_fit, tx_fit, lambda_, w0, MAX_ITERS, GAMMA)
    return tx_fit @ w, tx_apply @ w


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths are relative to here
    x_train, x_test, y, test_ids = features.load_data(DATA_DIR)
    s_train, s_test = logistic_scores(x_train, y, x_test)
    threshold = best_threshold(y, s_train)
    pred = np.where(s_test >= threshold, 1, -1)
    create_csv_submission(test_ids, pred, "submission.csv")
    print(f"threshold {threshold:.4f}, positive rate {np.mean(pred == 1):.4f}")


if __name__ == "__main__":
    main()
