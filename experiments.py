"""Cross-validation experiments behind the numbers of the report."""

import os

import numpy as np

import cv
import features
import run

NEWTON_ITERS = 10


def newton_scores(x_fit, y_fit, x_apply):
    """Minimize the objective of run.logistic_scores with Newton's method.

    It reaches the exact optimum, to check that gradient descent has converged.
    """
    tx_fit, tx_apply, _ = features.build_features(x_fit, x_apply)
    w = np.zeros(tx_fit.shape[1])
    for _ in range(NEWTON_ITERS):
        p = 1 / (1 + np.exp(-(tx_fit @ w)))
        grad = tx_fit.T @ (p - y_fit) + run.LAMBDA * w
        hess = (tx_fit.T * (p * (1 - p))) @ tx_fit + run.LAMBDA * np.eye(len(w))
        w -= np.linalg.solve(hess, grad)
    return tx_fit @ w, tx_apply @ w


def evaluate(name, scores_fn, x, y, folds, config, reference=None):
    """Cross-validate scores_fn(x_fit, y_fit, x_apply); print, save and return it."""
    res, _ = cv.cross_validate(lambda tr, te: scores_fn(x[tr], y[tr], x[te]), y, folds)
    print(f"{name}:", cv.summary(res, reference))
    cv.save_result(name, config, res)
    return res


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths are relative to here
    x, _, y, _ = features.load_data(run.DATA_DIR)
    folds = cv.stratified_folds(y)
    config = {"lambda": run.LAMBDA, "gamma": run.GAMMA, "max_iters": run.MAX_ITERS}
    ref = evaluate("logistic", run.logistic_scores, x, y, folds, config)
    config = {"lambda": run.LAMBDA, "iters": NEWTON_ITERS}
    evaluate("newton", newton_scores, x, y, folds, config, ref)


if __name__ == "__main__":
    main()
