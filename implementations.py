"""The six methods required by Project 1, implemented with NumPy only.

Shapes: y (N,), tx (N, D), w and initial_w (D,). Every method returns (w, loss):
the final weights and the loss at those weights. Linear models use the MSE with
a factor 1/2; logistic models expect y in {0, 1}. Regularized methods return the
loss without the penalty term.
"""

import numpy as np


def _mse_loss(y, tx, w):
    """Return the MSE loss 1/(2N) * ||y - tx w||^2."""
    e = y - tx @ w
    return (e @ e) / (2 * len(y))


def _mse_gradient(y, tx, w):
    """Return the gradient of the MSE loss with respect to w."""
    return tx.T @ (tx @ w - y) / len(y)


def _sigmoid(z):
    """Return 1 / (1 + exp(-z)) without overflow for large |z|."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0, e) / (1.0 + e)


def _logistic_loss(y, tx, w):
    """Return the mean of log(1 + exp(z)) - y z, with z = tx w."""
    z = tx @ w
    return np.mean(np.logaddexp(0.0, z) - y * z)


def _logistic_gradient(y, tx, w):
    """Return the gradient of the logistic loss with respect to w."""
    return tx.T @ (_sigmoid(tx @ w) - y) / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression with gradient descent on the MSE loss."""
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w -= gamma * _mse_gradient(y, tx, w)
    return w, _mse_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression with stochastic gradient descent (batch size 1).

    Samples are drawn uniformly with np.random, so call np.random.seed
    beforehand for reproducible results. The loss is on all samples.
    """
    w = np.array(initial_w, dtype=float)
    for i in np.random.randint(len(y), size=max_iters):
        w -= gamma * _mse_gradient(y[i : i + 1], tx[i : i + 1], w)
    return w, _mse_loss(y, tx, w)


def least_squares(y, tx):
    """Least squares regression with the normal equations."""
    return ridge_regression(y, tx, 0.0)


def ridge_regression(y, tx, lambda_):
    """Ridge regression with the normal equations.

    Minimizes MSE(w) + lambda_ * ||w||^2 by solving
    (tx^T tx + 2 N lambda_ I) w = tx^T y. The returned loss has no penalty.
    """
    n, d = tx.shape
    w = np.linalg.solve(tx.T @ tx + 2 * n * lambda_ * np.eye(d), tx.T @ y)
    return w, _mse_loss(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression with gradient descent."""
    return reg_logistic_regression(y, tx, 0.0, initial_w, max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression with gradient descent.

    Minimizes the logistic loss + lambda_ * ||w||^2, so the penalty adds
    2 * lambda_ * w to the gradient. The returned loss has no penalty.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w -= gamma * (_logistic_gradient(y, tx, w) + 2 * lambda_ * w)
    return w, _logistic_loss(y, tx, w)
