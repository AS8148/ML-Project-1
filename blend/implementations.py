"""Implementations of the ML methods required for Project 1.

All functions take 1D arrays for vectors (shape (N,) / (D,)) and return a
tuple (w, loss), where w is the final weight vector and loss is the value of
the (unregularized) cost function at w.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Helper functions: losses and gradients
# ---------------------------------------------------------------------------


def compute_mse(y, tx, w):
    """Compute the mean squared error loss L(w) = 1/(2N) * sum((y - tx @ w)^2).

    Args:
        y: numpy array of shape (N,), targets.
        tx: numpy array of shape (N, D), features.
        w: numpy array of shape (D,), weights.

    Returns:
        Scalar (0-d numpy float), the MSE loss.
    """
    e = y - tx @ w
    return np.mean(e**2) / 2


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the MSE loss: -1/N * tx^T (y - tx @ w).

    Args:
        y: numpy array of shape (N,), targets.
        tx: numpy array of shape (N, D), features.
        w: numpy array of shape (D,), weights.

    Returns:
        numpy array of shape (D,), the gradient of the MSE at w.
    """
    e = y - tx @ w
    return -tx.T @ e / len(y)


def sigmoid(t):
    """Numerically stable logistic function sigma(t) = 1 / (1 + exp(-t)).

    Args:
        t: scalar or numpy array.

    Returns:
        sigma(t), same shape as t.
    """
    # exp(-logaddexp(0, -t)) = 1 / (1 + exp(-t)) without overflow for large |t|
    return np.exp(-np.logaddexp(0, -t))


def compute_logistic_loss(y, tx, w):
    """Compute the logistic (negative log-likelihood) loss, averaged over samples.

    L(w) = 1/N * sum( log(1 + exp(x_n^T w)) - y_n * x_n^T w ),  with y_n in {0, 1}.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), features.
        w: numpy array of shape (D,), weights.

    Returns:
        Scalar (0-d numpy float), the logistic loss.
    """
    z = tx @ w
    # np.logaddexp(0, z) = log(1 + exp(z)) computed stably
    return np.mean(np.logaddexp(0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of the logistic loss: 1/N * tx^T (sigma(tx @ w) - y).

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), features.
        w: numpy array of shape (D,), weights.

    Returns:
        numpy array of shape (D,), the gradient of the logistic loss at w.
    """
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


# ---------------------------------------------------------------------------
# Required methods
# ---------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using (full) gradient descent on the MSE loss.

    Args:
        y: numpy array of shape (N,), targets.
        tx: numpy array of shape (N, D), features.
        initial_w: numpy array of shape (D,), initial weights.
        max_iters: int, number of gradient steps.
        gamma: float, step size.

    Returns:
        w: numpy array of shape (D,), last weight vector.
        loss: scalar, MSE loss at the last w.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (mini-batch size 1).

    At each iteration one sample is drawn uniformly at random and the weights
    are updated with the gradient of the MSE on that single sample.

    Args:
        y: numpy array of shape (N,), targets.
        tx: numpy array of shape (N, D), features.
        initial_w: numpy array of shape (D,), initial weights.
        max_iters: int, number of SGD steps.
        gamma: float, step size.

    Returns:
        w: numpy array of shape (D,), last weight vector.
        loss: scalar, MSE loss on the full dataset at the last w.
    """
    w = np.array(initial_w, dtype=float)
    n = len(y)
    for _ in range(max_iters):
        i = np.random.randint(n)
        # Slicing with i:i+1 keeps the (1,) / (1, D) shapes for the gradient helper
        w = w - gamma * compute_mse_gradient(y[i : i + 1], tx[i : i + 1], w)
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Least squares regression using the normal equations tx^T tx w = tx^T y.

    Args:
        y: numpy array of shape (N,), targets.
        tx: numpy array of shape (N, D), features.

    Returns:
        w: numpy array of shape (D,), optimal weights.
        loss: scalar, MSE loss at w.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Minimizes 1/(2N) ||y - tx w||^2 + lambda_ ||w||^2, whose solution is
    (tx^T tx + 2 N lambda_ I) w = tx^T y.

    Args:
        y: numpy array of shape (N,), targets.
        tx: numpy array of shape (N, D), features.
        lambda_: float, regularization parameter.

    Returns:
        w: numpy array of shape (D,), optimal weights.
        loss: scalar, MSE loss at w (without the penalty term).
    """
    n, d = tx.shape
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    w = np.linalg.solve(a, tx.T @ y)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), features.
        initial_w: numpy array of shape (D,), initial weights.
        max_iters: int, number of gradient steps.
        gamma: float, step size.

    Returns:
        w: numpy array of shape (D,), last weight vector.
        loss: scalar, logistic loss at the last w.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y in {0, 1}).

    Minimizes L(w) + lambda_ ||w||^2, where L is the logistic loss. The
    gradient of the penalty is 2 * lambda_ * w.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), features.
        lambda_: float, regularization parameter.
        initial_w: numpy array of shape (D,), initial weights.
        max_iters: int, number of gradient steps.
        gamma: float, step size.

    Returns:
        w: numpy array of shape (D,), last weight vector.
        loss: scalar, logistic loss at the last w (without the penalty term).
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)
