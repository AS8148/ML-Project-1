"""Unit tests for implementations.py.

Run from the repository root with:  pytest tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import implementations as impl  # noqa: E402

RTOL = 1e-4
ATOL = 1e-8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small():
    """Tiny dataset from the official public tests (known expected values)."""
    y = np.array([0.1, 0.3, 0.5])
    tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
    initial_w = np.array([0.5, 1.0])
    return y, tx, initial_w


@pytest.fixture
def regression_data():
    """Random linear regression problem y = tx @ w_true + small noise."""
    rng = np.random.default_rng(0)
    n, d = 200, 4
    tx = np.c_[np.ones(n), rng.normal(size=(n, d - 1))]
    w_true = np.array([1.0, -2.0, 0.5, 3.0])
    y = tx @ w_true + 0.01 * rng.normal(size=n)
    return y, tx, w_true


@pytest.fixture
def classification_data():
    """Random, non-separable binary classification problem with labels in {0, 1}."""
    rng = np.random.default_rng(1)
    n, d = 300, 3
    tx = np.c_[np.ones(n), rng.normal(size=(n, d - 1))]
    w_true = np.array([0.3, 1.5, -1.0])
    p = 1 / (1 + np.exp(-tx @ w_true))
    y = (rng.uniform(size=n) < p).astype(float)
    return y, tx


def numerical_gradient(f, w, eps=1e-6):
    """Central finite-difference approximation of the gradient of f at w."""
    grad = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = eps
        grad[i] = (f(w + e) - f(w - e)) / (2 * eps)
    return grad


# ---------------------------------------------------------------------------
# Reference values
# ---------------------------------------------------------------------------


def test_reference_values(small):
    y, tx, w0 = small
    yb = (y > 0.2) * 1.0
    cases = [
        (
            impl.mean_squared_error_gd(y, tx, w0, 2, 0.1),
            [-0.050586, 0.203718],
            0.051534,
        ),
        (
            impl.mean_squared_error_sgd(y[:1], tx[:1], w0, 2, 0.1),
            [0.063058, 0.39208],
            0.844595,
        ),
        (impl.least_squares(y, tx), [0.218786, -0.053837], 0.026942),
        (impl.ridge_regression(y, tx, 0.0), [0.218786, -0.053837], 0.026942),
        (impl.ridge_regression(y, tx, 1.0), [0.054303, 0.042713], 0.03175),
        (impl.logistic_regression(yb, tx, w0, 2, 0.1), [0.378561, 0.801131], 1.348358),
        (
            impl.reg_logistic_regression(yb, tx, 1.0, w0, 2, 0.1),
            [0.216062, 0.467747],
            0.972165,
        ),
    ]
    for (w, loss), exp_w, exp_loss in cases:
        np.testing.assert_allclose(w, exp_w, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(loss, exp_loss, rtol=RTOL, atol=ATOL)


def test_zero_steps_return_initial_w_and_its_loss(small):
    y, tx, _ = small
    yb = (y > 0.2) * 1.0
    w0 = np.array([0.413044, 0.875757])
    w, loss = impl.mean_squared_error_gd(y, tx, w0, 0, 0.1)
    np.testing.assert_allclose(w, w0)
    np.testing.assert_allclose(loss, 2.959836, rtol=RTOL)

    w0 = np.array([0.463156, 0.939874])
    w, loss = impl.logistic_regression(yb, tx, w0, 0, 0.1)
    np.testing.assert_allclose(w, w0)
    np.testing.assert_allclose(loss, 1.533694, rtol=RTOL)

    w0 = np.array([0.409111, 0.843996])
    w, loss = impl.reg_logistic_regression(yb, tx, 1.0, w0, 0, 0.1)
    np.testing.assert_allclose(w, w0)
    np.testing.assert_allclose(loss, 1.407327, rtol=RTOL)


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


def test_output_shapes_and_no_mutation(small):
    y, tx, w0 = small
    yb = (y > 0.2) * 1.0
    w0_copy = w0.copy()
    results = [
        impl.mean_squared_error_gd(y, tx, w0, 5, 0.1),
        impl.mean_squared_error_sgd(y, tx, w0, 5, 0.1),
        impl.least_squares(y, tx),
        impl.ridge_regression(y, tx, 0.1),
        impl.logistic_regression(yb, tx, w0, 5, 0.1),
        impl.reg_logistic_regression(yb, tx, 0.1, w0, 5, 0.1),
    ]
    for w, loss in results:
        assert w.shape == (2,)  # 1D array, not (2, 1)
        assert np.ndim(loss) == 0
    # The caller's initial_w must not be modified in place
    np.testing.assert_array_equal(w0, w0_copy)


# ---------------------------------------------------------------------------
# Linear regression
# ---------------------------------------------------------------------------


def test_mse_gradient_matches_finite_differences(regression_data):
    y, tx, _ = regression_data
    w = np.array([0.1, 0.2, -0.3, 0.4])
    num = numerical_gradient(lambda v: impl.compute_mse(y, tx, v), w)
    np.testing.assert_allclose(impl.compute_mse_gradient(y, tx, w), num, rtol=1e-5)


def test_least_squares_recovers_true_weights(regression_data):
    y, tx, w_true = regression_data
    w, loss = impl.least_squares(y, tx)
    np.testing.assert_allclose(w, w_true, atol=5e-3)
    # Optimality: gradient of the MSE is zero at the solution
    np.testing.assert_allclose(impl.compute_mse_gradient(y, tx, w), 0, atol=1e-10)
    # Loss is 1/2 * noise variance (~0.5 * 1e-4)
    assert loss < 1e-4


def test_least_squares_exact_on_noiseless_data():
    tx = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = 1.0 + 2.0 * tx[:, 1]
    w, loss = impl.least_squares(y, tx)
    np.testing.assert_allclose(w, [1.0, 2.0])
    np.testing.assert_allclose(loss, 0.0, atol=1e-20)


def test_gd_converges_to_least_squares(regression_data):
    y, tx, _ = regression_data
    w_ls, loss_ls = impl.least_squares(y, tx)
    w, loss = impl.mean_squared_error_gd(y, tx, np.zeros(4), 500, 0.5)
    np.testing.assert_allclose(w, w_ls, atol=1e-6)
    np.testing.assert_allclose(loss, loss_ls, rtol=1e-6)


def test_gd_loss_decreases(regression_data):
    y, tx, _ = regression_data
    losses = [
        impl.mean_squared_error_gd(y, tx, np.zeros(4), k, 0.1)[1] for k in range(10)
    ]
    assert all(a > b for a, b in zip(losses, losses[1:]))


def test_sgd_approaches_least_squares(regression_data):
    y, tx, _ = regression_data
    np.random.seed(42)
    w_ls, loss_ls = impl.least_squares(y, tx)
    w, loss = impl.mean_squared_error_sgd(y, tx, np.zeros(4), 5000, 0.01)
    np.testing.assert_allclose(w, w_ls, atol=0.05)
    assert loss < loss_ls + 1e-3


def test_sgd_uses_single_sample_step():
    """One SGD step must equal a GD step computed on the single sampled point."""
    y = np.array([1.0, 2.0])
    tx = np.array([[1.0, 2.0], [3.0, 4.0]])
    w0 = np.zeros(2)
    np.random.seed(0)
    i = np.random.randint(2)
    np.random.seed(0)
    w, _ = impl.mean_squared_error_sgd(y, tx, w0, 1, 0.1)
    expected = w0 + 0.1 * (y[i] - tx[i] @ w0) * tx[i]
    np.testing.assert_allclose(w, expected)


def test_ridge_lambda0_equals_least_squares(regression_data):
    y, tx, _ = regression_data
    w_r, l_r = impl.ridge_regression(y, tx, 0.0)
    w_ls, l_ls = impl.least_squares(y, tx)
    np.testing.assert_allclose(w_r, w_ls)
    np.testing.assert_allclose(l_r, l_ls)


def test_ridge_is_stationary_point_of_objective(regression_data):
    y, tx, _ = regression_data
    lam = 0.3
    w, loss = impl.ridge_regression(y, tx, lam)
    # grad of 1/(2N)||y - tx w||^2 + lam ||w||^2 must vanish
    grad = impl.compute_mse_gradient(y, tx, w) + 2 * lam * w
    np.testing.assert_allclose(grad, 0, atol=1e-10)
    # Returned loss excludes the penalty
    np.testing.assert_allclose(loss, impl.compute_mse(y, tx, w))


def test_ridge_shrinks_weights(regression_data):
    y, tx, _ = regression_data
    norms = [
        np.linalg.norm(impl.ridge_regression(y, tx, lam)[0])
        for lam in [0, 0.01, 0.1, 1, 10]
    ]
    assert all(a > b for a, b in zip(norms, norms[1:]))


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------


def test_sigmoid_is_stable_and_correct():
    t = np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0])
    s = impl.sigmoid(t)
    assert np.all(np.isfinite(s))
    np.testing.assert_allclose(
        s, [0.0, 1 / (1 + np.exp(5)), 0.5, 1 / (1 + np.exp(-5)), 1.0]
    )


def test_logistic_loss_is_stable_for_large_margins():
    tx = np.array([[1000.0], [-1000.0]])
    y = np.array([1.0, 0.0])
    loss = impl.compute_logistic_loss(y, tx, np.array([1.0]))
    assert np.isfinite(loss) and loss < 1e-10


def test_logistic_loss_at_zero_is_log2(classification_data):
    y, tx = classification_data
    np.testing.assert_allclose(
        impl.compute_logistic_loss(y, tx, np.zeros(3)), np.log(2)
    )


def test_logistic_gradient_matches_finite_differences(classification_data):
    y, tx = classification_data
    w = np.array([0.2, -0.4, 0.7])
    num = numerical_gradient(lambda v: impl.compute_logistic_loss(y, tx, v), w)
    np.testing.assert_allclose(impl.compute_logistic_gradient(y, tx, w), num, rtol=1e-5)


def test_logistic_regression_converges(classification_data):
    y, tx = classification_data
    w, loss = impl.logistic_regression(y, tx, np.zeros(3), 3000, 1.0)
    # At the optimum the gradient of the loss vanishes
    np.testing.assert_allclose(impl.compute_logistic_gradient(y, tx, w), 0, atol=1e-6)
    assert loss < np.log(2)


def test_logistic_regression_loss_decreases(classification_data):
    y, tx = classification_data
    losses = [
        impl.logistic_regression(y, tx, np.zeros(3), k, 0.5)[1] for k in range(10)
    ]
    assert all(a > b for a, b in zip(losses, losses[1:]))


def test_reg_logistic_lambda0_equals_logistic(classification_data):
    y, tx = classification_data
    w0 = np.array([0.1, 0.1, 0.1])
    w_a, l_a = impl.reg_logistic_regression(y, tx, 0.0, w0, 50, 0.5)
    w_b, l_b = impl.logistic_regression(y, tx, w0, 50, 0.5)
    np.testing.assert_allclose(w_a, w_b)
    np.testing.assert_allclose(l_a, l_b)


def test_reg_logistic_converges_to_regularized_optimum(classification_data):
    y, tx = classification_data
    lam = 0.05
    w, loss = impl.reg_logistic_regression(y, tx, lam, np.zeros(3), 3000, 1.0)
    grad = impl.compute_logistic_gradient(y, tx, w) + 2 * lam * w
    np.testing.assert_allclose(grad, 0, atol=1e-6)
    # Returned loss excludes the penalty term
    np.testing.assert_allclose(loss, impl.compute_logistic_loss(y, tx, w))
    # Regularization shrinks the weights compared to the unregularized solution
    w_unreg, _ = impl.logistic_regression(y, tx, np.zeros(3), 3000, 1.0)
    assert np.linalg.norm(w) < np.linalg.norm(w_unreg)
