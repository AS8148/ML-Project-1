"""Unit tests for models.py.

Run from the repository root with:  pytest tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import implementations as impl  # noqa: E402
import models  # noqa: E402


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(1)
    n = 2000
    tx = np.c_[np.ones(n), rng.normal(size=(n, 3))]
    w_true = np.array([-1.0, 1.5, -1.0, 0.5])
    y = (rng.uniform(size=n) < models.sigmoid(tx @ w_true)).astype(float)
    return y, tx, w_true


def test_newton_matches_gradient_descent_optimum(classification_data):
    y, tx, _ = classification_data
    w_newton = models.logistic_newton(y, tx, lambda_=0.0)
    w_gd, _ = impl.logistic_regression(y, tx, np.zeros(4), 5000, 1.0)
    np.testing.assert_allclose(w_newton, w_gd, atol=1e-5)
    # Stationarity: the gradient of the loss vanishes
    np.testing.assert_allclose(
        impl.compute_logistic_gradient(y, tx, w_newton), 0, atol=1e-7
    )


def test_newton_recovers_true_weights(classification_data):
    y, tx, w_true = classification_data
    w = models.logistic_newton(y, tx)
    np.testing.assert_allclose(w, w_true, atol=0.25)


def test_newton_regularization_does_not_penalize_bias(classification_data):
    y, tx, _ = classification_data
    w = models.logistic_newton(y, tx, lambda_=10.0)
    assert np.all(np.abs(w[1:]) < 0.1)  # weights shrunk to ~0
    # bias alone must still match the class frequency
    np.testing.assert_allclose(models.sigmoid(w[0]), y.mean(), atol=0.02)


def test_newton_pos_weight_increases_predicted_probabilities(classification_data):
    y, tx, _ = classification_data
    p1 = models.sigmoid(tx @ models.logistic_newton(y, tx, pos_weight=1.0))
    p3 = models.sigmoid(tx @ models.logistic_newton(y, tx, pos_weight=3.0))
    assert p3.mean() > p1.mean()


def test_mlp_learns_nonlinear_problem():
    """XOR-like labels: a linear model is at chance, the MLP is not."""
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 2))
    y = ((x[:, 0] * x[:, 1]) > 0).astype(float)
    tx = np.c_[np.ones(n), x]
    params = models.train_mlp(y, tx, hidden=16, epochs=30, lr=1e-2, seed=0)
    acc_mlp = np.mean((models.predict_mlp(tx, params) > 0.5) == y)
    w = models.logistic_newton(y, tx)
    acc_lin = np.mean((models.sigmoid(tx @ w) > 0.5) == y)
    assert acc_mlp > 0.95 and acc_lin < 0.7


@pytest.mark.parametrize("hidden", [5, (6, 4)])
def test_mlp_gradients_match_finite_differences(hidden):
    rng = np.random.default_rng(0)
    xb = rng.normal(size=(20, 3))
    yb = (rng.uniform(size=20) < 0.4).astype(float)
    sb = np.where(yb == 1, 2.0, 1.0)
    params = models.train_mlp(yb, xb, hidden=hidden, epochs=1, batch_size=20)
    _, grads = models.mlp_loss_and_grads(params, xb, yb, sb, l2=0.1)
    eps = 1e-6
    for key, value in params.items():
        num = np.zeros(np.shape(value))
        for i in np.ndindex(np.shape(value)):
            plus = {k: np.array(v, dtype=float) for k, v in params.items()}
            minus = {k: np.array(v, dtype=float) for k, v in params.items()}
            plus[key][i] += eps
            minus[key][i] -= eps
            lp, _ = models.mlp_loss_and_grads(plus, xb, yb, sb, l2=0.1)
            lm, _ = models.mlp_loss_and_grads(minus, xb, yb, sb, l2=0.1)
            num[i] = (lp - lm) / (2 * eps)
        np.testing.assert_allclose(grads[key], num, atol=1e-6, err_msg=key)


def test_deeper_mlp_with_dropout_learns_nonlinear_problem():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 2))
    y = ((x[:, 0] * x[:, 1]) > 0).astype(float)
    tx = np.c_[np.ones(n), x]
    params = models.train_mlp(
        y, tx, hidden=(32, 16), epochs=30, lr=1e-2, dropout=0.1, seed=0
    )
    assert np.mean((models.predict_mlp(tx, params) > 0.5) == y) > 0.95


def test_expansion_shapes_and_scaling():
    rng = np.random.default_rng(0)
    n = 500
    tx = np.c_[np.ones(n), rng.normal(size=(n, 4))]
    y = (tx[:, 1] + tx[:, 2] > 0).astype(float)
    is_cont = np.array([True, False, True, False])
    exp = models.fit_expansion(tx, y, is_cont, squares=True, n_interact=3)
    out = models.apply_expansion(tx, exp)
    # 4 original + bias, 2 squares, 3 pairs
    assert out.shape == (n, 5 + 2 + 3)
    np.testing.assert_allclose(out[:, 5:].mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(out[:, 5:].std(axis=0), 1, atol=1e-6)
    # the two features used to build y are among the interacting ones
    used = {i for pair in exp["pairs"] for i in pair}
    assert {1, 2} <= used
