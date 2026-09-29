"""Unit tests for data_cleaning.py and cross_validation.py.

Run from the repository root with:  pytest tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cross_validation as cv  # noqa: E402
import data_cleaning as dc  # noqa: E402

# ---------------------------------------------------------------------------
# replace_missing_codes / drop_columns
# ---------------------------------------------------------------------------


def test_missing_codes_generic_rules():
    columns = np.array(["YESNO", "EMPLOY1", "COUNT99", "CODE999", "PHYSHLTH", "FRAC"])
    x = np.array(
        [
            [1, 1, 5, 101, 88, 0.5],
            [2, 6, 77, 777, 3, 99.0],
            [7, 7, 99, 888, 77, 1.25],
            [9, 9, 12, 999, 99, np.nan],
        ],
        dtype=float,
    )
    out = dc.replace_missing_codes(x, columns)
    # max 9 and no 6: both 7 and 9 are missing
    np.testing.assert_array_equal(out[:, 0], [1, 2, np.nan, np.nan])
    # max 9 with 6 present: 7 is a real answer (retired), only 9 is missing
    np.testing.assert_array_equal(out[:, 1], [1, 6, 7, np.nan])
    # max 99: 77 and 99 are missing
    np.testing.assert_array_equal(out[:, 2], [5, np.nan, np.nan, 12])
    # max 999: 777/999 missing, 888 means zero
    np.testing.assert_array_equal(out[:, 3], [101, np.nan, 0, np.nan])
    # 88 = "none" -> 0 days
    np.testing.assert_array_equal(out[:, 4], [0, 3, np.nan, np.nan])
    # fractional (calculated) columns are left untouched
    np.testing.assert_array_equal(out[:, 5], x[:, 5])
    # input is not modified
    assert x[2, 0] == 7


def test_missing_codes_explicit_rules():
    columns = np.array(["_AGEG5YR", "DIABAGE2"])
    x = np.array([[1, 77], [13, 98], [14, 99]], dtype=float)
    out = dc.replace_missing_codes(x, columns)
    np.testing.assert_array_equal(out[:, 0], [1, 13, np.nan])
    np.testing.assert_array_equal(out[:, 1], [77, np.nan, np.nan])  # 77 is an age here


def test_drop_columns():
    columns = np.array(["SEQNO", "GENHLTH", "WEIGHT2", "SEX"])
    x = np.arange(8).reshape(2, 4).astype(float)
    x2, c2 = dc.drop_columns(x, columns)
    np.testing.assert_array_equal(c2, ["GENHLTH", "SEX"])
    np.testing.assert_array_equal(x2, x[:, [1, 3]])


# ---------------------------------------------------------------------------
# fit_cleaner / transform
# ---------------------------------------------------------------------------


@pytest.fixture
def toy():
    """Train/val data with a mostly-missing column, a constant one and a nominal one."""
    rng = np.random.default_rng(0)
    n = 200
    columns = np.array(["A", "MOSTLY_NAN", "CONST", "MARITAL", "B"])
    x = np.c_[
        rng.normal(10, 2, n),
        np.where(rng.uniform(size=n) < 0.8, np.nan, 1.0),
        np.ones(n),
        rng.integers(1, 4, n).astype(float),
        rng.normal(size=n),
    ]
    x[::10, 0] = np.nan  # 10% missing in A
    return x, columns


def test_fit_transform_drops_imputes_and_standardizes(toy):
    x, columns = toy
    params = dc.fit_cleaner(x, columns, max_nan_frac=0.5, clip_quantiles=None)
    tx = dc.transform(x, params)

    names = list(params["feature_names"])
    assert "MOSTLY_NAN" not in names  # 80% missing > 50%
    assert "CONST" not in names  # zero variance
    assert names == ["A", "B", "MARITAL=1", "MARITAL=2", "MARITAL=3"]

    np.testing.assert_array_equal(tx[:, 0], 1.0)  # bias column
    assert not np.isnan(tx).any()
    np.testing.assert_allclose(tx[:, 1:].mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(tx[:, 1:].std(axis=0), 1, atol=1e-10)


def test_one_hot_encoding(toy):
    x, columns = toy
    params = dc.fit_cleaner(x, columns, clip_quantiles=None)
    z = dc._build_features(x, params)[:, params["keep_features"]]
    onehot = z[:, 2:]
    np.testing.assert_array_equal(onehot.sum(axis=1), 1)
    np.testing.assert_array_equal(np.argmax(onehot, axis=1) + 1, x[:, 3])


def test_transform_uses_only_training_statistics(toy):
    """Validation data must be processed with train medians/means (no leakage)."""
    x, columns = toy
    params = dc.fit_cleaner(x, columns, clip_quantiles=None)
    x_val = x[:3].copy()
    x_val[:, 0] = [np.nan, 1000.0, 1000.0]  # extreme values must not change stats
    tx_val = dc.transform(x_val, params, add_bias=False)
    median_a = np.nanmedian(x[:, 0])
    expected_first = (median_a - params["mean"][0]) / params["std"][0]
    np.testing.assert_allclose(tx_val[0, 0], expected_first)
    np.testing.assert_allclose(
        tx_val[1, 0], (1000.0 - params["mean"][0]) / params["std"][0]
    )


def test_clipping_and_nan_indicators():
    columns = np.array(["CONT"])
    x = np.r_[np.arange(100.0), [np.nan, 1e6]][:, None]
    params = dc.fit_cleaner(
        x, columns, clip_quantiles=(0.0, 0.99), add_nan_indicators=True
    )
    assert list(params["feature_names"]) == ["CONT", "CONT_isnan"]
    z = dc._build_features(x, params)
    assert z[-1, 0] == params["upper"][0] < 1e6  # outlier clipped
    np.testing.assert_array_equal(z[:, 1], np.r_[np.zeros(100), [1, 0]])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    assert cv.confusion_counts(y_true, y_pred) == (2, 1, 1, 4)
    assert cv.accuracy(y_true, y_pred) == 6 / 8
    # precision = recall = 2/3 -> F1 = 2/3
    np.testing.assert_allclose(cv.f1_score(y_true, y_pred), 2 / 3)
    assert cv.f1_score(y_true, np.zeros(8)) == 0.0


def test_best_threshold_separable_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    thr, f1 = cv.best_threshold(y, scores, candidates=np.linspace(0, 1, 101))
    assert f1 == 1.0 and 0.3 < thr <= 0.7


# ---------------------------------------------------------------------------
# Splits and cross-validation
# ---------------------------------------------------------------------------


def test_stratified_folds_partition_and_balance():
    y = np.r_[np.ones(100), np.zeros(900)]
    folds = cv.stratified_k_indices(y, 5, seed=3)
    all_idx = np.sort(np.concatenate(folds))
    np.testing.assert_array_equal(all_idx, np.arange(1000))  # partition
    for f in folds:
        assert len(f) == 200
        assert y[f].sum() == 20  # same 10% positives in each fold


def test_train_val_split():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=1000) < 0.1).astype(float)
    x = np.arange(1000)[:, None].astype(float)
    x_tr, x_val, y_tr, y_val = cv.train_val_split(y, x, val_ratio=0.2)
    assert abs(len(x_val) - 200) <= 1 and len(x_tr) + len(x_val) == 1000
    assert set(x_tr[:, 0]).isdisjoint(x_val[:, 0])
    np.testing.assert_allclose(y_val.mean(), y.mean(), atol=0.01)


def test_cross_validate_on_easy_problem():
    """A linear classifier must get near-perfect CV scores on separable data."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + 0.5 * x[:, 1] > 1).astype(float)
    columns = np.array(["F0", "F1", "F2"])

    def train_fn(y, tx):
        return np.linalg.solve(tx.T @ tx, tx.T @ y)

    res = cv.cross_validate(
        y, x, columns, train_fn, lambda tx, w: tx @ w, k_fold=4, cleaner_kwargs={}
    )
    assert res["f1"].shape == (4,)
    assert res["f1"].mean() > 0.9
    assert res["accuracy"].mean() > 0.95
    assert res["oof_scores"].shape == (n,)
