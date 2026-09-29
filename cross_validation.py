"""Local validation: hold-out split, stratified k-fold cross-validation, metrics.

The cleaning statistics (medians, quantiles, means/stds) are re-fitted on the
training part of every split, so the validation score is an honest estimate of
the test score.
"""

import numpy as np

from data_cleaning import fit_cleaner, transform

# ---------------------------------------------------------------------------
# Metrics (labels in {0, 1})
# ---------------------------------------------------------------------------


def confusion_counts(y_true, y_pred):
    """Return (tp, fp, fn, tn) for binary labels in {0, 1}."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    return tp, fp, fn, tn


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    return np.mean(y_true == y_pred)


def f1_score(y_true, y_pred):
    """F1 score of the positive class: 2 * precision * recall / (precision + recall)."""
    tp, fp, fn, _ = confusion_counts(y_true, y_pred)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def best_threshold(y_true, scores, candidates=None):
    """Threshold on `scores` that maximizes the F1 score on (y_true, scores).

    Args:
        y_true: (N,) labels in {0, 1}.
        scores: (N,) predicted scores (e.g. probabilities).
        candidates: thresholds to try, default: 99 quantiles of the scores.

    Returns:
        (threshold, f1) of the best candidate.
    """
    if candidates is None:
        candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    f1s = [f1_score(y_true, (scores >= t).astype(int)) for t in candidates]
    i = int(np.argmax(f1s))
    return candidates[i], f1s[i]


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def stratified_k_indices(y, k_fold, seed=1):
    """Split sample indices into k folds that each keep the class proportions.

    With only ~9% positives, a plain random split can give folds with noticeably
    different class balance; stratifying removes this source of noise.

    Args:
        y: (N,) labels.
        k_fold: number of folds.
        seed: random seed.

    Returns:
        list of k arrays of indices (a partition of range(N)).
    """
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k_fold)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for f, part in enumerate(np.array_split(idx, k_fold)):
            folds[f].append(part)
    return [rng.permutation(np.concatenate(f)) for f in folds]


def train_val_split(y, x, val_ratio=0.2, seed=1):
    """Stratified hold-out split.

    Args:
        y: (N,) labels.
        x: (N, D) features.
        val_ratio: fraction of samples used for validation.
        seed: random seed.

    Returns:
        x_train, x_val, y_train, y_val
    """
    rng = np.random.default_rng(seed)
    val_idx, train_idx = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_val = int(round(val_ratio * len(idx)))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    val_idx = rng.permutation(np.concatenate(val_idx))
    train_idx = rng.permutation(np.concatenate(train_idx))
    return x[train_idx], x[val_idx], y[train_idx], y[val_idx]


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate(
    y, x, columns, train_fn, predict_fn, k_fold=5, seed=1, cleaner_kwargs=None
):
    """Stratified k-fold cross-validation, re-fitting the cleaner in every fold.

    For each fold, the decision threshold is chosen to maximize F1 on the
    *training* part, then applied to the validation part.

    Args:
        y: (N,) labels in {0, 1}.
        x: (N, D) features after `replace_missing_codes` (not yet fitted/scaled).
        columns: (D,) column names.
        train_fn: function (y, tx) -> w.
        predict_fn: function (tx, w) -> scores (higher = more likely positive).
        k_fold: number of folds.
        seed: random seed of the split.
        cleaner_kwargs: dict of keyword arguments for `fit_cleaner`.

    Returns:
        dict with per-fold arrays "f1", "accuracy", "threshold",
        "f1_at_0.5" (fixed threshold 0.5, for reference), and
        "oof_scores": out-of-fold scores for every sample.
    """
    cleaner_kwargs = cleaner_kwargs or {}
    folds = stratified_k_indices(y, k_fold, seed)
    res = {"f1": [], "accuracy": [], "threshold": [], "f1_at_0.5": []}
    oof = np.zeros(len(y))
    for f in range(k_fold):
        val_idx = folds[f]
        tr_idx = np.concatenate([folds[i] for i in range(k_fold) if i != f])

        params = fit_cleaner(x[tr_idx], columns, **cleaner_kwargs)
        tx_tr = transform(x[tr_idx], params)
        tx_val = transform(x[val_idx], params)

        w = train_fn(y[tr_idx], tx_tr)
        thr, _ = best_threshold(y[tr_idx], predict_fn(tx_tr, w))
        s_val = predict_fn(tx_val, w)
        pred = (s_val >= thr).astype(int)
        oof[val_idx] = s_val

        res["f1"].append(f1_score(y[val_idx], pred))
        res["accuracy"].append(accuracy(y[val_idx], pred))
        res["threshold"].append(thr)
        res["f1_at_0.5"].append(f1_score(y[val_idx], (s_val >= 0.5).astype(int)))

    res = {key: np.array(v) for key, v in res.items()}
    res["oof_scores"] = oof
    return res
