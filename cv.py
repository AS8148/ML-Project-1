"""Stratified k-fold cross-validation with an F1-optimal decision threshold."""

import json
import os
import subprocess
import time

import numpy as np


def stratified_folds(y, k=5, seed=42):
    """Return the fold index in [0, k) of every sample, stratified by class."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    return folds


def f1_accuracy(y, pred):
    """Return the F1 of the positive class and the accuracy (y, pred in {0, 1})."""
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    return 2 * tp / (2 * tp + fp + fn), np.mean(pred == y)


def best_threshold(y, scores):
    """Return the threshold t maximizing the F1 of the rule scores >= t."""
    order = np.argsort(-scores, kind="stable")
    s, ys = scores[order], y[order]
    tp = np.cumsum(ys)
    f1 = 2 * tp / (np.arange(1, len(ys) + 1) + ys.sum())  # 2tp / (tp+fp + tp+fn)
    f1[:-1][s[1:] == s[:-1]] = -1  # never split tied scores
    return s[np.argmax(f1)]


def cross_validate(fit_score, y, folds):
    """Cross-validate fit_score(train_idx, test_idx) -> (train_scores, test_scores).

    In each fold the threshold is chosen on the training scores only.
    Returns per-fold F1, accuracy, threshold and fit time, and out-of-fold scores.
    """
    res = {"f1": [], "accuracy": [], "threshold": [], "seconds": []}
    oof = np.zeros(len(y))
    for k in range(folds.max() + 1):
        train, test = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        start = time.time()
        s_train, s_test = fit_score(train, test)
        res["seconds"].append(time.time() - start)
        t = best_threshold(y[train], s_train)
        f1, acc = f1_accuracy(y[test], s_test >= t)
        res["f1"].append(f1)
        res["accuracy"].append(acc)
        res["threshold"].append(t)
        oof[test] = s_test
    return res, oof


def summary(res, reference=None):
    """Return mean +- std of F1 and accuracy, and the paired F1 delta vs a reference."""
    text = "F1 %.4f +- %.4f | accuracy %.4f +- %.4f" % (
        np.mean(res["f1"]),
        np.std(res["f1"]),
        np.mean(res["accuracy"]),
        np.std(res["accuracy"]),
    )
    if reference is not None:
        delta = 100 * np.subtract(res["f1"], reference["f1"])
        text += " | delta F1 %+.2f +- %.2f pp" % (delta.mean(), delta.std())
    return text


def save_result(name, config, res, folder="results"):
    """Write the configuration and the per-fold results of a run to folder/name.json."""
    try:
        commit = subprocess.run(
            ["git", "describe", "--always", "--dirty"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:  # git is not installed
        commit = ""
    record = {"name": name, "commit": commit, "config": config, **res}
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name + ".json"), "w") as f:
        json.dump(record, f, indent=1)
