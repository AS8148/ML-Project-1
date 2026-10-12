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
    """Return the threshold t maximizing the F1 of the rule scores >= t.

    NaN scores are skipped (a model may score only some of its training rows).
    """
    known = ~np.isnan(scores)
    y, scores = y[known], scores[known]
    order = np.argsort(-scores, kind="stable")
    s, ys = scores[order], y[order]
    tp = np.cumsum(ys)
    f1 = 2 * tp / (np.arange(1, len(ys) + 1) + ys.sum())  # 2tp / (tp+fp + tp+fn)
    f1[:-1][s[1:] == s[:-1]] = -1  # never split tied scores
    return s[np.argmax(f1)]


def cross_validate(fit_score, y, folds):
    """Cross-validate fit_score(train_idx, test_idx) -> (train_scores, test_scores).

    Returns (res, oof, fit): the per-fold results of score_folds plus the fit times,
    the out-of-fold test scores, and one row of training scores per fold (NaN where
    fit_score gave no score, e.g. outside the validation rows of the boosting).
    """
    seconds, oof = [], np.zeros(len(y))
    fit = np.full((folds.max() + 1, len(y)), np.nan)
    for k in range(len(fit)):
        train, test = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        start = time.time()
        fit[k, train], oof[test] = fit_score(train, test)
        seconds.append(time.time() - start)
    return {**score_folds(y, folds, oof, fit), "seconds": seconds}, oof, fit


def score_folds(y, folds, oof, fit):
    """Return the per-fold F1, accuracy and threshold of saved scores.

    oof holds the test scores of every row, fit[k] the training scores of fold k
    (NaN where missing). The threshold of fold k is chosen on fit[k] only.
    """
    res = {"f1": [], "accuracy": [], "threshold": []}
    for k in range(len(fit)):
        t = best_threshold(y, fit[k])
        f1, acc = f1_accuracy(y[folds == k], oof[folds == k] >= t)
        res["f1"].append(f1)
        res["accuracy"].append(acc)
        res["threshold"].append(t)
    return res


def summary(res, reference=None):
    """Return F1, accuracy and the paired F1 delta vs a reference: mean +- std over folds."""
    f1, acc = np.array(res["f1"]), np.array(res["accuracy"])
    text = "F1 %.4f +- %.4f | accuracy %.4f +- %.4f" % (
        f1.mean(),
        f1.std(ddof=1),
        acc.mean(),
        acc.std(ddof=1),
    )
    if reference is not None:
        delta = 100 * (f1 - reference["f1"])
        text += " | delta F1 %+.2f +- %.2f pp" % (delta.mean(), delta.std(ddof=1))
    return text


def _git(*args):
    """Return the output of a git command, or "" if git is not available."""
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout
    except OSError:
        return ""


def save_result(name, config, res, folder="results"):
    """Write the configuration and the per-fold results of a run to folder/name.json.

    The commit gets a "-dirty" suffix when the Python code has uncommitted changes.
    """
    commit = _git("rev-parse", "--short", "HEAD").strip()
    if _git("status", "--porcelain", "--", "*.py"):
        commit += "-dirty"
    record = {"name": name, "commit": commit, "config": config, **res}
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name + ".json"), "w") as f:
        json.dump(record, f, indent=1)


def load_result(name, folder="results"):
    """Return the record written by save_result."""
    with open(os.path.join(folder, name + ".json")) as f:
        return json.load(f)
