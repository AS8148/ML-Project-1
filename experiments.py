"""Cross-validation experiments behind the tables and the figure of the report.

`python experiments.py [final] [methods] [ablations] [gbdt] [scores]` runs the chosen
groups (by default all but scores, which repeats runs of final and gbdt: three to
four hours, of which gbdt takes almost two). The logistic regression baseline, the
reference of the paired deltas, is always run. Every experiment is saved to
results/<name>.json, its test and training scores to results/oof_<name>.npy and
oof_<name>_fit.npy, the figure to report/threshold.pdf.
"""

import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import cv
import features
import run
from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    ridge_regression,
)

NEWTON_ITERS = 10
# Secondary questions, removed together in the ablation study.
EXTRA_QUESTIONS = """NUMADULT LANDLINE HHADULT CHOLCHK DIABAGE2 RENTHOM1 PREGNANT
STOPSMK2 LASTSMK2 LMTJOIN3 ARTHDIS2 ARTHSOCL JOINPAIN SEATBELT FLUSHOT6 PNEUVAC3
HIVTST6 INSULIN BLDSUGAR FEETCHK2 DOCTDIAB CHKHEMO3 FEETCHK EYEEXAM DIABEYE DIABEDU
CVDASPRN RDUCHART RDUCSTRK MSCODE HTM4 WTKG3 FTJUDA1_ FRUTDA1_ BEANDAY_ GRENDAY_
ORNGDAY_ VEGEDA1_ METVL11_ METVL21_ STRFREQ_ PA1VIGM_ _PACAT1""".split()


def newton(y, tx):
    """Minimize the objective of run.py with Newton's method (its exact optimum)."""
    w = np.zeros(tx.shape[1])
    for _ in range(NEWTON_ITERS):
        p = 1 / (1 + np.exp(-(tx @ w)))
        grad = tx.T @ (p - y) + run.LAMBDA * w
        hess = (tx.T * (p * (1 - p))) @ tx + run.LAMBDA * np.eye(len(w))
        w -= np.linalg.solve(hess, grad)
    return w


def mse_gd(y, tx):
    """MSE gradient descent with step 0.9 / L, L the top eigenvalue of tx^T tx / N."""
    gamma = 0.9 / np.linalg.eigvalsh(tx.T @ tx / len(y)).max()
    return mean_squared_error_gd(y, tx, np.zeros(tx.shape[1]), 100, gamma)[0]


def mse_sgd(y, tx):
    """MSE stochastic gradient descent for three epochs, with a fixed seed."""
    np.random.seed(0)
    return mean_squared_error_sgd(y, tx, np.zeros(tx.shape[1]), 3 * len(y), 2e-5)[0]


def logistic_gd(y, tx):
    """Unregularized logistic regression with the step and iterations of run.py."""
    w0 = np.zeros(tx.shape[1])
    return logistic_regression(y, tx, w0, run.MAX_ITERS, run.GAMMA)[0]


# Newton (convergence check of run.py) and the other five required methods.
FITS = [
    ("newton", newton, {"lam": run.LAMBDA, "iters": NEWTON_ITERS}),
    ("least_squares", lambda y, tx: least_squares(y, tx)[0], {}),
    (
        "ridge_regression",
        lambda y, tx: ridge_regression(y, tx, 1e-5)[0],
        {"lambda_": 1e-5},
    ),
    ("mean_squared_error_gd", mse_gd, {"max_iters": 100, "gamma": "0.9 / L"}),
    ("mean_squared_error_sgd", mse_sgd, {"max_iters": "3 N", "gamma": 2e-5}),
    (
        "logistic_regression",
        logistic_gd,
        {"gamma": run.GAMMA, "max_iters": run.MAX_ITERS},
    ),
]

# Each ablation changes one part of the logistic regression baseline.
ABLATIONS = [
    ("no_decoding", {"decoded": False}),
    ("no_extra_questions", {"removed": EXTRA_QUESTIONS}),
    ("no_state", {"removed": ["_STATE"]}),
    ("no_rehab_flag", {"removed": [features.REHAB]}),
    ("no_missing_indicators", {"indicator_rate": 2.0}),
    ("lambda_1", {"lam": 1.0}),
    ("lambda_100", {"lam": 100.0}),
]

# Gradient boosting (gbdt.py, other parameters at their defaults): (depth, rate).
GBDT_GRID = [(3, 0.1), (4, 0.1), (3, 0.05), (4, 0.05)]

GROUPS = ["final", "methods", "ablations", "gbdt", "scores"]


def linear_scores(fit, x_fit, y_fit, x_apply):
    """Fit w = fit(y, tx) on the features of x_fit; return the scores of both sets."""
    tx_fit, tx_apply, _ = features.build_features(x_fit, x_apply)
    w = fit(y_fit, tx_fit)
    return tx_fit @ w, tx_apply @ w


def without(x, names):
    """Return a copy of x with the columns names set to 0, so they carry no signal."""
    x = x.copy()
    # Zeroed, not deleted: build_features expects the column layout of USED.
    x[:, [features.USED.index(c) for c in names]] = 0
    return x


def evaluate(name, scores_fn, x, y, folds, config, reference=None):
    """Cross-validate scores_fn(x_fit, y_fit, x_apply) and print the result.

    Saves the results and the scores (see the module docstring); returns (res, oof).
    """
    res, oof, fit = cv.cross_validate(
        lambda tr, te: scores_fn(x[tr], y[tr], x[te]), y, folds
    )
    print(f"{name}:", cv.summary(res, reference))
    cv.save_result(name, config, res)
    np.save(f"results/oof_{name}.npy", oof)
    np.save(f"results/oof_{name}_fit.npy", fit)
    return res, oof


def fixed_threshold(y, folds, oof, reference):
    """Score the out-of-fold logits of the baseline with probability 0.5."""
    res = {"f1": [], "accuracy": []}
    for k in range(folds.max() + 1):
        f1, acc = cv.f1_accuracy(y[folds == k], oof[folds == k] >= 0)
        res["f1"].append(f1)
        res["accuracy"].append(acc)
    print("threshold_0.5:", cv.summary(res, reference))
    cv.save_result("threshold_0.5", {"probability": 0.5, "logit": 0.0}, res)


def plot_threshold(y, folds, oof, res, path):
    """Plot the test F1 of every fold against the probability threshold."""
    probs = np.linspace(0.03, 0.6, 115)
    logits = np.log(probs / (1 - probs))
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for k in range(folds.max() + 1):
        test = folds == k
        f1 = [cv.f1_accuracy(y[test], oof[test] >= t)[0] for t in logits]
        ax.plot(probs, f1, "C0", lw=0.8, label="test folds" if k == 0 else None)
    chosen = 1 / (1 + np.exp(-np.array(res["threshold"])))  # logits to probabilities
    ax.plot(chosen, res["f1"], "k.", ms=5, label="chosen on training data")
    ax.axvline(0.5, color="grey", ls="--", lw=0.8, label="default 0.5")
    ax.set_xlabel("probability threshold")
    ax.set_ylabel("F1")
    ax.legend(fontsize=7, frameon=False, loc="lower center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(groups):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths are relative to here
    x, _, y, _ = features.load_data(run.DATA_DIR)
    folds = cv.stratified_folds(y)
    baseline = {"lam": run.LAMBDA, "gamma": run.GAMMA, "max_iters": run.MAX_ITERS}
    ref, oof = evaluate("logistic", run.logistic_scores, x, y, folds, baseline)

    if "final" in groups:
        plot_threshold(y, folds, oof, ref, "report/threshold.pdf")
        fixed_threshold(y, folds, oof, ref)
        # All choices were made on the folds of seed 42: check on unseen folds.
        new_folds = cv.stratified_folds(y, seed=43)
        ref43, _ = evaluate(
            "logistic_seed43", run.logistic_scores, x, y, new_folds, baseline
        )
        config = {"depth": run.DEPTH, "rate": run.RATE}
        evaluate("gbdt_seed43", run.gbdt_scores, x, y, new_folds, config, ref43)

    if "methods" in groups:
        for name, fit, config in FITS:
            evaluate(name, partial(linear_scores, fit), x, y, folds, config, ref)

    if "ablations" in groups:
        for name, change in ABLATIONS:
            options = dict(change)
            x_ablation = x
            if not options.pop("decoded", True):
                x_ablation = features.load_data(run.DATA_DIR, decoded=False)[0]
            if "removed" in options:
                x_ablation = without(x, options.pop("removed"))
            scores_fn = partial(run.logistic_scores, **options)
            config = {**baseline, **change}
            evaluate(name, scores_fn, x_ablation, y, folds, config, ref)

    if "gbdt" in groups:
        for depth, rate in GBDT_GRID:
            config = {"depth": depth, "rate": rate}
            scores_fn = partial(run.gbdt_scores, **config)
            evaluate(
                f"gbdt_depth{depth}_rate{rate}", scores_fn, x, y, folds, config, ref
            )

    if "scores" in groups:
        # The final gradient boosting again, to save its scores on both fold seeds.
        # The baseline of seed 43 is read from its saved results (group final).
        ref43 = cv.load_result("logistic_seed43")
        new_folds = cv.stratified_folds(y, seed=43)
        config = {"depth": run.DEPTH, "rate": run.RATE}
        name = f"gbdt_depth{run.DEPTH}_rate{run.RATE}"
        evaluate(name, run.gbdt_scores, x, y, folds, config, ref)
        evaluate("gbdt_seed43", run.gbdt_scores, x, y, new_folds, config, ref43)


if __name__ == "__main__":
    if not set(sys.argv[1:]) <= set(GROUPS):
        sys.exit("usage: python experiments.py " + " ".join(f"[{g}]" for g in GROUPS))
    main(sys.argv[1:] or GROUPS[:-1])
