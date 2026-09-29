"""Cross-validation experiments behind the tables and the figure of the report.

`python experiments.py` takes about one hour: every experiment is saved to
results/<name>.json and the figure to report/threshold.pdf.
"""

import os

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
# Lower-priority questions, removed together in the ablation study.
EXTRA_QUESTIONS = """NUMADULT LANDLINE HHADULT CHOLCHK DIABAGE2 RENTHOM1 PREGNANT
STOPSMK2 LASTSMK2 LMTJOIN3 ARTHDIS2 ARTHSOCL JOINPAIN SEATBELT FLUSHOT6 PNEUVAC3
HIVTST6 INSULIN BLDSUGAR FEETCHK2 DOCTDIAB CHKHEMO3 FEETCHK EYEEXAM DIABEYE DIABEDU
CVDASPRN RDUCHART RDUCSTRK MSCODE HTM4 WTKG3 FTJUDA1_ FRUTDA1_ BEANDAY_ GRENDAY_
ORNGDAY_ VEGEDA1_ METVL11_ METVL21_ STRFREQ_ PA1VIGM_ _PACAT1""".split()


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


def mse_gd(y, tx):
    """MSE gradient descent with step 0.9 / L, L the largest eigenvalue of tx^T tx / N."""
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


# The required methods other than reg_logistic_regression, which is run.py's model.
METHODS = [
    ("least_squares", lambda y, tx: least_squares(y, tx)[0], {}),
    (
        "ridge_regression",
        lambda y, tx: ridge_regression(y, tx, 1e-5)[0],
        {"lambda_": 1e-5},
    ),
    ("mean_squared_error_gd", mse_gd, {"max_iters": 100, "gamma": "0.9 / L"}),
    ("mean_squared_error_sgd", mse_sgd, {"steps": "3 N", "gamma": 2e-5, "seed": 0}),
    (
        "logistic_regression",
        logistic_gd,
        {"gamma": run.GAMMA, "max_iters": run.MAX_ITERS},
    ),
]


def linear_scores(fit, x_fit, y_fit, x_apply):
    """Fit w = fit(y, tx) on the features of x_fit; return the scores of both sets."""
    tx_fit, tx_apply, _ = features.build_features(x_fit, x_apply)
    w = fit(y_fit, tx_fit)
    return tx_fit @ w, tx_apply @ w


def without(x, names):
    """Return a copy of x where the columns names are constant, i.e. carry no information."""
    x = x.copy()
    x[:, [features.USED.index(c) for c in names]] = 0
    return x


def evaluate(name, scores_fn, x, y, folds, config, reference=None):
    """Cross-validate scores_fn(x_fit, y_fit, x_apply); print, save and return it."""
    res, oof = cv.cross_validate(
        lambda tr, te: scores_fn(x[tr], y[tr], x[te]), y, folds
    )
    print(f"{name}:", cv.summary(res, reference))
    cv.save_result(name, config, res)
    return res, oof


def fixed_threshold(y, folds, oof, name, reference):
    """Score the out-of-fold logits with the default threshold 0 (probability 0.5)."""
    res = {"f1": [], "accuracy": []}
    for k in range(folds.max() + 1):
        f1, acc = cv.f1_accuracy(y[folds == k], oof[folds == k] >= 0)
        res["f1"].append(f1)
        res["accuracy"].append(acc)
    print(f"{name}:", cv.summary(res, reference))
    cv.save_result(name, {"threshold": 0.0}, res)


def plot_threshold(y, folds, oof, thresholds, path):
    """Plot the test F1 of every fold against the probability threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probs = np.linspace(0.03, 0.6, 115)
    logits = np.log(probs / (1 - probs))
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    points = []
    for k in range(folds.max() + 1):
        test = folds == k
        f1 = [cv.f1_accuracy(y[test], oof[test] >= t)[0] for t in logits]
        ax.plot(probs, f1, "C0", lw=0.8, label="test folds" if k == 0 else None)
        chosen = 1 / (1 + np.exp(-thresholds[k]))
        points.append((chosen, np.interp(chosen, probs, f1)))
    ax.plot(*zip(*points), "k.", ms=5, label="chosen on training data")
    ax.axvline(0.5, color="grey", ls="--", lw=0.8, label="default 0.5")
    ax.set_xlabel("probability threshold")
    ax.set_ylabel("F1")
    ax.legend(fontsize=7, frameon=False, loc="lower center")
    fig.tight_layout()
    fig.savefig(path)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths are relative to here
    x, _, y, _ = features.load_data(run.DATA_DIR)
    folds = cv.stratified_folds(y)
    final = {"lambda": run.LAMBDA, "gamma": run.GAMMA, "max_iters": run.MAX_ITERS}
    ref, oof = evaluate("logistic", run.logistic_scores, x, y, folds, final)
    plot_threshold(y, folds, oof, ref["threshold"], "report/threshold.pdf")
    fixed_threshold(y, folds, oof, "threshold_0.5", ref)

    config = {"lambda": run.LAMBDA, "iters": NEWTON_ITERS}
    evaluate("newton", newton_scores, x, y, folds, config, ref)
    for name, fit, config in METHODS:
        scores_fn = lambda a, b, c, fit=fit: linear_scores(fit, a, b, c)
        evaluate(name, scores_fn, x, y, folds, config, ref)

    x_raw = features.load_data(run.DATA_DIR, decoded=False)[0]
    ablations = [
        ("no_decoding", x_raw, {}),
        ("no_extra_questions", without(x, EXTRA_QUESTIONS), {}),
        ("no_state", without(x, ["_STATE"]), {}),
        ("no_rehab_flag", without(x, [features.REHAB]), {}),
        ("no_missing_indicators", x, {"indicator_rate": 2.0}),
        ("lambda_1", x, {"lam": 1.0}),
        ("lambda_100", x, {"lam": 100.0}),
    ]
    for name, x_ablation, options in ablations:
        scores_fn = lambda a, b, c, o=options: run.logistic_scores(a, b, c, **o)
        evaluate(name, scores_fn, x_ablation, y, folds, {**final, **options}, ref)


if __name__ == "__main__":
    main()
