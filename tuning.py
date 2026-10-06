"""Hyper-parameter and model selection with stratified 5-fold cross-validation.

Usage:  python tuning.py            (all stages, ~1 hour)
        python tuning.py A B        (only some stages)

Every configuration is evaluated on the same 5 stratified folds. The cleaner
(and the optional feature expansion) is fitted on the training part of each
fold, and the decision threshold is chosen to maximize F1 on the training part,
then applied to the validation part. Results are appended to
results/tuning_results.md; the best configuration is copied into run.py.
"""

import os
import sys
import time

import numpy as np

from cross_validation import best_threshold, f1_score, stratified_k_indices
from data_cleaning import (
    drop_columns,
    fit_cleaner,
    load_data,
    replace_missing_codes,
    transform,
    ADMIN_COLUMNS,
    REDUNDANT_COLUMNS,
)
from implementations import reg_logistic_regression, ridge_regression
from models import (
    apply_expansion,
    fit_expansion,
    logistic_newton,
    predict_mlp,
    sigmoid,
    train_mlp,
)

K_FOLD = 5
SEED = 1
RESULTS = "results/tuning_results.md"


# ---------------------------------------------------------------------------
# Data and folds
# ---------------------------------------------------------------------------


def load_decoded(keep_state=False):
    """Raw training data -> decoded features (stateless steps only)."""
    x, _, y_pm, _, _, cols = load_data()
    to_drop = [c for c in ADMIN_COLUMNS + REDUNDANT_COLUMNS]
    if keep_state:
        to_drop.remove("_STATE")
    x, cols = drop_columns(x, cols, to_drop)
    return replace_missing_codes(x, cols), (y_pm == 1).astype(float), cols


def prepare_folds(x, y, cols, cleaner_kwargs):
    """Clean every CV fold once (fit on the training part) and cache it."""
    folds = stratified_k_indices(y, K_FOLD, SEED)
    out = []
    for f in range(K_FOLD):
        val_idx = folds[f]
        tr_idx = np.concatenate([folds[i] for i in range(K_FOLD) if i != f])
        params = fit_cleaner(x[tr_idx], cols, **cleaner_kwargs)
        out.append(
            {
                "tx_tr": transform(x[tr_idx], params),
                "y_tr": y[tr_idx],
                "tx_val": transform(x[val_idx], params),
                "y_val": y[val_idx],
                "val_idx": val_idx,
                "is_continuous": params["is_continuous"],
            }
        )
    return out


def evaluate(folds, fit_predict):
    """Cross-validated F1 of a model.

    Args:
        folds: output of `prepare_folds`.
        fit_predict: function (y_tr, tx_tr, tx_val, is_continuous)
            -> (scores_tr, scores_val).

    Returns:
        dict with per-fold F1, mean/std, thresholds, the per-fold scores
        (to blend models without retraining them) and the time.
    """
    t0 = time.time()
    scores = [
        fit_predict(fd["y_tr"], fd["tx_tr"], fd["tx_val"], fd["is_continuous"])
        for fd in folds
    ]
    return evaluate_scores(folds, scores, time.time() - t0)


def evaluate_scores(folds, scores, elapsed=0.0):
    """F1 of given per-fold (train, validation) scores.

    The threshold is chosen to maximize F1 on the training part of each fold.
    """
    f1s, thresholds = [], []
    for fd, (s_tr, s_val) in zip(folds, scores):
        thr, _ = best_threshold(fd["y_tr"], s_tr)
        f1s.append(f1_score(fd["y_val"], (s_val >= thr).astype(int)))
        thresholds.append(thr)
    f1s = np.array(f1s)
    return {
        "f1": f1s,
        "mean": f1s.mean(),
        "std": f1s.std(),
        "thresholds": np.array(thresholds),
        "scores": scores,
        "time": elapsed,
    }


def blend_scores(scores_a, scores_b, alpha):
    """Per-fold weighted average alpha * a + (1 - alpha) * b of two models' scores."""
    return [
        (alpha * a_tr + (1 - alpha) * b_tr, alpha * a_val + (1 - alpha) * b_val)
        for (a_tr, a_val), (b_tr, b_val) in zip(scores_a, scores_b)
    ]


def log(stage, name, res):
    """Print a result and append it to the results file."""
    line = (
        f"| {stage} | {name} | {res['mean']:.4f} | {res['std']:.4f} | "
        f"{res['thresholds'].mean():.3f} | {res['time']:.0f}s |"
    )
    print(line, flush=True)
    with open(RESULTS, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Model factories: each returns fit_predict(y_tr, tx_tr, tx_val, is_continuous)
# ---------------------------------------------------------------------------


def newton(lambda_=1e-4, pos_weight=1.0):
    def fit_predict(y, tx, tx_val, is_cont=None):
        w = logistic_newton(y, tx, lambda_=lambda_, pos_weight=pos_weight)
        return sigmoid(tx @ w), sigmoid(tx_val @ w)

    return fit_predict


def ridge(lambda_):
    def fit_predict(y, tx, tx_val, is_cont=None):
        w, _ = ridge_regression(y, tx, lambda_)
        return tx @ w, tx_val @ w

    return fit_predict


def logistic_gd(lambda_=1e-4, iters=300, gamma=0.5):
    def fit_predict(y, tx, tx_val, is_cont=None):
        w, _ = reg_logistic_regression(
            y, tx, lambda_, np.zeros(tx.shape[1]), iters, gamma
        )
        return sigmoid(tx @ w), sigmoid(tx_val @ w)

    return fit_predict


def mlp(n_models=1, **kwargs):
    """Average of `n_models` networks trained with different seeds."""

    def fit_predict(y, tx, tx_val, is_cont=None):
        s_tr, s_val = 0.0, 0.0
        for seed in range(n_models):
            p = train_mlp(y, tx, seed=seed, **kwargs)
            s_tr = s_tr + predict_mlp(tx, p) / n_models
            s_val = s_val + predict_mlp(tx_val, p) / n_models
        return s_tr, s_val

    return fit_predict


def expanded(fit_predict, **expansion):
    """Fit the feature expansion on the training part, then the model."""

    def wrapped(y, tx, tx_val, is_cont):
        exp = fit_expansion(tx, y, is_cont, **expansion)
        return fit_predict(y, apply_expansion(tx, exp), apply_expansion(tx_val, exp))

    return wrapped


def blend(fp_a, fp_b, alpha=0.5):
    """Weighted average of the probabilities of two models."""

    def fit_predict(y, tx, tx_val, is_cont):
        a_tr, a_val = fp_a(y, tx, tx_val, is_cont)
        b_tr, b_val = fp_b(y, tx, tx_val, is_cont)
        return alpha * a_tr + (1 - alpha) * b_tr, alpha * a_val + (1 - alpha) * b_val

    return fit_predict


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

BASE_CLEANER = dict(max_nan_frac=1.0, add_nan_indicators=True)


def stage_a():
    """Cleaning options (model: Newton logistic regression, lambda = 1e-4)."""
    x, y, cols = load_decoded()
    configs = {
        "NaN<=0.5, no indicators": dict(max_nan_frac=0.5),
        "NaN<=0.9 + indicators": dict(max_nan_frac=0.9, add_nan_indicators=True),
        "keep all + indicators": BASE_CLEANER,
        "keep all + indicators, no clipping": dict(BASE_CLEANER, clip_quantiles=None),
    }
    for name, kw in configs.items():
        log("A cleaning", name, evaluate(prepare_folds(x, y, cols, kw), newton()))
    x, y, cols = load_decoded(keep_state=True)
    kw = dict(BASE_CLEANER, extra_nominal=("_STATE",))
    folds = prepare_folds(x, y, cols, kw)
    log(
        "A cleaning", "keep all + indicators + one-hot state", evaluate(folds, newton())
    )


def best_folds():
    """Folds for the best cleaning found in stage A (keep all + state)."""
    x, y, cols = load_decoded(keep_state=True)
    kw = dict(BASE_CLEANER, extra_nominal=("_STATE",))
    return prepare_folds(x, y, cols, kw)


def stage_b(folds):
    """Linear models and their hyper-parameters."""
    for lam in [1e-5, 1e-3, 1e-1]:
        log(
            "B linear", f"ridge regression, lambda={lam:g}", evaluate(folds, ridge(lam))
        )
    log("B linear", "logistic GD (300 it, gamma=0.5)", evaluate(folds, logistic_gd()))
    for lam in [0, 1e-5, 1e-4, 1e-3, 1e-2]:
        log(
            "B linear", f"logistic Newton, lambda={lam:g}", evaluate(folds, newton(lam))
        )
    for pw in [2, 5, 10]:
        log(
            "B linear",
            f"logistic Newton, lambda=1e-4, pos_weight={pw}",
            evaluate(folds, newton(1e-4, pw)),
        )


def stage_c(folds, lam=1e-4):
    """Feature expansion for logistic regression."""
    for name, exp in {
        "squares": dict(squares=True),
        "squares + 20 interactions": dict(squares=True, n_interact=20),
        "squares + 40 interactions": dict(squares=True, n_interact=40),
    }.items():
        for l2 in [lam, 1e-3]:
            log(
                "C features",
                f"logistic Newton lambda={l2:g}, {name}",
                evaluate(folds, expanded(newton(l2), **exp)),
            )


def stage_d(folds):
    """One-hidden-layer neural network."""
    grid = [
        dict(hidden=32, epochs=4, lr=1e-3, l2=1e-4),
        dict(hidden=64, epochs=4, lr=1e-3, l2=1e-4),
        dict(hidden=64, epochs=8, lr=5e-4, l2=1e-4),
        dict(hidden=128, epochs=4, lr=5e-4, l2=1e-3),
        dict(hidden=64, epochs=4, lr=1e-3, l2=1e-3),
    ]
    for kw in grid:
        name = ", ".join(f"{k}={v:g}" for k, v in kw.items())
        log("D MLP", f"MLP {name}", evaluate(folds, mlp(**kw)))


def stage_d2(folds):
    """Neural network: more regularization / shorter training (after stage D)."""
    grid = [
        dict(hidden=64, epochs=2, lr=1e-3, l2=1e-3),
        dict(hidden=64, epochs=4, lr=1e-3, l2=3e-3),
        dict(hidden=128, epochs=3, lr=1e-3, l2=3e-3),
    ]
    for kw in grid:
        name = ", ".join(f"{k}={v:g}" for k, v in kw.items())
        log("D MLP", f"MLP {name}", evaluate(folds, mlp(**kw)))


def stage_e(folds, mlp_kwargs, lam, expansion):
    """Ensembles: several MLP seeds, MLP + logistic regression blends."""
    log("E ensemble", "MLP x5 seeds", evaluate(folds, mlp(n_models=5, **mlp_kwargs)))
    lin = expanded(newton(lam), **expansion)
    net = mlp(n_models=5, **mlp_kwargs)
    for alpha in [0.3, 0.5, 0.7]:
        log(
            "E ensemble",
            f"{alpha:g} * logistic(expanded) + {1 - alpha:g} * MLP x5",
            evaluate(folds, blend(lin, net, alpha)),
        )


def stage_f(folds):
    """Deeper / dropout-regularized networks, then offline blends with logistic."""
    lin = evaluate(folds, expanded(newton(1e-4), squares=True, n_interact=20))
    log("F blend", "logistic Newton, squares + 20 interactions", lin)
    grid = [
        dict(hidden=64, epochs=4, lr=1e-3, l2=1e-3),
        dict(hidden=64, epochs=4, lr=1e-3, l2=1e-4, dropout=0.2),
        dict(hidden=256, epochs=4, lr=1e-3, l2=1e-4, dropout=0.4),
        dict(hidden=(128, 64), epochs=4, lr=1e-3, l2=1e-4, dropout=0.2),
        dict(hidden=(256, 128), epochs=5, lr=5e-4, l2=1e-4, dropout=0.3),
    ]
    best, best_kw = None, None
    for kw in grid:
        name = ", ".join(f"{k}={v}" for k, v in kw.items())
        res = evaluate(folds, mlp(**kw))
        log("F MLP", f"MLP {name}", res)
        if best is None or res["mean"] > best["mean"]:
            best, best_kw = res, kw
    name = ", ".join(f"{k}={v}" for k, v in best_kw.items())
    ens = evaluate(folds, mlp(n_models=5, **best_kw))
    log("F MLP", f"MLP x5 seeds, {name}", ens)
    for alpha in [0.2, 0.35, 0.5, 0.65, 0.8]:
        res = evaluate_scores(folds, blend_scores(lin["scores"], ens["scores"], alpha))
        log(
            "F blend",
            f"{alpha:g} * logistic(expanded) + {1 - alpha:g} * MLP x5 ({name})",
            res,
        )


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    stages = sys.argv[1:] or ["A", "B", "C", "D", "E"]
    if not os.path.exists(RESULTS):
        with open(RESULTS, "w") as f:
            f.write("| stage | configuration | CV F1 | std | threshold | time |\n")
            f.write("|---|---|---|---|---|---|\n")
    if "A" in stages:
        stage_a()
    folds = best_folds() if set(stages) - {"A"} else None
    if "B" in stages:
        stage_b(folds)
    if "C" in stages:
        stage_c(folds)
    if "D" in stages:
        stage_d(folds)
    if "D2" in stages:
        stage_d2(folds)
    if "F" in stages:
        stage_f(folds)
    if "E" in stages:
        stage_e(
            folds,
            mlp_kwargs=dict(hidden=64, epochs=4, lr=1e-3, l2=1e-3),
            lam=1e-4,
            expansion=dict(squares=True, n_interact=20),
        )
