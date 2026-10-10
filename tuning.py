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
    apply_bins,
    apply_expansion,
    fit_bins,
    fit_expansion,
    logistic_newton,
    predict_gbdt,
    predict_mlp,
    sigmoid,
    train_gbdt,
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


def mlp(n_models=1, first_seed=0, **kwargs):
    """Average of `n_models` networks trained with different seeds."""

    def fit_predict(y, tx, tx_val, is_cont=None):
        s_tr, s_val = 0.0, 0.0
        for seed in range(first_seed, first_seed + n_models):
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


# ---------------------------------------------------------------------------
# Stage G: gradient-boosted trees, better networks, 3-model blends.
# Out-of-fold (OOF) scores are cached in results/cache/ and the threshold is
# chosen on the OOF predictions (one threshold for all folds, as in run.py),
# because trees fit their training rows too well for a train-fold threshold.
# ---------------------------------------------------------------------------

CACHE = "results/cache"
THRESHOLDS = np.linspace(0.05, 0.6, 221)


def save_scores(name, scores):
    """Cache per-fold (train or None, validation) scores."""
    os.makedirs(CACHE, exist_ok=True)
    arrays = {}
    for i, (s_tr, s_val) in enumerate(scores):
        arrays[f"val{i}"] = s_val
        if s_tr is not None:
            arrays[f"tr{i}"] = s_tr
    np.savez(os.path.join(CACHE, name + ".npz"), **arrays)


def load_scores(name):
    """Validation scores of every fold from the cache."""
    d = np.load(os.path.join(CACHE, name + ".npz"))
    return [d[f"val{i}"] for i in range(K_FOLD)]


def evaluate_oof(y, val_scores, elapsed=0.0):
    """F1 per fold with one threshold maximizing the F1 of all OOF scores."""
    folds = stratified_k_indices(y, K_FOLD, SEED)
    y_all = np.concatenate([y[f] for f in folds])
    s_all = np.concatenate(val_scores)
    thr, _ = best_threshold(y_all, s_all, THRESHOLDS)
    f1s = np.array(
        [f1_score(y[f], (s >= thr).astype(int)) for f, s in zip(folds, val_scores)]
    )
    return {
        "f1": f1s,
        "mean": f1s.mean(),
        "std": f1s.std(),
        "thresholds": np.array([thr]),
        "time": elapsed,
    }


def stage_g0(y):
    """Cache the OOF scores of the current best logistic model and networks."""
    folds = best_folds()
    lin = evaluate(folds, expanded(newton(1e-4), squares=True, n_interact=20))
    save_scores("logistic_exp", lin["scores"])
    log(
        "G oof",
        "logistic Newton, squares + 20 interactions",
        evaluate_oof(y, [v for _, v in lin["scores"]], lin["time"]),
    )
    kw = dict(hidden=(128, 64), epochs=4, lr=1e-3, l2=1e-4, dropout=0.2)
    for seed in range(10):
        res = evaluate(folds, mlp(first_seed=seed, **kw))
        save_scores(f"mlp128_64_s{seed}", res["scores"])
        log(
            "G oof",
            f"MLP (128, 64) dropout 0.2, seed {seed}",
            evaluate_oof(y, [v for _, v in res["scores"]], res["time"]),
        )
    for n in [5, 10]:
        avg = [
            np.mean([load_scores(f"mlp128_64_s{s}")[i] for s in range(n)], axis=0)
            for i in range(K_FOLD)
        ]
        log(
            "G oof",
            f"MLP (128, 64) dropout 0.2, mean of {n} seeds",
            evaluate_oof(y, avg),
        )


GBDT_CONFIGS = {
    "d6_lr05": dict(
        max_depth=6, lr=0.05, min_child_weight=10, reg_lambda=1.0, n_trees=500
    ),
    "d4_lr1": dict(
        max_depth=4, lr=0.1, min_child_weight=10, reg_lambda=1.0, n_trees=600
    ),
    "d8_lr05": dict(
        max_depth=8, lr=0.05, min_child_weight=50, reg_lambda=5.0, n_trees=400
    ),
    "d6_lr03": dict(
        max_depth=6, lr=0.03, min_child_weight=20, reg_lambda=2.0, n_trees=600
    ),
}


def stage_g1(x, y, names):
    """Gradient-boosted trees on the decoded raw columns (NaN = own bin)."""
    folds = stratified_k_indices(y, K_FOLD, SEED)
    for name in names:
        cfg = dict(GBDT_CONFIGS[name])
        n_trees = cfg["n_trees"]
        checkpoints = list(range(50, n_trees + 1, 50))
        staged = {c: [] for c in checkpoints}
        t0 = time.time()
        for f in range(K_FOLD):
            va = folds[f]
            tr = np.concatenate([folds[i] for i in range(K_FOLD) if i != f])
            edges = fit_bins(x[tr])
            model = train_gbdt(
                y[tr],
                apply_bins(x[tr], edges),
                subsample=0.5,
                colsample=0.5,
                seed=f,
                **cfg,
            )
            pred = predict_gbdt(apply_bins(x[va], edges), model, checkpoints)
            for c in checkpoints:
                staged[c].append(pred[c])
            print(f"  gbdt {name} fold {f + 1}: {time.time() - t0:.0f}s", flush=True)
        for c in checkpoints:
            save_scores(f"gbdt_{name}_{c}", [(None, s) for s in staged[c]])
            desc = ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "n_trees")
            log(
                "G gbdt",
                f"GBDT {desc}, {c} trees",
                evaluate_oof(y, staged[c], time.time() - t0),
            )


def stage_g4(x, y, name, n_trees, seeds):
    """Extra seeds of one GBDT configuration (bagging over the random subsamples)."""
    folds = stratified_k_indices(y, K_FOLD, SEED)
    cfg = dict(GBDT_CONFIGS[name], n_trees=n_trees)
    for seed in seeds:
        t0 = time.time()
        scores = []
        for f in range(K_FOLD):
            va = folds[f]
            tr = np.concatenate([folds[i] for i in range(K_FOLD) if i != f])
            edges = fit_bins(x[tr])
            model = train_gbdt(
                y[tr],
                apply_bins(x[tr], edges),
                subsample=0.5,
                colsample=0.5,
                seed=100 * seed + f,
                **cfg,
            )
            scores.append((None, predict_gbdt(apply_bins(x[va], edges), model)))
        save_scores(f"gbdt_{name}_{n_trees}_seed{seed}", scores)
        log(
            "G gbdt",
            f"GBDT {name}, {n_trees} trees, seed {seed}",
            evaluate_oof(y, [v for _, v in scores], time.time() - t0),
        )


def stage_g2(y):
    """Networks with a cosine learning-rate schedule and longer training."""
    folds = best_folds()
    grid = {
        "mlp128_64_cos6": dict(
            hidden=(128, 64),
            epochs=6,
            lr=1e-3,
            l2=1e-4,
            dropout=0.2,
            lr_schedule="cosine",
        ),
        "mlp256_128_cos6": dict(
            hidden=(256, 128),
            epochs=6,
            lr=1e-3,
            l2=1e-4,
            dropout=0.3,
            lr_schedule="cosine",
        ),
        "mlp512_256_cos8": dict(
            hidden=(512, 256),
            epochs=8,
            lr=5e-4,
            l2=1e-4,
            dropout=0.4,
            lr_schedule="cosine",
        ),
    }
    for name, kw in grid.items():
        res = evaluate(folds, mlp(**kw))
        save_scores(name + "_s0", res["scores"])
        desc = ", ".join(f"{k}={v}" for k, v in kw.items())
        log(
            "G mlp",
            f"MLP {desc}",
            evaluate_oof(y, [v for _, v in res["scores"]], res["time"]),
        )


def stage_g5(y, seeds):
    """More seeds of the network with a cosine learning-rate schedule."""
    folds = best_folds()
    kw = dict(
        hidden=(128, 64), epochs=6, lr=1e-3, l2=1e-4, dropout=0.2, lr_schedule="cosine"
    )
    for seed in seeds:
        res = evaluate(folds, mlp(first_seed=seed, **kw))
        save_scores(f"mlp128_64_cos6_s{seed}", res["scores"])
        log(
            "G mlp",
            f"MLP (128, 64) cosine 6 epochs, seed {seed}",
            evaluate_oof(y, [v for _, v in res["scores"]], res["time"]),
        )


def stage_g3(y, members):
    """Blends of cached OOF scores: grid over weights summing to one (step 0.1)."""
    scores = {}
    for name, parts in members.items():
        lists = [load_scores(p) for p in parts]
        scores[name] = [np.mean([l[i] for l in lists], axis=0) for i in range(K_FOLD)]
        log("G blend", f"member {name}", evaluate_oof(y, scores[name]))
    names = list(scores)
    steps = np.round(np.arange(0, 1.01, 0.1), 2)
    results = []
    for w in np.array(np.meshgrid(*[steps] * len(names))).T.reshape(-1, len(names)):
        if abs(w.sum() - 1) > 1e-9 or (w > 0).sum() < 2:
            continue
        blended = [
            sum(wk * scores[n][i] for wk, n in zip(w, names)) for i in range(K_FOLD)
        ]
        results.append((w, evaluate_oof(y, blended)))
    results.sort(key=lambda r: -r[1]["mean"])
    for w, res in results[:5]:
        desc = " + ".join(f"{wk:g} {n}" for wk, n in zip(w, names) if wk > 0)
        log("G blend", desc, res)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    stages = sys.argv[1:] or ["A", "B", "C", "D", "E"]
    if not os.path.exists(RESULTS):
        with open(RESULTS, "w") as f:
            f.write("| stage | configuration | CV F1 | std | threshold | time |\n")
            f.write("|---|---|---|---|---|---|\n")
    if "A" in stages:
        stage_a()
    if any(st.startswith("G") for st in stages):
        x_raw, y_raw, _ = load_decoded(keep_state=True)
        if "G0" in stages:
            stage_g0(y_raw)
        if "G1" in stages:
            stage_g1(x_raw, y_raw, ["d6_lr05", "d4_lr1", "d8_lr05"])
        if "G6" in stages:
            stage_g1(x_raw, y_raw, ["d6_lr03"])
        if "G2" in stages:
            stage_g2(y_raw)
        if "G5" in stages:
            stage_g5(y_raw, seeds=range(1, 8))
        if "G4" in stages:
            stage_g4(x_raw, y_raw, "d6_lr05", 300, seeds=[1, 2, 3, 4])
        sys.exit()
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
