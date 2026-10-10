"""Additional models and feature expansions used for the competition (step 3).

Everything is implemented with NumPy only:
  * logistic_newton: L2-regularized, class-weighted logistic regression trained
    with Newton's method (converges in ~10 iterations instead of thousands of
    gradient steps);
  * train_mlp / predict_mlp: one-hidden-layer neural network trained with Adam;
  * fit_expansion / apply_expansion: squared continuous features and pairwise
    interactions of the most label-correlated features.
"""

import numpy as np


def sigmoid(t):
    """Numerically stable logistic function."""
    return np.exp(-np.logaddexp(0, -t))


# ---------------------------------------------------------------------------
# Logistic regression with Newton's method
# ---------------------------------------------------------------------------


def _weighted_logistic_objective(y, tx, w, s, reg):
    """Weighted mean logistic loss + L2 penalty (penalty weights in `reg`)."""
    z = tx @ w
    return np.mean(s * (np.logaddexp(0, z) - y * z)) + 0.5 * np.sum(reg * w**2)


def logistic_newton(
    y, tx, lambda_=0.0, pos_weight=1.0, max_iters=30, tol=1e-8, chunk=50000
):
    """Regularized logistic regression trained with Newton's method.

    Minimizes  1/N sum_n s_n [log(1 + exp(x_n^T w)) - y_n x_n^T w] + lambda_ ||w[1:]||^2,
    where s_n = pos_weight for positive samples and 1 otherwise (normalized to
    mean 1), and the first column of tx is the (unpenalized) bias.

    Each iteration solves H d = g with the exact Hessian
    H = 1/N X^T diag(s p (1 - p)) X + 2 lambda_ I, followed by a backtracking
    line search, so the objective decreases monotonically.

    Args:
        y: (N,) labels in {0, 1}.
        tx: (N, D) features, first column = bias.
        lambda_: L2 regularization strength.
        pos_weight: weight of the positive class (> 1 up-weights the rare class).
        max_iters: maximum number of Newton iterations.
        tol: stop when the objective decreases by less than tol.
        chunk: number of rows processed at once to build the Hessian (memory).

    Returns:
        (D,) weight vector.
    """
    n, d = tx.shape
    s = np.where(y == 1, pos_weight, 1.0)
    s = s / s.mean()
    reg = np.full(d, 2.0 * lambda_)
    reg[0] = 0.0
    w = np.zeros(d)
    obj = _weighted_logistic_objective(y, tx, w, s, reg)
    for _ in range(max_iters):
        p = sigmoid(tx @ w)
        grad = tx.T @ (s * (p - y)) / n + reg * w
        h = np.diag(reg + 1e-10)
        r = s * p * (1 - p)
        for start in range(0, n, chunk):
            xc = tx[start : start + chunk]
            h += (xc * r[start : start + chunk, None]).T @ xc / n
        step = np.linalg.solve(h, grad)
        t = 1.0
        while True:
            w_new = w - t * step
            obj_new = _weighted_logistic_objective(y, tx, w_new, s, reg)
            if obj_new <= obj or t < 1e-6:
                break
            t /= 2
        converged = obj - obj_new < tol
        w, obj = w_new, obj_new
        if converged:
            break
    return w


# ---------------------------------------------------------------------------
# One-hidden-layer neural network
# ---------------------------------------------------------------------------


def train_mlp(
    y,
    tx,
    hidden=64,
    epochs=8,
    lr=1e-3,
    batch_size=512,
    l2=1e-4,
    pos_weight=1.0,
    seed=0,
    dropout=0.0,
    lr_schedule="constant",
):
    """Train a fully-connected ReLU network with a sigmoid output.

    Loss: class-weighted binary cross-entropy + l2 * sum ||W_k||^2 / 2,
    optimized with Adam on shuffled mini-batches. Optional (inverted) dropout
    on the hidden layers during training.

    Args:
        y: (N,) labels in {0, 1}.
        tx: (N, D) standardized features.
        hidden: number of hidden units (int, one hidden layer) or a tuple with
            the size of each hidden layer, e.g. (128, 64).
        epochs: number of passes over the data.
        lr: Adam learning rate.
        batch_size: mini-batch size.
        l2: weight decay coefficient.
        pos_weight: weight of the positive class.
        seed: random seed (initialization, shuffling and dropout).
        dropout: probability of dropping a hidden unit during training.
        lr_schedule: "constant" or "cosine" (learning rate decayed from lr to 0
            along a half cosine over the whole training).

    Returns:
        dict of parameters {"W1", "b1", ..., "WL", "bL"}; the last layer has a
        vector weight and a scalar bias.
    """
    rng = np.random.default_rng(seed)
    n, d = tx.shape
    sizes = [d] + ([hidden] if np.isscalar(hidden) else list(hidden))
    n_layers = len(sizes)  # hidden layers + output layer
    p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    params = {}
    for k in range(1, n_layers):
        params[f"W{k}"] = rng.normal(0, np.sqrt(2 / sizes[k - 1]), sizes[k - 1 : k + 1])
        params[f"b{k}"] = np.zeros(sizes[k])
    params[f"W{n_layers}"] = rng.normal(0, np.sqrt(1 / sizes[-1]), sizes[-1])
    params[f"b{n_layers}"] = np.array(np.log(p0 / (1 - p0)))

    m = {k: np.zeros_like(val) for k, val in params.items()}
    v = {k: np.zeros_like(val) for k, val in params.items()}
    beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
    s_all = np.where(y == 1, pos_weight, 1.0)
    s_all = s_all / s_all.mean()
    total_steps = epochs * int(np.ceil(n / batch_size))

    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            _, grads = mlp_loss_and_grads(
                params, tx[idx], y[idx], s_all[idx], l2, dropout, rng
            )
            # Adam update
            step += 1
            lr_t = lr
            if lr_schedule == "cosine":
                lr_t = lr * 0.5 * (1 + np.cos(np.pi * (step - 1) / total_steps))
            for k in params:
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                m_hat = m[k] / (1 - beta1**step)
                v_hat = v[k] / (1 - beta2**step)
                params[k] = params[k] - lr_t * m_hat / (np.sqrt(v_hat) + eps)
    return params


def mlp_loss_and_grads(params, xb, yb, sb, l2=0.0, dropout=0.0, rng=None):
    """Loss and gradients of the network on one mini-batch (backpropagation).

    Args:
        params: network parameters (see `train_mlp`).
        xb: (B, D) inputs; yb: (B,) labels in {0, 1}; sb: (B,) sample weights.
        l2: weight decay coefficient.
        dropout: dropout probability on hidden units (needs `rng` if > 0).
        rng: numpy Generator used for the dropout masks.

    Returns:
        (loss, grads) with grads a dict with the same keys as params.
    """
    n_layers = len(params) // 2
    # forward: keep the input of every layer and the ReLU (and dropout) masks
    inputs, masks = [xb], []
    for k in range(1, n_layers):
        a = inputs[-1] @ params[f"W{k}"] + params[f"b{k}"]
        mask = a > 0
        if dropout > 0:
            mask = mask & (rng.uniform(size=a.shape) >= dropout)
            h = np.where(mask, a, 0) / (1 - dropout)
        else:
            h = np.where(mask, a, 0)
        inputs.append(h)
        masks.append(mask)
    z = inputs[-1] @ params[f"W{n_layers}"] + params[f"b{n_layers}"]
    loss = np.mean(sb * (np.logaddexp(0, z) - yb * z))
    loss += 0.5 * l2 * sum(np.sum(params[f"W{k}"] ** 2) for k in range(1, n_layers + 1))

    # backward (mean weighted cross-entropy)
    dz = sb * (sigmoid(z) - yb) / len(yb)
    grads = {
        f"W{n_layers}": inputs[-1].T @ dz + l2 * params[f"W{n_layers}"],
        f"b{n_layers}": np.array(dz.sum()),
    }
    delta = np.outer(dz, params[f"W{n_layers}"])
    for k in range(n_layers - 1, 0, -1):
        delta = delta * masks[k - 1]
        if dropout > 0:
            delta = delta / (1 - dropout)
        grads[f"W{k}"] = inputs[k - 1].T @ delta + l2 * params[f"W{k}"]
        grads[f"b{k}"] = delta.sum(axis=0)
        if k > 1:
            delta = delta @ params[f"W{k}"].T
    return loss, grads


def predict_mlp(tx, params):
    """Predicted probabilities of the network trained by `train_mlp`."""
    n_layers = len(params) // 2
    h = tx
    for k in range(1, n_layers):
        h = np.maximum(h @ params[f"W{k}"] + params[f"b{k}"], 0)
    return sigmoid(h @ params[f"W{n_layers}"] + params[f"b{n_layers}"])


# ---------------------------------------------------------------------------
# Feature expansion (fitted on the training data only)
# ---------------------------------------------------------------------------


def fit_expansion(tx, y, is_continuous, squares=True, n_interact=0):
    """Choose and standardize extra features: squares and pairwise interactions.

    Args:
        tx: (N, 1 + D) cleaned features, first column = bias.
        y: (N,) labels in {0, 1}.
        is_continuous: (D,) bool, continuous features (from `fit_cleaner`).
        squares: add the square of every continuous feature.
        n_interact: add all pairwise products of the n_interact features most
            correlated with the label (n_interact * (n_interact - 1) / 2 features).

    Returns:
        dict to pass to `apply_expansion`.
    """
    exp = {"square_idx": np.where(is_continuous)[0] + 1 if squares else [], "pairs": []}
    if n_interact > 1:
        corr = np.abs(tx[:, 1:].T @ (y - y.mean())) / len(y)
        top = np.argsort(-corr)[:n_interact] + 1
        exp["pairs"] = [(i, j) for a, i in enumerate(top) for j in top[a + 1 :]]
    extra = _expansion_features(tx, exp)
    exp["mean"] = extra.mean(axis=0)
    exp["std"] = extra.std(axis=0) + 1e-12
    return exp


def _expansion_features(tx, exp):
    """Raw (unscaled) extra features described by `exp`."""
    blocks = [tx[:, exp["square_idx"]] ** 2]
    if exp["pairs"]:
        i, j = np.array(exp["pairs"]).T
        blocks.append(tx[:, i] * tx[:, j])
    return np.hstack(blocks)


def apply_expansion(tx, exp):
    """Append the standardized extra features chosen by `fit_expansion`."""
    extra = (_expansion_features(tx, exp) - exp["mean"]) / exp["std"]
    return np.hstack([tx, extra])


# ---------------------------------------------------------------------------
# Gradient-boosted decision trees (histogram-based, logistic loss)
# ---------------------------------------------------------------------------


def fit_bins(x, max_bins=64, sample=100000, seed=0):
    """Bin edges of every feature (fitted on training data).

    Bin 0 is reserved for missing values (NaN), so that trees can split
    "missing" vs "observed". Features with few distinct values get one bin per
    value; the others get quantile bins computed on a random subsample of rows.

    Args:
        x: (N, D) features, may contain NaN.
        max_bins: total number of bins per feature, including the NaN bin (<= 256).
        sample: number of rows used to compute the quantiles.
        seed: random seed of the subsample.

    Returns:
        list of D arrays of increasing bin edges.
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(x), min(len(x), sample), replace=False)
    edges = []
    for j in range(x.shape[1]):
        col = x[rows, j]
        col = col[~np.isnan(col)]
        values = np.unique(col)
        if len(values) <= max_bins - 1:
            edges.append((values[:-1] + values[1:]) / 2)
        else:
            q = np.linspace(0, 1, max_bins)[1:-1]
            edges.append(np.unique(np.quantile(col, q)))
    return edges


def apply_bins(x, edges):
    """Bin index of every value (uint8): 0 for NaN, else 1 + number of edges <= value."""
    out = np.empty(x.shape, dtype=np.uint8)
    for j, e in enumerate(edges):
        col = x[:, j]
        b = np.searchsorted(e, col, side="right") + 1
        b[np.isnan(col)] = 0
        out[:, j] = b
    return out


def _histograms(xs_t, rows, local_node, g, h, n_nodes, n_bins):
    """Gradient / hessian histograms of several nodes at once.

    Args:
        xs_t: (K, M) binned features of the tree's rows (one row per feature).
        rows: indices (into M) of the rows to count.
        local_node: node number (0 .. n_nodes-1) of each of these rows.
        g, h: (M,) gradients and hessians.

    Returns:
        two arrays of shape (n_nodes, K, n_bins).
    """
    k = xs_t.shape[0]
    hg = np.empty((n_nodes, k, n_bins))
    hh = np.empty((n_nodes, k, n_bins))
    base = local_node * n_bins
    g_r, h_r = g[rows], h[rows]
    size = n_nodes * n_bins
    for j in range(k):
        key = base + xs_t[j, rows]
        hg[:, j, :] = np.bincount(key, weights=g_r, minlength=size).reshape(
            n_nodes, n_bins
        )
        hh[:, j, :] = np.bincount(key, weights=h_r, minlength=size).reshape(
            n_nodes, n_bins
        )
    return hg, hh


def _build_tree(
    xb, rows, feats, g, h, max_depth, min_child_weight, reg_lambda, lr, n_bins
):
    """Grow one regression tree on the gradients (depth-wise, exact on bins).

    Split gain (second-order / XGBoost formula):
        GL^2 / (HL + lambda) + GR^2 / (HR + lambda) - G^2 / (H + lambda),
    leaf value: -lr * G / (H + lambda).

    Returns:
        dict of node arrays: feature (-1 for leaves), threshold bin (go left if
        bin <= threshold), left / right children and value.
    """
    xs_t = np.ascontiguousarray(xb[np.ix_(rows, feats)].T)  # (K, M)
    gs, hs = g[rows], h[rows]
    m = len(rows)
    n_max = 2 ** (max_depth + 1)
    feature = np.full(n_max, -1)
    threshold = np.zeros(n_max, dtype=np.int64)
    left = np.zeros(n_max, dtype=np.int64)
    right = np.zeros(n_max, dtype=np.int64)
    value = np.zeros(n_max)

    all_rows = np.arange(m)
    hg, hh = _histograms(xs_t, all_rows, np.zeros(m, dtype=np.int64), gs, hs, 1, n_bins)
    # level: list of (node id, row indices, grad hist, hess hist)
    level = [(0, all_rows, hg[0], hh[0])]
    next_id = 1
    for depth in range(max_depth + 1):
        splits = []
        for node, node_rows, ng, nh in level:
            G, H = ng[0].sum(), nh[0].sum()
            value[node] = -lr * G / (H + reg_lambda)
            if depth == max_depth or H < 2 * min_child_weight:
                continue
            gl = np.cumsum(ng, axis=1)[:, :-1]
            hl = np.cumsum(nh, axis=1)[:, :-1]
            gr, hr = G - gl, H - hl
            gain = gl**2 / (hl + reg_lambda) + gr**2 / (hr + reg_lambda)
            gain[(hl < min_child_weight) | (hr < min_child_weight)] = -np.inf
            best = int(np.argmax(gain))
            j, b = divmod(best, n_bins - 1)
            if not gain[j, b] - G**2 / (H + reg_lambda) > 1e-10:
                continue
            go_left = xs_t[j, node_rows] <= b
            feature[node], threshold[node] = feats[j], b
            left[node], right[node] = next_id, next_id + 1
            next_id += 2
            splits.append((node, j, node_rows[go_left], node_rows[~go_left], ng, nh))
        if not splits:
            break
        # Histograms of the smaller child of each split, the sibling by subtraction
        small_rows, small_ids = [], []
        for i, (_, _, lr_rows, rr_rows, _, _) in enumerate(splits):
            small = lr_rows if len(lr_rows) <= len(rr_rows) else rr_rows
            small_rows.append(small)
            small_ids.append(np.full(len(small), i, dtype=np.int64))
        hg, hh = _histograms(
            xs_t,
            np.concatenate(small_rows),
            np.concatenate(small_ids),
            gs,
            hs,
            len(splits),
            n_bins,
        )
        level = []
        for i, (node, _, lr_rows, rr_rows, ng, nh) in enumerate(splits):
            if len(lr_rows) <= len(rr_rows):
                lg, lh, rg, rh = hg[i], hh[i], ng - hg[i], nh - hh[i]
            else:
                rg, rh, lg, lh = hg[i], hh[i], ng - hg[i], nh - hh[i]
            level.append((left[node], lr_rows, lg, lh))
            level.append((right[node], rr_rows, rg, rh))
    return {
        "feature": feature[:next_id],
        "threshold": threshold[:next_id],
        "left": left[:next_id],
        "right": right[:next_id],
        "value": value[:next_id],
    }


def _predict_tree(tree, xb):
    """Output of one tree for binned features xb (N, D)."""
    node = np.zeros(len(xb), dtype=np.int64)
    rows = np.arange(len(xb))
    feature = tree["feature"]
    while True:
        f = feature[node]
        internal = f >= 0
        if not internal.any():
            break
        idx = rows[internal]
        nd = node[idx]
        go_left = xb[idx, f[internal]] <= tree["threshold"][nd]
        node[idx] = np.where(go_left, tree["left"][nd], tree["right"][nd])
    return tree["value"][node]


def train_gbdt(
    y,
    xb,
    n_trees=300,
    lr=0.1,
    max_depth=6,
    min_child_weight=10.0,
    reg_lambda=1.0,
    subsample=0.7,
    colsample=0.5,
    n_bins=64,
    seed=0,
):
    """Gradient boosting of regression trees for the logistic loss.

    At each round, the gradient g = p - y and hessian h = p (1 - p) of the
    logistic loss are computed for the current scores, and a tree fitted to
    them (second-order step) is added with shrinkage `lr`. Each tree sees a
    random fraction `subsample` of the rows and `colsample` of the features.

    Args:
        y: (N,) labels in {0, 1}.
        xb: (N, D) binned features (uint8, from `apply_bins`).
        n_trees: number of boosting rounds.
        lr: shrinkage (learning rate).
        max_depth: maximum depth of each tree.
        min_child_weight: minimum hessian sum in a leaf.
        reg_lambda: L2 regularization of the leaf values.
        subsample: fraction of rows used by each tree.
        colsample: fraction of features used by each tree.
        n_bins: number of histogram bins (>= max bin index + 1).
        seed: random seed.

    Returns:
        dict {"base": initial score, "trees": list of trees}.
    """
    rng = np.random.default_rng(seed)
    n, d = xb.shape
    p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    base = float(np.log(p0 / (1 - p0)))
    f = np.full(n, base)
    trees = []
    n_rows = max(1, int(subsample * n))
    n_feats = max(1, int(colsample * d))
    for _ in range(n_trees):
        p = sigmoid(f)
        g, h = p - y, p * (1 - p)
        rows = (
            np.sort(rng.choice(n, n_rows, replace=False))
            if subsample < 1
            else np.arange(n)
        )
        feats = np.sort(rng.choice(d, n_feats, replace=False))
        tree = _build_tree(
            xb, rows, feats, g, h, max_depth, min_child_weight, reg_lambda, lr, n_bins
        )
        f += _predict_tree(tree, xb)
        trees.append(tree)
    return {"base": base, "trees": trees}


def predict_gbdt(xb, model, checkpoints=None):
    """Predicted probabilities of a boosted model.

    Args:
        xb: (N, D) binned features.
        model: output of `train_gbdt`.
        checkpoints: optional list of numbers of trees; if given, returns a dict
            {n_trees: probabilities using only the first n_trees trees}.
    """
    f = np.full(len(xb), model["base"])
    if checkpoints is None:
        for tree in model["trees"]:
            f += _predict_tree(tree, xb)
        return sigmoid(f)
    out = {}
    for t, tree in enumerate(model["trees"], start=1):
        f += _predict_tree(tree, xb)
        if t in checkpoints:
            out[t] = sigmoid(f)
    return out
