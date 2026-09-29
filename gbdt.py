"""Gradient boosted trees for the logistic loss, with histogram splits (NumPy only).

Features are binned once into quantiles (code 0 = missing). Trees are grown level
by level to a fixed depth; each split sends missing values to the better side.
"""

import numpy as np


def _bin(x, edges):
    """Return codes of shape (D, N): 0 if missing, else 1 + #edges <= value."""
    codes = np.zeros(x.shape[::-1], dtype=np.uint8)
    for j, e in enumerate(edges):
        known = ~np.isnan(x[:, j])
        codes[j, known] = 1 + np.searchsorted(e, x[known, j], side="right")
    return codes


def _edges(values, n_bins):
    """Bin edges: every distinct value if at most n_bins, else n_bins quantiles."""
    distinct = np.unique(values)
    quantiles = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return distinct[1:] if len(distinct) <= n_bins else np.unique(quantiles)


def _descend(codes, node, tree):
    """Move every sample one level down the tree (nodes are heap indices)."""
    feature, split, missing_right = tree[:3]
    c = codes[feature[node], np.arange(len(node))]
    return 2 * node + 1 + np.where(c == 0, missing_right[node], c > split[node])


def _predict_tree(codes, tree):
    """Return the value of the leaf reached by every sample."""
    node = np.zeros(codes.shape[1], dtype=int)
    for _ in range(len(tree[0]).bit_length()):  # depth: there are 2^depth - 1 splits
        node = _descend(codes, node, tree)
    return tree[3][node - len(tree[0])]


def _score(gh, lam):
    """Return g^2 / (h + lam): twice the loss decrease of a leaf with sums (g, h)."""
    return gh[0] ** 2 / (gh[1] + lam)


def _grow(codes, g, h, cols, depth, lam, min_hess, n_codes):
    """Grow one tree (feature, split, missing_right, leaf values) in heap order."""
    n_splits = 2**depth - 1
    feature = np.zeros(n_splits, dtype=int)
    split = np.full(n_splits, 255)  # known codes <= split go left (255: no split)
    missing_right = np.zeros(n_splits, dtype=bool)
    node = np.zeros(len(g), dtype=int)
    for level in range(depth):
        first, n_nodes = 2**level - 1, 2**level
        # Histograms of (g, h) over (node, feature, code), one bincount per feature.
        key, size = (node - first) * n_codes, n_nodes * n_codes
        index = [key + codes[j] for j in cols]
        hist = np.array([[np.bincount(i, w, size) for i in index] for w in (g, h)])
        hist = hist.reshape(2, len(cols), n_nodes, n_codes).transpose(0, 2, 1, 3)
        # Split after code t: known codes 1..t go left, missing ones on either side.
        miss, known = hist[..., :1], np.cumsum(hist[..., 1:], -1)
        total = (known[..., -1:] + miss)[:, :, None]  # (g/h, node, 1, feature, 1)
        left = np.stack([known + miss, known], 2)  # (g/h, node, side, feature, t)
        right = total - left
        gains = _score(left, lam) + _score(right, lam) - _score(total, lam)
        gains[(left[1] < min_hess) | (right[1] < min_hess)] = 0  # hessian sums
        gains = gains.reshape(n_nodes, -1)
        side, f, t = np.unravel_index(gains.argmax(1), (2, len(cols), n_codes - 1))
        has_split = gains.max(1) > 0
        nodes = first + np.arange(n_nodes)
        feature[nodes] = cols[f]
        split[nodes] = np.where(has_split, t + 1, 255)  # t counts from code 1
        missing_right[nodes] = has_split & (side == 1)
        node = _descend(codes, node, (feature, split, missing_right))
    g_leaf, h_leaf = (np.bincount(node - n_splits, w, n_splits + 1) for w in (g, h))
    return feature, split, missing_right, -g_leaf / (h_leaf + lam)


def fit(
    x,
    y,
    x_valid,
    y_valid,
    depth=4,
    rate=0.1,
    lam=10.0,
    min_hess=5.0,
    subsample=0.5,
    n_bins=32,
    max_trees=1000,
    patience=30,
    seed=0,
):
    """Boost trees on (x, y) until the log-loss on (x_valid, y_valid) stops improving.

    x has NaN for missing values, y is in {0, 1}. Each tree sees a random subsample of
    rows and of columns, has leaves penalized by lam and children with hessian sums
    >= min_hess, and is shrunk by rate. Returns (edges, base logit, rate, best trees).
    """
    rng = np.random.default_rng(seed)
    edges = [_edges(col[~np.isnan(col)], n_bins) for col in x.T]
    codes, codes_valid = _bin(x, edges), _bin(x_valid, edges)
    base = np.log(y.mean() / (1 - y.mean()))
    score, score_valid = np.full(len(y), base), np.full(len(y_valid), base)
    n_cols = max(1, int(subsample * x.shape[1]))
    trees, losses = [], []
    for _ in range(max_trees):
        p = 1 / (1 + np.exp(-score))
        rows = np.flatnonzero(rng.random(len(y)) < subsample)
        cols = np.sort(rng.choice(x.shape[1], n_cols, replace=False))
        g, h = (p - y)[rows], (p * (1 - p))[rows]
        tree = _grow(codes[:, rows], g, h, cols, depth, lam, min_hess, n_bins + 1)
        trees.append(tree)
        score += rate * _predict_tree(codes, tree)
        score_valid += rate * _predict_tree(codes_valid, tree)
        losses.append(np.mean(np.logaddexp(0, score_valid) - y_valid * score_valid))
        best = np.argmin(losses)
        if len(losses) - 1 - best >= patience:
            break
    return edges, base, rate, trees[: best + 1]


def predict(model, x):
    """Return the logits of the model for the rows of x."""
    edges, base, rate, trees = model
    codes = _bin(x, edges)
    score = np.full(len(x), base)
    for tree in trees:
        score += rate * _predict_tree(codes, tree)
    return score
