"""Data loading and cleaning for the BRFSS heart-disease (MICHD) dataset.

The cleaning pipeline follows the usual "fit on train, apply everywhere" pattern
so that it can be used safely inside cross-validation (no information from the
validation fold leaks into the statistics used for cleaning):

    columns, x = drop_columns(x, columns)            # stateless
    x = replace_missing_codes(x, columns)            # stateless
    params = fit_cleaner(x_train, columns)           # learns statistics on train only
    x_train_clean = transform(x_train, params)
    x_val_clean = transform(x_val, params)
"""

import os

import numpy as np

from codebook import MISSING_CODES, NOMINAL_COLUMNS, RECODE, UNIT_CODED, ZERO_CODES
from helpers import load_csv_data

# data/dataset of the repository, whatever the current directory
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset"
)

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_data(data_dir=DATA_DIR):
    """Load the train/test data, using a .npz cache to avoid re-parsing the CSVs.

    Parsing the CSV files takes ~30s, the cache (data_dir/cache.npz) loads in
    a couple of seconds.

    Args:
        data_dir: folder containing x_train.csv, y_train.csv and x_test.csv.

    Returns:
        x_train: (N, D) array of raw features.
        x_test: (N_test, D) array of raw features.
        y_train: (N,) array of labels in {-1, 1}.
        train_ids, test_ids: sample ids.
        columns: (D,) array of column names (without the Id column).
    """
    cache = os.path.join(data_dir, "cache.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return (
            d["x_train"],
            d["x_test"],
            d["y_train"],
            d["train_ids"],
            d["test_ids"],
            d["columns"],
        )

    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_dir)
    with open(os.path.join(data_dir, "x_train.csv")) as f:
        columns = np.array(f.readline().strip().split(",")[1:])
    np.savez(
        cache,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        train_ids=train_ids,
        test_ids=test_ids,
        columns=columns,
    )
    return x_train, x_test, y_train, train_ids, test_ids, columns


# ---------------------------------------------------------------------------
# Domain knowledge about the columns (BRFSS 2015 codebook)
# ---------------------------------------------------------------------------

# Survey administration / sampling design: carry no health information.
# fmt: off
ADMIN_COLUMNS = [
    "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO",
    "_PSU", "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
    "CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS", "CSTATE", "LANDLINE",
    "QSTVER", "QSTLANG", "MSCODE", "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE",
    "_CLLCPWT", "_DUALUSE", "_DUALCOR", "_LLCPWT",
]
# fmt: on

# Raw answers with a special encoding (e.g. 1xx = per day, 2xx = per week) for
# which the survey already provides a clean calculated version, dates, and
# duplicated derived variables.
# fmt: off
REDUNDANT_COLUMNS = [
    "ALCDAY5",   # -> DROCDY3_
    "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1",  # -> *DA1_ / *DAY_
    "EXEROFT1", "EXEROFT2", "EXERHMM1", "EXERHMM2", "STRENGTH",  # -> PAFREQ*_, PADUR*_, STRFREQ_
    "EXRACT11", "EXRACT21",  # activity type (76 nominal codes), summarized by METVL*_ / ACTIN*_
    "WEIGHT2", "HEIGHT3", "HTIN4",  # -> WTKG3, HTM4, _BMI5
    "FLSHTMY2", "HIVTSTD3",  # dates (MMYYYY)
    "_PRACE1", "_MRACE1", "_HISPANC", "_RACEG21", "_RACEGR3", "_RACE_G1",  # -> _RACE
    "_CRACE1", "_CPRACE", "_CHISPNC",  # race of the randomly selected child
    "PCDMDECN",  # multiple-response code
]
# fmt: on


# ---------------------------------------------------------------------------
# Stateless steps
# ---------------------------------------------------------------------------


def drop_columns(x, columns, to_drop=None):
    """Remove the given columns (default: administrative + redundant ones).

    Args:
        x: (N, D) array.
        columns: (D,) array of column names.
        to_drop: iterable of column names, defaults to ADMIN + REDUNDANT columns.

    Returns:
        x: (N, D') array without the dropped columns.
        columns: (D',) array of the remaining column names.
    """
    if to_drop is None:
        to_drop = ADMIN_COLUMNS + REDUNDANT_COLUMNS
    keep = ~np.isin(columns, list(to_drop))
    return x[:, keep], columns[keep]


def replace_missing_codes(x, columns):
    """Decode the BRFSS answer codes using the 2015 codebook (see codebook.py).

    For each column, in this order:
      1. MISSING_CODES ("don't know", "refused", "not applicable") -> NaN;
      2. ZERO_CODES ("none", "never") -> 0;
      3. RECODE: explicit value mappings for code sets that are not ordered
         (e.g. BPHIGH4: 1 = yes, 3 = no, 4 = borderline -> 2, 0, 1);
      4. UNIT_CODED: <unit><count> answers converted to a common unit
         (e.g. BLDSUGAR 203 = 3 times per week -> 156 times per year).

    Args:
        x: (N, D) array of raw features (not modified).
        columns: (D,) array of column names.

    Returns:
        (N, D) float array with decoded values and NaN for missing answers.
    """
    x = x.astype(float, copy=True)
    for j, name in enumerate(columns):
        col = x[:, j]
        if name in MISSING_CODES:
            col[np.isin(col, MISSING_CODES[name])] = np.nan
        if name in ZERO_CODES:
            col[np.isin(col, ZERO_CODES[name])] = 0
        if name in RECODE:
            original = col.copy()
            for old, new in RECODE[name].items():
                col[original == old] = new
        if name in UNIT_CODED:
            unit = col // 100
            count = col % 100
            decoded = np.full_like(col, np.nan)
            decoded[col == 0] = 0
            for u, mult in UNIT_CODED[name].items():
                decoded[unit == u] = count[unit == u] * mult
            col[:] = decoded
    return x


# ---------------------------------------------------------------------------
# Fit / transform (statistics learned on the training data only)
# ---------------------------------------------------------------------------


def fit_cleaner(
    x,
    columns,
    max_nan_frac=0.5,
    clip_quantiles=(0.005, 0.995),
    add_nan_indicators=False,
    extra_nominal=(),
):
    """Learn the cleaning parameters on (training) data.

    Steps learned here and applied by `transform`:
      1. drop columns with more than `max_nan_frac` missing values;
      2. one-hot encode NOMINAL_COLUMNS (a missing value gives all zeros);
      3. for the other (numeric) columns: optionally add a 0/1 "was missing"
         indicator, clip outliers of continuous columns to train quantiles,
         impute missing values with the train median;
      4. drop constant columns and standardize (zero mean, unit variance).

    Args:
        x: (N, D) array after `replace_missing_codes`.
        columns: (D,) array of column names.
        max_nan_frac: float in [0, 1], columns with a larger missing fraction are dropped.
        clip_quantiles: (low, high) quantiles used to clip continuous columns, or None.
        add_nan_indicators: bool, add a missing-value indicator per numeric column
            that has missing values.
        extra_nominal: additional column names to one-hot encode (e.g. "_STATE").

    Returns:
        dict of parameters to pass to `transform`. Besides the fitted statistics
        it contains "feature_names" and "is_continuous" (bool per output feature).
    """
    nan_frac = np.isnan(x).mean(axis=0)
    keep = nan_frac <= max_nan_frac

    nominal_idx, categories = [], []
    numeric_idx = []
    for j in np.where(keep)[0]:
        if columns[j] in NOMINAL_COLUMNS or columns[j] in extra_nominal:
            nominal_idx.append(j)
            categories.append(np.unique(x[~np.isnan(x[:, j]), j]))
        else:
            numeric_idx.append(j)
    numeric_idx = np.array(numeric_idx, dtype=int)

    xn = x[:, numeric_idx]
    indicator_idx = (
        np.where(np.isnan(xn).any(axis=0))[0]
        if add_nan_indicators
        else np.array([], dtype=int)
    )

    # Clip only continuous columns (many distinct values), not categorical codes
    n_unique = np.array([len(np.unique(c[~np.isnan(c)])) for c in xn.T])
    cont = n_unique > 20
    lower = np.full(len(numeric_idx), -np.inf)
    upper = np.full(len(numeric_idx), np.inf)
    if clip_quantiles is not None:
        lower[cont] = np.nanquantile(xn[:, cont], clip_quantiles[0], axis=0)
        upper[cont] = np.nanquantile(xn[:, cont], clip_quantiles[1], axis=0)

    medians = np.nanmedian(xn, axis=0)

    params = {
        "numeric_idx": numeric_idx,
        "nominal_idx": nominal_idx,
        "categories": categories,
        "indicator_idx": indicator_idx,
        "lower": lower,
        "upper": upper,
        "medians": medians,
        "input_columns": columns,
    }

    # Compute standardization statistics on the transformed training data
    z = _build_features(x, params)
    mean, std = z.mean(axis=0), z.std(axis=0)
    params["keep_features"] = std > 1e-12  # drop constant columns
    params["mean"] = mean[params["keep_features"]]
    params["std"] = std[params["keep_features"]]
    params["feature_names"] = _feature_names(params)[params["keep_features"]]
    is_cont = np.r_[cont, np.zeros(z.shape[1] - len(cont), dtype=bool)]
    params["is_continuous"] = is_cont[params["keep_features"]]
    return params


def _build_features(x, params):
    """Apply clipping, imputation, indicators and one-hot encoding (no scaling)."""
    xn = x[:, params["numeric_idx"]]
    blocks = []
    if len(params["indicator_idx"]):
        blocks.append(np.isnan(xn[:, params["indicator_idx"]]).astype(float))
    xn = np.clip(xn, params["lower"], params["upper"])
    xn = np.where(np.isnan(xn), params["medians"], xn)
    blocks.insert(0, xn)
    for j, cats in zip(params["nominal_idx"], params["categories"]):
        blocks.append((x[:, [j]] == cats[None, :]).astype(float))
    return np.hstack(blocks)


def _feature_names(params):
    """Names of the columns produced by `_build_features`."""
    cols = params["input_columns"]
    names = list(cols[params["numeric_idx"]])
    names += [
        f"{cols[params['numeric_idx'][i]]}_isnan" for i in params["indicator_idx"]
    ]
    for j, cats in zip(params["nominal_idx"], params["categories"]):
        names += [f"{cols[j]}={int(c)}" for c in cats]
    return np.array(names)


def transform(x, params, add_bias=True):
    """Apply the cleaning learned by `fit_cleaner` to any data (train, val or test).

    Args:
        x: (N, D) array after `replace_missing_codes`, same columns as in fit.
        params: dict returned by `fit_cleaner`.
        add_bias: bool, prepend a column of ones (intercept).

    Returns:
        (N, D') cleaned and standardized feature matrix.
    """
    z = _build_features(x, params)[:, params["keep_features"]]
    z = (z - params["mean"]) / params["std"]
    if add_bias:
        z = np.c_[np.ones(len(z)), z]
    return z
