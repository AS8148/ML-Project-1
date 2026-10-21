"""Feature pipeline for the BRFSS 2015 data: load, decode, select, encode, scale."""

import os

import numpy as np

from helpers import load_csv_data

# Yes/no questions: 1 = yes, 2 = no.
YES_NO = """LANDLINE HLTHPLN1 MEDCOST BPMEDS BLOODCHO TOLDHI2 CVDSTRK3 ASTHMA3
CHCSCNCR CHCOCNCR CHCCOPD1 HAVARTH3 ADDEPEV2 CHCKIDNY VETERAN3 INTERNET PREGNANT
QLACTLM2 USEEQUIP BLIND DECIDE DIFFWALK DIFFDRES DIFFALON STOPSMK2 EXERANY2
LMTJOIN3 ARTHDIS2 FLUSHOT6 PNEUVAC3 HIVTST6 INSULIN DIABEYE DIABEDU CVDASPRN
RDUCHART RDUCSTRK""".split()

# Binary, ordinal, count and continuous variables, used as numbers.
NUMERIC = """SEX GENHLTH CHECKUP1 CHOLCHK EDUCA INCOME2 USENOW3 LASTSMK2 ARTHSOCL
SEATBELT EYEEXAM _PACAT1 PHYSHLTH MENTHLTH POORHLTH CHILDREN NUMADULT HHADULT
DIABAGE2 JOINPAIN ALCDAY5 AVEDRNK2 DRNK3GE5 MAXDRNKS _DRNKWEK BLDSUGAR FEETCHK2
DOCTDIAB CHKHEMO3 FEETCHK _AGE80 HTM4 WTKG3 _BMI5 FTJUDA1_ FRUTDA1_ BEANDAY_
GRENDAY_ ORNGDAY_ VEGEDA1_ _FRUTSUM _VEGESUM METVL11_ METVL21_ STRFREQ_ PA1MIN_
PA1VIGM_""".split()

# Nominal variables, one-hot encoded.
NOMINAL = """PERSDOC2 BPHIGH4 DIABETE3 MARITAL RENTHOM1 EMPLOY1 QSTVER MSCODE
_RACEGR3 _SMOKER3 _STATE""".split()

# Cardiac rehabilitation, asked only after a heart attack: used as an "answered" flag.
REHAB = "HAREHAB1"

USED = YES_NO + NUMERIC + NOMINAL + [REHAB]

NAN = np.nan
# Answer codes replaced by a value (don't know / refused -> NaN, none -> 0).
RECODES = [
    (YES_NO, {2: 0, 7: NAN, 9: NAN}),
    (["SEX"], {1: 0, 2: 1}),
    (
        """GENHLTH CHOLCHK EDUCA USENOW3 ARTHSOCL PERSDOC2 BPHIGH4 DIABETE3 MARITAL
        RENTHOM1""".split(),
        {7: NAN, 9: NAN},
    ),
    (["CHECKUP1", "EYEEXAM"], {7: NAN, 8: 5, 9: NAN}),  # never: after the last interval
    (["SEATBELT"], {7: NAN, 8: 1, 9: NAN}),  # 8 = never rides in a car
    (["EMPLOY1", "_PACAT1", "_RACEGR3", "_SMOKER3"], {9: NAN}),
    (
        "PHYSHLTH MENTHLTH POORHLTH DRNK3GE5 DOCTDIAB FEETCHK".split(),
        {77: NAN, 88: 0, 99: NAN},
    ),
    (["CHILDREN"], {88: 0, 99: NAN}),
    (["CHKHEMO3"], {77: NAN, 88: 0, 98: 0, 99: NAN}),
    (
        "INCOME2 LASTSMK2 HHADULT JOINPAIN AVEDRNK2 MAXDRNKS".split(),
        {77: NAN, 99: NAN},
    ),
    (["DIABAGE2"], {98: NAN, 99: NAN}),
    (["STRFREQ_"], {99: NAN}),  # code 99000, stored divided by 1000
    (["_DRNKWEK"], {99900: NAN}),
]


def load_data(data_dir, decoder=None):
    """Return x_train, x_test, labels y in {0, 1} and the test ids.

    x_train and x_test are decoder(raw matrix, column names), by default decode
    (columns USED). The gradient boosting uses codebook.decode, the ablation study
    select_raw. The parsed CSVs are cached in data_dir/raw.npz, because parsing
    takes minutes.
    """
    cache = os.path.join(data_dir, "raw.npz")
    if not os.path.exists(cache):
        x_train, x_test, y, _, test_ids = load_csv_data(data_dir)
        tmp = cache.replace(".npz", "_tmp.npz")  # renamed once fully written
        np.savez(tmp, x_train=x_train, x_test=x_test, y=y, test_ids=test_ids)
        os.replace(tmp, cache)
    with open(os.path.join(data_dir, "x_train.csv")) as f:
        columns = f.readline().strip().split(",")[1:]
    if decoder is None:
        decoder = decode
    with np.load(cache) as raw:
        x_train = decoder(raw["x_train"], columns)
        x_test = decoder(raw["x_test"], columns)
        return x_train, x_test, (raw["y"] == 1).astype(float), raw["test_ids"]


def select_raw(x, columns):
    """Return the columns USED of the raw matrix x, with the answer codes kept raw."""
    return x[:, [columns.index(c) for c in USED]]


def _per_period(a, factors):
    """Decode answers kxx (xx times per period k) to xx * factors[k]; 888 = never."""
    out = np.full(a.shape, NAN)
    for k, factor in factors.items():
        m = a // 100 == k
        out[m] = a[m] % 100 * factor
    out[a == 888] = 0
    return out


def decode(x, columns):
    """Return the columns USED of the raw matrix x, with the survey codes decoded."""
    d = {c: x[:, columns.index(c)].astype(float) for c in USED}
    for names, codes in RECODES:
        for c in names:
            raw = d[c].copy()
            for code, value in codes.items():
                d[c][raw == code] = value
    d["_DRNKWEK"] /= 100  # two implied decimals
    d["ALCDAY5"] = _per_period(d["ALCDAY5"], {1: 30 / 7, 2: 1})  # days in 30
    for c in ("BLDSUGAR", "FEETCHK2"):  # times per day
        d[c] = _per_period(d[c], {1: 1, 2: 1 / 7, 3: 1 / 30, 4: 1 / 365})

    # Questions skipped because the answer is known to be zero.
    skips = [
        (["POORHLTH"], (d["PHYSHLTH"] == 0) & (d["MENTHLTH"] == 0)),
        (["AVEDRNK2", "DRNK3GE5", "MAXDRNKS"], d["ALCDAY5"] == 0),
        (["METVL11_", "METVL21_", "PA1MIN_", "PA1VIGM_"], d["EXERANY2"] == 0),
    ]
    for names, skipped in skips:
        for c in names:
            d[c][np.isnan(d[c]) & skipped] = 0
    return np.column_stack([d[c] for c in USED])


def build_features(x_fit, x_apply, min_level=200, indicator_rate=0.1, clip=8.0):
    """Encode two decoded matrices with statistics estimated on x_fit only.

    Numeric columns: median imputation, plus a missing indicator when at least
    indicator_rate of x_fit is missing (identical indicators kept once).
    Nominal columns: one-hot with missing as a level; levels with fewer than
    min_level rows and the most frequent level are dropped. HAREHAB1 becomes
    an "answered" flag. Features are standardized, clipped to [-clip, clip]
    and a bias column is prepended. Returns (tx_fit, tx_apply, names).
    """
    n_num = len(YES_NO) + len(NUMERIC)
    median = np.nan_to_num(np.nanmedian(x_fit[:, :n_num], axis=0))
    missing = np.isnan(x_fit[:, :n_num])
    ind = np.flatnonzero(missing.mean(axis=0) >= indicator_rate)
    _, first = np.unique(missing[:, ind], axis=1, return_index=True)
    ind = ind[np.sort(first)]
    names = USED[:n_num] + [f"{USED[j]}_missing" for j in ind]

    levels = []
    for j in range(n_num, n_num + len(NOMINAL)):
        codes = np.nan_to_num(x_fit[:, j], nan=-1)  # missing is level -1
        values, counts = np.unique(codes, return_counts=True)
        keep = counts >= min_level
        keep[np.argmax(counts)] = False
        levels.append((j, values[keep]))
        names += [f"{USED[j]}={v:g}" for v in values[keep]]
    names.append(f"{REHAB}_answered")

    def encode(x):
        num = x[:, :n_num]
        parts = [np.where(np.isnan(num), median, num), np.isnan(num[:, ind])]
        for j, values in levels:
            parts.append(np.nan_to_num(x[:, j], nan=-1)[:, None] == values)
        parts.append(~np.isnan(x[:, [USED.index(REHAB)]]))
        return np.hstack(parts)

    a_fit = encode(x_fit)
    mean, std = a_fit.mean(axis=0), a_fit.std(axis=0)
    std[std < 1e-12] = 1

    def scale(a):
        a = np.clip((a - mean) / std, -clip, clip)
        return np.column_stack([np.ones(len(a)), a])

    return scale(a_fit), scale(encode(x_apply)), ["bias"] + names
