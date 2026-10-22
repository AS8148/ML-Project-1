"""Decoding of the BRFSS 2015 columns with the CDC codebook, for the gradient boosting.

Written by Samuel Bachmann, team member: the tables and the decoding steps are those
of his codebook.py and data_cleaning.py, restricted to the columns kept.

Source: "Behavioral Risk Factor Surveillance System, 2015 Codebook Report,
Land-Line and Cell-Phone data" (CDC, August 2016),
https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf

MISSING_CODES was generated from the value tables of the codebook: every value
whose label is "Don't know / Not sure", "Refused", "Missing" or
"Not applicable" (checked by hand afterwards). Variables with implied decimal
places (e.g. MAXVO2_, PAFREQ1_) are already divided by 100 or 1000 in the
provided CSV files, so their codes are given in the scale of the data
(99900 -> 999, 99000 -> 99).
"""

import numpy as np

# Columns dropped by decode. First, survey administration and sampling design, which
# carry no health information (the state, _STATE, is kept).
ADMIN_COLUMNS = """FMONTH IDATE IMONTH IDAY IYEAR DISPCODE SEQNO _PSU CTELENUM PVTRESD1
COLGHOUS STATERES CELLFON3 LADULT CTELNUM1 CELLFON2 CADULT PVTRESD2 CCLGHOUS CSTATE
LANDLINE QSTVER QSTLANG MSCODE _STSTR _STRWT _RAWRAKE _WT2RAKE _CLLCPWT _DUALUSE
_DUALCOR _LLCPWT""".split()

# Then raw answers for which the survey provides a clean calculated version: ALCDAY5
# (DROCDY3_), fruits and vegetables (*DA1_, *DAY_), exercise frequency, duration and
# type (PAFREQ*_, PADUR*_, STRFREQ_, METVL*_, ACTIN*_), weight and height (WTKG3,
# HTM4, _BMI5). Then two dates (MMYYYY), race variables duplicating _RACE, the race
# of the randomly selected child and a multiple-response code.
REDUNDANT_COLUMNS = """ALCDAY5 FRUITJU1 FRUIT1 FVBEANS FVGREEN FVORANG VEGETAB1
EXEROFT1 EXEROFT2 EXERHMM1 EXERHMM2 STRENGTH EXRACT11 EXRACT21 WEIGHT2 HEIGHT3 HTIN4
FLSHTMY2 HIVTSTD3 _PRACE1 _MRACE1 _HISPANC _RACEG21 _RACEGR3 _RACE_G1 _CRACE1
_CPRACE _CHISPNC PCDMDECN""".split()

# Codes meaning "don't know", "refused", "missing" or "not applicable" -> NaN.
MISSING_CODES = {
    "HHADULT": [77, 99],
    "GENHLTH": [7, 9],
    "PHYSHLTH": [77, 99],
    "MENTHLTH": [77, 99],
    "POORHLTH": [77, 99],
    "HLTHPLN1": [7, 9],
    "PERSDOC2": [7, 9],
    "MEDCOST": [7, 9],
    "CHECKUP1": [7, 9],
    "BPHIGH4": [7, 9],
    "BPMEDS": [7, 9],
    "BLOODCHO": [7, 9],
    "CHOLCHK": [7, 9],
    "TOLDHI2": [7, 9],
    "CVDSTRK3": [7, 9],
    "ASTHMA3": [7, 9],
    "ASTHNOW": [7, 9],
    "CHCSCNCR": [7, 9],
    "CHCOCNCR": [7, 9],
    "CHCCOPD1": [7, 9],
    "HAVARTH3": [7, 9],
    "ADDEPEV2": [7, 9],
    "CHCKIDNY": [7, 9],
    "DIABETE3": [7, 9],
    "DIABAGE2": [98, 99],
    "MARITAL": [9],
    "EDUCA": [9],
    "RENTHOM1": [7, 9],
    "NUMHHOL2": [7, 9],
    "NUMPHON2": [7, 9],
    "CPDEMO1": [7, 9],
    "VETERAN3": [7, 9],
    "EMPLOY1": [9],
    "CHILDREN": [99],
    "INCOME2": [77, 99],
    "INTERNET": [7, 9],
    "PREGNANT": [7, 9],
    "QLACTLM2": [7, 9],
    "USEEQUIP": [7, 9],
    "BLIND": [7, 9],
    "DECIDE": [7, 9],
    "DIFFWALK": [7, 9],
    "DIFFDRES": [7, 9],
    "DIFFALON": [7, 9],
    "SMOKE100": [7, 9],
    "SMOKDAY2": [7, 9],
    "STOPSMK2": [7, 9],
    "LASTSMK2": [77, 99],
    "USENOW3": [7, 9],
    "AVEDRNK2": [77, 99],
    "DRNK3GE5": [77, 99],
    "MAXDRNKS": [77, 99],
    "EXERANY2": [7, 9],
    "LMTJOIN3": [7, 9],
    "ARTHDIS2": [7, 9],
    "ARTHSOCL": [7, 9],
    "JOINPAIN": [77, 99],
    "SEATBELT": [7, 9],
    "FLUSHOT6": [7, 9],
    "IMFVPLAC": [77, 99],
    "PNEUVAC3": [7, 9],
    "HIVTST6": [7, 9],
    "WHRTST10": [77, 99],
    "PDIABTST": [7, 9],
    "PREDIAB1": [7, 9],
    "INSULIN": [9],
    "BLDSUGAR": [777, 999],
    "FEETCHK2": [777, 999],
    "DOCTDIAB": [77, 99],
    "CHKHEMO3": [77, 99],
    "FEETCHK": [77, 99],
    "EYEEXAM": [7, 9],
    "DIABEYE": [7, 9],
    "DIABEDU": [7, 9],
    "CAREGIV1": [7, 9],
    "CRGVREL1": [77, 99],
    "CRGVLNG1": [7, 9],
    "CRGVHRS1": [7, 9],
    "CRGVPRB1": [77, 99],
    "CRGVPERS": [7, 9],
    "CRGVHOUS": [7, 9],
    "CRGVMST2": [7, 9],
    "CRGVEXPT": [7, 9],
    "VIDFCLT2": [7, 8],
    "VIREDIF3": [7],
    "VIPRFVS2": [7, 9],
    "VINOCRE2": [8, 77, 99],
    "VIEYEXM2": [7, 8, 9],
    "VIINSUR2": [7, 8, 9],
    "VICTRCT4": [7],
    "VIGLUMA2": [7],
    "VIMACDG2": [7],
    "CIMEMLOS": [7, 9],
    "CDHOUSE": [7, 9],
    "CDASSIST": [7, 9],
    "CDHELP": [7, 9],
    "CDSOCIAL": [7, 9],
    "CDDISCUS": [7, 9],
    "WTCHSALT": [7, 9],
    "LONGWTCH": [777, 999],
    "DRADVISE": [7, 9],
    "ASTHMAGE": [98, 99],
    "ASATTACK": [7],
    "ASERVIST": [98],
    "ASDRVIST": [98],
    "ASRCHKUP": [98, 99],
    "ASACTLIM": [777, 999],
    "ASYMPTOM": [7, 9],
    "ASNOSLEP": [7],
    "ASTHMED3": [7, 9],
    "ASINHALR": [7, 9],
    "HAREHAB1": [7, 9],
    "STREHAB1": [7, 9],
    "CVDASPRN": [7, 9],
    "ASPUNSAF": [7, 9],
    "RLIVPAIN": [7, 9],
    "RDUCHART": [7, 9],
    "RDUCSTRK": [7, 9],
    "ARTTODAY": [7, 9],
    "ARTHWGT": [7, 9],
    "ARTHEXER": [7, 9],
    "ARTHEDU": [7, 9],
    "TETANUS": [7, 9],
    "HPVADVC2": [7, 9],
    "HPVADSHT": [77, 99],
    "SHINGLE2": [7, 9],
    "HADMAM": [7, 9],
    "HOWLONG": [7, 9],
    "HADPAP2": [7, 9],
    "LASTPAP2": [7, 9],
    "HPVTEST": [7, 9],
    "HPLSTTST": [7, 9],
    "HADHYST2": [7, 9],
    "PROFEXAM": [7, 9],
    "LENGEXAM": [7, 9],
    "BLDSTOOL": [7, 9],
    "LSTBLDS3": [7, 9],
    "HADSIGM3": [7, 9],
    "HADSGCO1": [7, 9],
    "LASTSIG3": [7, 9],
    "PCPSAAD2": [7, 9],
    "PCPSADI1": [7, 9],
    "PCPSARE1": [7, 9],
    "PSATEST1": [7, 9],
    "PSATIME": [7, 9],
    "PCPSARS1": [7, 9],
    "PCPSADE1": [9],
    "SCNTMNY1": [7, 8, 9],
    "SCNTMEL1": [7, 8, 9],
    "SCNTPAID": [7, 9],
    "SCNTWRK1": [97, 99],
    "SCNTLPAD": [7, 9],
    "SCNTLWK1": [97, 99],
    "SXORIENT": [7, 9],
    "TRNSGNDR": [7, 9],
    "RCSGENDR": [9],
    "RCSRLTN2": [7, 9],
    "CASTHDX2": [7, 9],
    "CASTHNO2": [7, 9],
    "EMTSUPRT": [7, 9],
    "LSATISFY": [7, 9],
    "ADPLEASR": [77, 99],
    "ADDOWN": [77, 99],
    "ADSLEEP": [77, 99],
    "ADENERGY": [77, 99],
    "ADEAT1": [77, 99],
    "ADFAIL": [77, 99],
    "ADTHINK": [77, 99],
    "ADMOVE": [77, 99],
    "MISTMNT": [7, 9],
    "ADANXEV": [7, 9],
    "_RFHLTH": [9],
    "_HCVU651": [9],
    "_RFHYPE5": [9],
    "_CHOLCHK": [9],
    "_RFCHOL": [9],
    "_LTASTH1": [9],
    "_CASTHM1": [9],
    "_ASTHMS1": [9],
    "_RACE": [9],
    "_AGEG5YR": [14],
    "_AGE65YR": [3],
    "_RFBMI5": [9],
    "_CHLDCNT": [9],
    "_EDUCAG": [9],
    "_INCOMG": [9],
    "_SMOKER3": [9],
    "_RFSMOK3": [9],
    "DRNKANY5": [7, 9],
    "DROCDY3_": [900],
    "_RFBING5": [9],
    "_DRNKWEK": [99900],
    "_RFDRHV5": [9],
    "_FRTLT1": [9],
    "_VEGLT1": [9],
    "_TOTINDA": [9],
    "MAXVO2_": [999],
    "FC60_": [999],
    "PAFREQ1_": [99],
    "PAFREQ2_": [99],
    "STRFREQ_": [99],
    "PAMISS1_": [9],
    "_PACAT1": [9],
    "_PAINDX1": [9],
    "_PA150R2": [9],
    "_PA300R2": [9],
    "_PA30021": [9],
    "_PASTRNG": [9],
    "_PAREC1": [9],
    "_PASTAE1": [9],
    "_LMTACT1": [9],
    "_LMTWRK1": [9],
    "_LMTSCL1": [9],
    "_RFSEAT2": [9],
    "_RFSEAT3": [9],
    "_FLSHOT6": [9],
    "_PNEUMO2": [9],
    "_AIDTST3": [9],
}

# Codes meaning "none / never / zero" for count-like answers -> replaced by 0.
ZERO_CODES = {
    **dict.fromkeys(
        """PHYSHLTH MENTHLTH POORHLTH CHILDREN DRNK3GE5 DOCTDIAB FEETCHK ASERVIST
        ASDRVIST ASRCHKUP ADPLEASR ADDOWN ADSLEEP ADENERGY ADEAT1 ADFAIL ADTHINK
        ADMOVE""".split(),
        [88],
    ),
    "CHKHEMO3": [88, 98],  # 98 = never heard of the A1C test
    "SCNTWRK1": [98],  # 98 = zero hours
    "SCNTLWK1": [98],
    "ASACTLIM": [888],
    "ASYMPTOM": [8],  # 8 = not at any time (scale 1-5 = increasing frequency)
    "ASNOSLEP": [8],  # 8 = none
    "ASTHMED3": [8],  # 8 = never
    "ASINHALR": [8],  # 8 = never
    "BLDSUGAR": [888],  # 888 = never
    "FEETCHK2": [888],
}

# Code sets that are not ordered in a meaningful way: explicit value mappings.
# Codes not listed are kept unchanged; NaN means "not applicable".
RECODE = {
    # 1 = yes, 2 = yes but only during pregnancy, 3 = no, 4 = borderline
    # -> 0 = no, 1 = borderline, 2 = yes
    "BPHIGH4": {1: 2, 2: 0, 3: 0, 4: 1},
    "DIABETE3": {1: 2, 2: 0, 3: 0, 4: 1},
    # 1 = yes, 2 = yes during pregnancy, 3 = no -> 1 = yes, 0 = no
    "PREDIAB1": {1: 1, 2: 0, 3: 0},
    # 1 = only one personal doctor, 2 = more than one, 3 = none -> number of doctors
    "PERSDOC2": {1: 1, 2: 2, 3: 0},
    # 1-3 = yes (different vaccine types), 4 = no
    "TETANUS": {1: 1, 2: 1, 3: 1, 4: 0},
    # 3 = doctor refused when asked -> not vaccinated
    "HPVADVC2": {3: 2},
    # 1, 2 = yes (not stomach / stomach related), 3 = no
    "ASPUNSAF": {1: 1, 2: 1, 3: 0},
    # 1 = cataracts now, 2 = had them removed, 3 = no
    "VICTRCT4": {1: 1, 2: 0, 3: 0},
    # "Never" coded 8 after 1-4 (time since last visit) -> one step after 4
    "CHECKUP1": {8: 5},
    "EYEEXAM": {8: 5},
    # 97 = age 10 or younger
    "ASTHMAGE": {97: 10},
    # not applicable
    "SEATBELT": {8: np.nan},  # never drive or ride in a car
    "CAREGIV1": {8: np.nan},  # care recipient died in the past 30 days
    "FEETCHK2": {555: np.nan},  # no feet
    "LONGWTCH": {555: np.nan},  # all my life
}

# Answers encoded as <unit digit><2-digit count> (e.g. 203 = 3 times per week),
# converted to a common unit. {unit digit: multiplier}
UNIT_CODED = {
    "BLDSUGAR": {1: 365, 2: 52, 3: 12, 4: 1},  # -> times per year (888 = never)
    "FEETCHK2": {1: 365, 2: 52, 3: 12, 4: 1},  # -> times per year (888 = never)
    "LONGWTCH": {1: 1 / 365, 2: 1 / 52, 3: 1 / 12, 4: 1},  # -> years
}


def decode(x, columns):
    """Return x without ADMIN_COLUMNS and REDUNDANT_COLUMNS, with the codes decoded.

    x is a raw matrix, which is not modified, and columns its column names. In each
    column, in this order: MISSING_CODES become NaN, ZERO_CODES become 0, RECODE
    maps codes that are not ordered (e.g. BPHIGH4: 1 = yes, 3 = no, 4 = borderline
    -> 2, 0, 1) and UNIT_CODED converts <unit><count> answers to a common unit
    (e.g. BLDSUGAR 203 = 3 times per week -> 156 times per year).
    """
    dropped = ADMIN_COLUMNS + REDUNDANT_COLUMNS
    names = [c for c in columns if c not in dropped]
    # Selecting columns copies x, so the copy is decoded in place.
    x = x[:, [columns.index(c) for c in names]].astype(float, copy=False)
    for name, col in zip(names, x.T):
        if name in MISSING_CODES:
            col[np.isin(col, MISSING_CODES[name])] = np.nan
        if name in ZERO_CODES:
            col[np.isin(col, ZERO_CODES[name])] = 0
        if name in RECODE:
            original = col.copy()
            for old, new in RECODE[name].items():
                col[original == old] = new
        if name in UNIT_CODED:
            unit, count = col // 100, col % 100
            decoded = np.full_like(col, np.nan)
            decoded[col == 0] = 0
            for u, factor in UNIT_CODED[name].items():
                decoded[unit == u] = count[unit == u] * factor
            col[:] = decoded
    return x
