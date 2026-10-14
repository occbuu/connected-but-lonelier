"""PISA 2022 helpers: recoding, weighted statistics with BRR (Fay) replicate weights
and plausible values combined with Rubin's rules (OECD, 2009, PISA Data Analysis Manual)."""
import numpy as np
import pandas as pd

REGION = ["VNM", "SGP", "JPN", "KOR", "TAP", "HKG", "MAC", "THA", "MYS", "IDN", "PHL", "BRN", "KHM"]
NAMES = {"VNM": "Viet Nam", "SGP": "Singapore", "JPN": "Japan", "KOR": "Korea", "TAP": "Chinese Taipei",
         "HKG": "Hong Kong (China)", "MAC": "Macao (China)", "THA": "Thailand", "MYS": "Malaysia",
         "IDN": "Indonesia", "PHL": "Philippines", "BRN": "Brunei Darussalam", "KHM": "Cambodia"}
W = "W_FSTUWT"
RW = [f"W_FSTURWT{i}" for i in range(1, 81)]
PV_MATH = [f"PV{i}MATH" for i in range(1, 11)]
FAY = 0.5
BRR_FACTOR = 1 / (len(RW) * (1 - FAY) ** 2)  # = 1/20


def recode(df):
    d = df.copy()
    for c in [c for c in d.columns if c.startswith("ST322")]:
        d[c] = d[c].where(d[c].between(1, 5))          # 6 = not applicable (no device)
    for c in [c for c in d.columns if c.startswith("ST273")]:
        d[c] = d[c].where(d[c].between(1, 4))
    # Digital distraction index: mean of the two digital items (1 never .. 4 every lesson)
    d["DIGDIST"] = (5 - d[["ST273Q06JA", "ST273Q07JA"]]).mean(axis=1, skipna=False)
    # Non-digital disciplinary climate (Q01-Q05; higher = better climate). The official
    # DISCLIM index includes the two digital items, so it is not used alongside DIGDIST.
    nd = [f"ST273Q0{i}JA" for i in range(1, 6)]
    d["DISCLIM_ND"] = d[nd].mean(axis=1).where(d[nd].notna().sum(axis=1) >= 3)
    # Higher = more frequent distraction (1 never .. 4 every lesson)
    d["DIST_SELF"] = 5 - d["ST273Q06JA"]
    d["DIST_PEERS"] = 5 - d["ST273Q07JA"]
    d["DIST_SELF_MOST"] = (d["ST273Q06JA"] <= 2).astype(float).where(d["ST273Q06JA"].notna())
    d["DIST_PEERS_MOST"] = (d["ST273Q07JA"] <= 2).astype(float).where(d["ST273Q07JA"].notna())
    d["ONLINE_PRESSURE"] = d["ST322Q06JA"]           # pressured to be online/answer messages in class
    d["DEVICE_ANXIETY"] = d["ST322Q07JA"]            # nervous when device not near
    d["NOTIF_OFF_CLASS"] = d["ST322Q01JA"]           # self-regulation: notifications off in class
    d["NOTIF_OFF_SLEEP"] = d["ST322Q02JA"]
    d["DEVICE_NEAR_HOME"] = d["ST322Q03JA"]
    d["DEVICE_OPEN_CLASS"] = d["ST322Q04JA"]         # device open for notes/search in class
    d["ONLINE_PRESSURE_HIGH"] = (d["ST322Q06JA"] >= 4).astype(float).where(d["ST322Q06JA"].notna())
    d["DEVICE_ANXIETY_HIGH"] = (d["ST322Q07JA"] >= 4).astype(float).where(d["ST322Q07JA"].notna())
    d["EASILY_DISTRACTED"] = d["ST309Q02JA"].where(d["ST309Q02JA"].between(1, 5))
    d["FEMALE"] = (d["ST004D01T"] == 1).astype(float).where(d["ST004D01T"].isin([1, 2]))
    d["OWN_SMARTPHONE"] = (d["ST250Q04JA"] == 1).astype(float).where(d["ST250Q04JA"].isin([1, 2]))
    d["MATH_MEAN"] = d[PV_MATH].mean(axis=1)
    return d


def wmean(x, w):
    m = np.isfinite(x)
    return np.sum(x[m] * w[m]) / np.sum(w[m])


def brr_stat(d, fn):
    """Statistic fn(d, weight_col) with BRR standard error (no plausible values)."""
    est = fn(d, W)
    reps = np.array([fn(d, rw) for rw in RW])
    se = np.sqrt(BRR_FACTOR * np.sum((reps - est) ** 2, axis=0))
    return est, se


def weighted_mean_se(d, col):
    return brr_stat(d, lambda x, w: wmean(x[col].to_numpy(float), x[w].to_numpy(float)))


def zscore_w(x, w):
    m = np.isfinite(x)
    mu = np.sum(x[m] * w[m]) / np.sum(w[m])
    sd = np.sqrt(np.sum(w[m] * (x[m] - mu) ** 2) / np.sum(w[m]))
    return (x - mu) / sd


def wls_brr_pv(d, outcomes, predictors, standardize=True):
    """WLS of each outcome (list of PVs or a single variable) on predictors.

    Returns DataFrame with coef, SE (sampling via BRR + imputation via Rubin), t, p, n.
    Predictors are z-standardised with the final student weight inside the analytic sample,
    so coefficients are 'outcome units per 1 SD of predictor' (binary vars left as 0/1).
    """
    from scipy import stats
    cols = list(dict.fromkeys(outcomes + predictors + [W] + RW))
    s = d[cols].dropna(subset=outcomes + predictors).copy()
    X = s[predictors].to_numpy(float, copy=True)
    w0 = s[W].to_numpy(float)
    if standardize:
        for j, p in enumerate(predictors):
            if set(np.unique(X[:, j])) - {0.0, 1.0}:
                X[:, j] = zscore_w(X[:, j], w0)
    X = np.column_stack([np.ones(len(X)), X])
    Wm = s[[W] + RW].to_numpy(float)

    def fit(y, w):
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        return beta

    ests, samp_vars, r2s = [], [], []
    for pv in outcomes:
        y = s[pv].to_numpy(float)
        b0 = fit(y, Wm[:, 0])
        reps = np.array([fit(y, Wm[:, k]) for k in range(1, Wm.shape[1])])
        ests.append(b0)
        samp_vars.append(BRR_FACTOR * np.sum((reps - b0) ** 2, axis=0))
        yhat = X @ b0
        ybar = np.average(y, weights=w0)
        r2s.append(1 - np.average((y - yhat) ** 2, weights=w0) / np.average((y - ybar) ** 2, weights=w0))
    ests, samp_vars = np.array(ests), np.array(samp_vars)
    M = len(outcomes)
    coef = ests.mean(0)
    U = samp_vars.mean(0)
    B = ests.var(0, ddof=1) if M > 1 else 0
    se = np.sqrt(U + (1 + 1 / M) * B)
    t = coef / se
    p = 2 * stats.norm.sf(np.abs(t))
    return pd.DataFrame({"term": ["Intercept"] + predictors, "coef": coef, "se": se, "t": t, "p": p,
                         "n": len(s), "r2": np.mean(r2s)})


def pv_mean_se(d, pvs):
    """Mean of plausible values with BRR + imputation variance."""
    ests, vars_ = [], []
    for pv in pvs:
        e, se = weighted_mean_se(d, pv)
        ests.append(e); vars_.append(se ** 2)
    M = len(pvs)
    return np.mean(ests), np.sqrt(np.mean(vars_) + (1 + 1 / M) * np.var(ests, ddof=1))
