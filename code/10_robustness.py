"""Robustness and mechanism analyses for Paper 1 (reviewer-proofing).

R0  administration mode (paper vs computer) by system x cycle
R1  item-level trends of the six belonging items (Viet Nam and region)
R2  measurement invariance (multi-group CFA, ML, weighted moments): belonging (VN, 4 cycles) and
    mathematics anxiety (VN, 2012 vs 2022); configural / metric / scalar / partial-scalar;
    latent mean change in 2012-SD units
R3  within-cycle student-level associations of home digital access with loneliness, belonging and anxiety
R4  DiNardo-Fortin-Lemieux reweighting decomposition of the 2012->2022 change in Viet Nam
    (step 1: composition = gender, grade, repetition, books; step 2: + computer, internet)
R5  inverse-probability-weighted (IPW) estimates of school phone bans (PISA 2022), pooled with system FE
R6  Benjamini-Hochberg q-values for the 13-system DIGDIST tests (M1/M2/A1/A2)
"""
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy import optimize, stats
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent))
from pisa_utils import REGION, W, RW, PV_MATH, BRR_FACTOR, recode, wls_brr_pv, weighted_mean_se

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RES = OUT / "results_robust"
RES.mkdir(exist_ok=True)
tr = pd.read_parquet(OUT / "pisa_trend_2012_2022.parquet")
B = [f"B{i}" for i in range(1, 7)]
A = [f"A{i}" for i in range(1, 6)]

# ---------------- R0 administration mode ----------------
r0 = tr.assign(CBA=(tr.ADMIN == 2).astype(float).where(tr.ADMIN.notna())).groupby(["CNT", "CYCLE"]).CBA.mean().unstack()
r0.round(2).to_csv(RES / "R0_admin_mode_share_computer.csv")
print(r0.round(2))

# ---------------- R1 item-level belonging trends ----------------
rows = []
for (cnt, cyc), d in tr.groupby(["CNT", "CYCLE"]):
    for b in B:
        d2 = d.assign(_agree=(d[b] >= 3).astype(float).where(d[b].notna()))
        m, se = weighted_mean_se(d2, "_agree")
        rows.append(dict(CNT=cnt, CYCLE=cyc, item=b, pct_belonging_response=100 * m, se=100 * se))
pd.DataFrame(rows).round(2).to_csv(RES / "R1_belonging_items.csv", index=False)


# ---------------- R2 multi-group CFA ----------------
def wmoments(X, w):
    w = w / w.sum()
    m = (X * w[:, None]).sum(0)
    Xc = X - m
    S = (Xc * w[:, None]).T @ Xc
    return m, S


def mgcfa(groups, level, free_tau=(), rcov=()):
    """One-factor MGCFA by ML on weighted moments. groups: list of (mean, cov, n).
    Identification: marker item 1 (lambda=1, tau=0); factor mean/variance free per group.
    level: 'configural' | 'metric' | 'scalar'. free_tau: item indices whose intercept is freed under scalar."""
    G, p = len(groups), len(groups[0][0])
    lam_shared = level in ("metric", "scalar")
    tau_shared = level == "scalar"
    # parameter layout
    idx, k = {}, 0
    def add(name, size):
        nonlocal k
        idx[name] = slice(k, k + size); k += size
    add("lam", p - 1) if lam_shared else [add(f"lam{g}", p - 1) for g in range(G)]
    if tau_shared:
        add("tau", p - 1)
        for g in range(G):
            add(f"taufree{g}", len(free_tau))
    else:
        [add(f"tau{g}", p - 1) for g in range(G)]
    for g in range(G):
        add(f"th{g}", p); add(f"psi{g}", 1); add(f"kap{g}", 1); add(f"rc{g}", len(rcov))
    x0 = np.zeros(k)
    for g, (m, S, n) in enumerate(groups):
        x0[idx[f"th{g}"]] = np.log(np.diag(S) * .5)
        x0[idx[f"psi{g}"]] = np.log(S[0, 0] * .5)
        x0[idx[f"kap{g}"]] = m[0]
    if lam_shared:
        x0[idx["lam"]] = 1
    else:
        for g in range(G):
            x0[idx[f"lam{g}"]] = 1
    N = sum(n for _, _, n in groups)

    def implied(x, g):
        lam = np.r_[1, x[idx["lam"] if lam_shared else idx[f"lam{g}"]]]
        if tau_shared:
            tau = np.r_[0, x[idx["tau"]]].copy()
            for j, it in enumerate(free_tau):
                tau[it] = x[idx[f"taufree{g}"]][j] if g > 0 else tau[it]
        else:
            tau = np.r_[0, x[idx[f"tau{g}"]]]
        th, psi, kap = np.exp(x[idx[f"th{g}"]]), np.exp(x[idx[f"psi{g}"]][0]), x[idx[f"kap{g}"]][0]
        Th = np.diag(th)
        for (i, j), r_ in zip(rcov, x[idx[f"rc{g}"]]):  # residual correlations (method effect)
            Th[i, j] = Th[j, i] = np.tanh(r_) * np.sqrt(th[i] * th[j])
        return tau + lam * kap, np.outer(lam, lam) * psi + Th, lam, tau, psi, kap

    def F(x):
        f = 0
        for g, (m, S, n) in enumerate(groups):
            mu, Sig, *_ = implied(x, g)
            try:
                L = np.linalg.cholesky(Sig)
            except np.linalg.LinAlgError:
                return 1e10
            Si = np.linalg.inv(Sig)
            dm = m - mu
            f += n / N * (2 * np.log(np.diag(L)).sum() + np.trace(S @ Si) - np.linalg.slogdet(S)[1] - p + dm @ Si @ dm)
        return f
    # under scalar with freed intercepts, groups>0 get their own intercept for the freed items
    r = optimize.minimize(F, x0, method="L-BFGS-B", options={"maxiter": 20000, "maxfun": 200000})
    chi2 = (N - 1) * r.fun
    df = G * p * (p + 3) // 2 - k + (len(free_tau) if tau_shared else 0)  # group-0 freed taus are unused
    est = [implied(r.x, g) for g in range(G)]
    return dict(chi2=chi2, df=df, N=N, est=est)


def null_chi2(groups):
    p = len(groups[0][0]); N = sum(n for *_, n in groups)
    f = sum(n / N * (np.log(np.prod(np.diag(S))) - np.linalg.slogdet(S)[1]) for m, S, n in groups)
    return (N - 1) * f, len(groups) * p * (p - 1) // 2


def fit_idx(res, null):
    c0, d0 = null
    G = len(res["est"])
    rmsea = np.sqrt(max(res["chi2"] - res["df"], 0) / (res["df"] * (res["N"] - 1))) * np.sqrt(G)
    cfi = 1 - max(res["chi2"] - res["df"], 0) / max(c0 - d0, res["chi2"] - res["df"], 1e-9)
    return round(cfi, 4), round(rmsea, 4)


inv_rows, lat_rows = [], []
NEG = ((0, 3), (0, 5), (3, 5))  # negatively worded belonging items: outsider, awkward, lonely
for scale, items, cycles, rc in [("Belonging", B, [2012, 2015, 2018, 2022], NEG),
                                 ("Belonging (excl. 2018)", B, [2012, 2015, 2022], NEG),
                                 ("Maths anxiety", A, [2012, 2022], ())]:
    v = tr[(tr.CNT == "VNM") & tr.CYCLE.isin(cycles)].dropna(subset=items)
    groups = []
    for c in cycles:
        g = v[v.CYCLE == c]
        m, S = wmoments(g[items].to_numpy(float), g[W].to_numpy(float))
        groups.append((m, S, len(g)))
    null = null_chi2(groups)
    loneliness_item = 5  # B6 "I feel lonely at school" (index in items)
    specs = [("configural", ()), ("metric", ()), ("scalar", ())]
    if scale.startswith("Belonging"):
        specs += [("scalar", (loneliness_item,)), ("scalar", (4,)), ("scalar", (4, 1))]  # B6; B5 'liked'; B5+B2
    for level, free in specs:
        res = mgcfa(groups, level, free, rc)
        cfi, rmsea = fit_idx(res, null)
        label = level if not free else "partial scalar (free intercepts: " + ", ".join(items[i] for i in free) + ")"
        inv_rows.append(dict(scale=scale, model=label, chi2=round(res["chi2"], 1), df=res["df"], CFI=cfi, RMSEA=rmsea))
        if level == "scalar":
            k0 = res["est"][0][5]; psi0 = res["est"][0][4]
            for c, e in zip(cycles, res["est"]):
                lat_rows.append(dict(scale=scale, model=label, CYCLE=c, latent_mean_2012SD=(e[5] - k0) / np.sqrt(psi0),
                                     latent_var_ratio=e[4] / psi0))
inv = pd.DataFrame(inv_rows); inv.to_csv(RES / "R2_invariance_fit.csv", index=False)
lat = pd.DataFrame(lat_rows).round(3); lat.to_csv(RES / "R2_latent_means.csv", index=False)
print(inv.to_string(index=False)); print(lat.to_string(index=False))

# ---------------- R3 within-cycle access associations ----------------
rows = []
for (cnt, cyc), d in tr.groupby(["CNT", "CYCLE"]):
    access = [a for a in ["COMPUTER", "INTERNET", "SMARTPHONE3"] if d[a].notna().mean() > .5]
    ctrl = ["ESCS", "FEMALE", "GRADE", "REPEAT", "BOOKS"]
    for y in ["LONELY_AGREE", "BELONG_RAW", "ANX_RAW"]:
        if d[y].notna().mean() < .3 or len(d.dropna(subset=[y] + access + ctrl)) < 300:
            continue
        dd = d.copy()
        if y == "LONELY_AGREE":
            dd[y] = dd[y] * 100  # percentage points
        r = wls_brr_pv(dd, [y], access + ctrl)
        r.insert(0, "outcome", y); r.insert(0, "CYCLE", cyc); r.insert(0, "CNT", cnt)
        rows.append(r[r.term.isin(access)])
r3 = pd.concat(rows); r3.to_csv(RES / "R3_access_wellbeing.csv", index=False)
print(r3[r3.CNT == "VNM"].round(3).to_string(index=False))


# ---------------- R4 DFL reweighting decomposition (Viet Nam 2012 -> 2022) ----------------
def dfl(d12, d22, X, y, wcol):
    """Counterfactual 2022 mean of y with the 2012 distribution of X (weights wcol)."""
    pool = pd.concat([d12.assign(_t=0), d22.assign(_t=1)])
    Xd = pd.get_dummies(pool[X].astype("category"), drop_first=True).astype(float) if X else None
    lr = LogisticRegression(max_iter=2000, C=1e4).fit(Xd, pool._t, sample_weight=pool[wcol])
    p = lr.predict_proba(Xd[pool._t.to_numpy() == 1])[:, 1]
    s22, s12 = d22[wcol].sum(), d12[wcol].sum()
    psi = (1 - p) / p * (s22 / s12)
    w = d22[wcol].to_numpy() * psi
    ok = d22[y].notna().to_numpy()
    return np.sum(w[ok] * d22[y].to_numpy()[ok]) / np.sum(w[ok])


v = tr[tr.CNT == "VNM"].copy()
v["BOOKS"] = v.BOOKS.fillna(-1); v["GRADE"] = v.GRADE.clip(-2, 1).fillna(-9)
v["REPEAT"] = v.REPEAT.fillna(-1); v["FEMALE"] = v.FEMALE.fillna(-1)
v["COMPUTER"] = v.COMPUTER.fillna(-1); v["INTERNET"] = v.INTERNET.fillna(-1)
d12, d22 = v[v.CYCLE == 2012], v[v.CYCLE == 2022]
X1 = ["FEMALE", "GRADE", "REPEAT", "BOOKS"]
X2 = X1 + ["COMPUTER", "INTERNET"]
rows = []
for y in ["LONELY_AGREE", "BELONG_RAW", "ANX_RAW", "TENSE_AGREE", "NOISE_MOST"]:
    sc = 100 if y in ("LONELY_AGREE", "TENSE_AGREE", "NOISE_MOST") else 1
    def est(wcol):
        a12 = np.average(d12[y].dropna(), weights=d12.loc[d12[y].notna(), wcol])
        a22 = np.average(d22[y].dropna(), weights=d22.loc[d22[y].notna(), wcol])
        return np.array([a12, a22, dfl(d12, d22, X1, y, wcol), dfl(d12, d22, X2, y, wcol)]) * sc
    base = est(W)
    reps = np.array([est(rw) for rw in RW[:20]])  # 20 replicates (Fay) for speed; SE approximate
    se = np.sqrt(BRR_FACTOR * 4 * np.sum((reps - base) ** 2, axis=0))  # rescale 80->20 replicates
    total = base[1] - base[0]
    rows.append(dict(outcome=y, y2012=base[0], y2022=base[1], change=total, cf_composition=base[2],
                     cf_composition_access=base[3], explained_composition=base[1] - base[2],
                     explained_access=base[2] - base[3], unexplained=base[3] - base[0],
                     share_explained_total=(base[1] - base[3]) / total if total else np.nan,
                     se_change=np.sqrt(se[0] ** 2 + se[1] ** 2)))
r4 = pd.DataFrame(rows).round(3); r4.to_csv(RES / "R4_dfl_decomposition_vnm.csv", index=False)
print(r4.to_string(index=False))

# ---------------- R5 IPW school phone bans ----------------
sav = glob.glob(str(ROOT / "data/raw/pisa2022/sch/*.SAV"))[0]
sch, _ = pyreadstat.read_sav(sav, usecols=["CNT", "CNTSCHID", "SC190Q02JA", "SC001Q01TA", "SCHSIZE", "SC013Q01TA"])
sch["BAN"] = (sch.SC190Q02JA == 1).astype(float).where(sch.SC190Q02JA.isin([1, 2]))
sch["URBAN"] = (sch.SC001Q01TA >= 4).astype(float).where(sch.SC001Q01TA.between(1, 6))
sch["PRIVATE"] = (sch.SC013Q01TA == 2).astype(float).where(sch.SC013Q01TA.isin([1, 2]))
stu = recode(pd.read_parquet(OUT / "pisa22_analysis.parquet"))
stu = stu[stu.CNT.isin(REGION)]
stu["SCH_ESCS"] = stu.groupby("CNTSCHID").ESCS.transform("mean")
sch = sch.merge(stu.groupby("CNTSCHID").SCH_ESCS.first().reset_index(), on="CNTSCHID", how="inner")
sch = sch[sch.CNT.isin(REGION)].dropna(subset=["BAN"])
ipw = []
for cnt, g in sch.groupby("CNT"):
    Xs = g[["SCH_ESCS", "URBAN", "SCHSIZE", "PRIVATE"]].copy()
    Xs["SCHSIZE"] = np.log1p(Xs.SCHSIZE)
    Xs = Xs.fillna(Xs.median()).fillna(0)
    if g.BAN.nunique() < 2:
        continue
    ps = LogisticRegression(max_iter=2000).fit(Xs, g.BAN).predict_proba(Xs)[:, 1].clip(.05, .95)
    ipw.append(pd.DataFrame({"CNTSCHID": g.CNTSCHID, "BAN": g.BAN,
                             "IPW": np.where(g.BAN == 1, g.BAN.mean() / ps, (1 - g.BAN.mean()) / (1 - ps))}))
ipw = pd.concat(ipw)
d = stu.merge(ipw, on="CNTSCHID", how="inner")
scale = 5000 / d.groupby("CNT")[W].transform("sum")
for c in [W] + RW:
    d[c] = d[c] * scale * d.IPW
for cnt in REGION[1:]:
    d[f"FE_{cnt}"] = (d.CNT == cnt).astype(float)
rows = []
for y, ys in [("DIGDIST", ["DIGDIST"]), ("ANXMAT", ["ANXMAT"]), ("MATH", PV_MATH)]:
    dd = d.dropna(subset=ys[:1])
    fe = [f for f in [f"FE_{c}" for c in REGION[1:]] if dd[f].sum() > 0]
    for spec, X in [("unadjusted", ["BAN"]), ("IPW + controls", ["BAN", "ESCS", "FEMALE", "GRADE"])]:
        r = wls_brr_pv(dd, ys, X + fe, standardize=(y != "MATH"))
        b = r[r.term == "BAN"].iloc[0]
        rows.append(dict(outcome=y, spec=spec, coef=b.coef, se=b.se, p=b.p, n=b.n))
r5 = pd.DataFrame(rows).round(4); r5.to_csv(RES / "R5_ipw_phone_ban.csv", index=False)
print(r5.to_string(index=False))

# ---------------- R6 Benjamini-Hochberg ----------------
t3 = pd.read_csv(OUT / "results/T3_math_regressions.csv"); t4 = pd.read_csv(OUT / "results/T4_anxiety_regressions.csv")
rows = []
for tab, models in [(t3, ["M1_controls", "M2_+climate"]), (t4, ["A1_controls", "A2_+climate_relations"])]:
    for m in models:
        s = tab[(tab.model == m) & (tab.term == "DIGDIST")].copy()
        s["q_BH"] = multipletests(s.p, method="fdr_bh")[1]
        rows.append(s[["CNT", "model", "coef", "se", "p", "q_BH"]])
r6 = pd.concat(rows); r6.round(4).to_csv(RES / "R6_bh_qvalues.csv", index=False)
print(r6[r6.CNT == "VNM"].round(4).to_string(index=False))
print("sig after BH:", r6.groupby("model").apply(lambda x: int((x.q_BH < .05).sum())).to_dict())
