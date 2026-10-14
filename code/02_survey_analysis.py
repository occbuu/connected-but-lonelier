"""Design-based analysis (final + 80 BRR replicate weights, 10 PVs).

T1  prevalence of digital distraction / digital-dependence indicators by system
T2  mean mathematics, ANXMAT, BELONG by system
T3  WLS: mathematics on digital indicators + controls (+ socio-emotional block), by system
T4  WLS: mathematics anxiety (ANXMAT) on digital indicators + controls, by system
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pisa_utils import REGION, NAMES, PV_MATH, recode, weighted_mean_se, pv_mean_se, wls_brr_pv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RES = OUT / "results"
RES.mkdir(exist_ok=True)

df = recode(pd.read_parquet(OUT / "pisa22_analysis.parquet"))
df = df[df.CNT.isin(REGION)]

# ---------- coverage check: which indicators were administered in each system ----------
check = ["DIST_SELF", "DIST_PEERS", "ONLINE_PRESSURE", "DEVICE_ANXIETY", "NOTIF_OFF_CLASS",
         "EASILY_DISTRACTED", "ESCS", "BELONG", "BULLIED", "ANXMAT", "MATHEFF", "DISCLIM",
         "TEACHSUP", "STRESAGR", "EMOCOAGR", "PERSEVAGR", "ICTWKDY", "ICTDISTR", "ICTREG", "LIFESAT"]
cov = df.groupby("CNT")[[c for c in check if c in df]].apply(lambda g: g.notna().mean()).round(2)
cov.to_csv(RES / "T0_coverage_nonmissing.csv")
print(cov.T.to_string())

# ---------- T1/T2 descriptives ----------
rows = []
for cnt in REGION:
    d = df[df.CNT == cnt]
    if d.empty:
        continue
    r = {"CNT": cnt, "system": NAMES[cnt], "n": len(d)}
    for v in ["DIST_SELF_MOST", "DIST_PEERS_MOST", "ONLINE_PRESSURE_HIGH", "DEVICE_ANXIETY_HIGH", "OWN_SMARTPHONE"]:
        if d[v].notna().mean() > .5:
            m, se = weighted_mean_se(d, v)
            r[v] = 100 * m; r[v + "_se"] = 100 * se
    m, se = pv_mean_se(d, PV_MATH); r["MATH"] = m; r["MATH_se"] = se
    for v in ["ANXMAT", "BELONG", "BULLIED", "ESCS"]:
        if d[v].notna().mean() > .5:
            m, se = weighted_mean_se(d, v); r[v] = m; r[v + "_se"] = se
    rows.append(r)
t1 = pd.DataFrame(rows)
t1.round(2).to_csv(RES / "T1_descriptives_by_system.csv", index=False)
print(t1.round(1).to_string())

# ---------- T3/T4 regressions ----------
# DIGDIST = digital distraction in maths lessons (mean of ST273Q06/Q07, available in all 13 systems).
# Digital-habit items (ST322) were only in the computer-based questionnaire -> not for VNM/KHM.
HABITS = ["ONLINE_PRESSURE", "DEVICE_ANXIETY", "NOTIF_OFF_CLASS"]
CONTROLS = ["ESCS", "FEMALE", "GRADE", "REPEAT"]
CLIMATE = ["DISCLIM_ND"]
SEL = ["BELONG", "BULLIED", "TEACHSUP", "MATHEFF", "ANXMAT"]


def usable(d, vars_):
    return [v for v in vars_ if d[v].notna().mean() > .5 and d[v].nunique() > 1]


specs_math = [("M1_controls", ["DIGDIST"] + CONTROLS),
              ("M2_+climate", ["DIGDIST"] + CONTROLS + CLIMATE),
              ("M3_+socioemotional", ["DIGDIST"] + CONTROLS + CLIMATE + SEL),
              ("M4_habits", ["DIGDIST"] + HABITS + CONTROLS + CLIMATE)]
specs_anx = [("A1_controls", ["DIGDIST"] + CONTROLS),
             ("A2_+climate_relations", ["DIGDIST"] + CONTROLS + CLIMATE + ["BELONG", "BULLIED", "TEACHSUP"]),
             ("A3_habits", ["DIGDIST"] + HABITS + CONTROLS + CLIMATE)]

out3, out4 = [], []
for cnt in REGION:
    d = df[df.CNT == cnt]
    if d.empty:
        continue
    for name, preds in specs_math:
        if name == "M4_habits" and d["ONLINE_PRESSURE"].notna().mean() < .3:
            continue
        preds = [v for v in preds if d[v].notna().mean() > .3 and d[v].nunique() > 1]
        r = wls_brr_pv(d, PV_MATH, preds)
        r.insert(0, "model", name); r.insert(0, "CNT", cnt)
        out3.append(r)
    if d["ANXMAT"].notna().mean() > .5:
        for name, preds in specs_anx:
            if name == "A3_habits" and d["ONLINE_PRESSURE"].notna().mean() < .3:
                continue
            preds = [v for v in preds if d[v].notna().mean() > .3 and d[v].nunique() > 1]
            r = wls_brr_pv(d, ["ANXMAT"], preds)
            r.insert(0, "model", name); r.insert(0, "CNT", cnt)
            out4.append(r)
    print("done", cnt)

t3 = pd.concat(out3); t3.to_csv(RES / "T3_math_regressions.csv", index=False)
t4 = pd.concat(out4); t4.to_csv(RES / "T4_anxiety_regressions.csv", index=False)


def star(x):
    return x.coef.map("{:.2f}".format) + np.select([x.p < .001, x.p < .01, x.p < .05], ["***", "**", "*"], "") + " (" + x.se.map("{:.2f}".format) + ")"


for tab, name, keep in [(t3, "T3", ["DIGDIST"] + HABITS + ["DISCLIM_ND"]), (t4, "T4", ["DIGDIST"] + HABITS + ["DISCLIM_ND"])]:
    sub = tab[tab.term.isin(keep)].assign(cell=star)
    wide = sub.pivot_table(index=["model", "term"], columns="CNT", values="cell", aggfunc="first")
    ns = tab.groupby(["model", "CNT"]).n.first().unstack()
    wide.to_csv(RES / f"{name}_key_coefs_wide.csv")
    ns.to_csv(RES / f"{name}_n.csv")
    print(name); print(wide.to_string())
