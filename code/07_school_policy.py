"""School-level digital-device policies (PISA 2022 school questionnaire) and student outcomes.

School variables (principal reports, SC190/SC189):
  SCH_PHONE_BAN   "The use of cell phones is not allowed on the school premises"
  SCH_CO_RULES    "Teachers establish rules in collaboration with students about their use of digital resources"
  SCH_RESP_PROG   "The school has a specific programme to prepare students for responsible internet behaviour"
  SCH_SEL_POLICY  "The school has a specific policy to encourage teaching and learning of social and emotional skills"
  SCH_ESCS        school mean of student ESCS (compositional control)
  URBAN           school located in a city of 100,000+ (SC001)

Models (design-based: final + 80 BRR weights; 10 PVs for maths), per system and pooled with system
fixed effects and senate-type weights (each system sums to the same total):
  P1 DIGDIST ~ policies + controls
  P2 ANXMAT  ~ policies + DIGDIST + controls
  P3 MATH    ~ policies + DIGDIST + controls
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

sys.path.insert(0, str(Path(__file__).parent))
from pisa_utils import REGION, NAMES, PV_MATH, W, RW, recode, wls_brr_pv, weighted_mean_se

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RES = OUT / "results_policy"
RES.mkdir(exist_ok=True)

sav = glob.glob(str(ROOT / "data/raw/pisa2022/sch/*.SAV"))[0]
cols = ["CNT", "CNTSCHID", "SC190Q01JA", "SC190Q02JA", "SC190Q06JA", "SC190Q07JA", "SC190Q08JA",
        "SC189Q01JA", "SC001Q01TA", "DIGDVPOL"]
sch, _ = pyreadstat.read_sav(sav, usecols=cols)
yes = lambda s: (s == 1).astype(float).where(s.isin([1, 2]))
sch = pd.DataFrame({"CNTSCHID": sch.CNTSCHID, "SCH_PHONE_BAN": yes(sch.SC190Q02JA),
                    "SCH_WRITTEN_STATEMENT": yes(sch.SC190Q01JA), "SCH_TEACHER_RULES": yes(sch.SC190Q06JA),
                    "SCH_CO_RULES": yes(sch.SC190Q07JA), "SCH_RESP_PROG": yes(sch.SC190Q08JA),
                    "SCH_SEL_POLICY": yes(sch.SC189Q01JA),
                    "URBAN": (sch.SC001Q01TA >= 4).astype(float).where(sch.SC001Q01TA.between(1, 6))})

stu = recode(pd.read_parquet(OUT / "pisa22_analysis.parquet"))
stu = stu[stu.CNT.isin(REGION)].merge(sch, on="CNTSCHID", how="left")
stu["SCH_ESCS"] = stu.groupby("CNTSCHID").ESCS.transform("mean")

POL = ["SCH_PHONE_BAN", "SCH_CO_RULES", "SCH_RESP_PROG", "SCH_SEL_POLICY"]
CTRL = ["ESCS", "SCH_ESCS", "FEMALE", "GRADE", "URBAN"]

# Descriptives: % of students in schools with each policy
desc = []
for cnt in REGION:
    d = stu[stu.CNT == cnt]
    r = {"CNT": cnt, "system": NAMES[cnt]}
    for v in POL + ["SCH_TEACHER_RULES", "SCH_WRITTEN_STATEMENT"]:
        m, se = weighted_mean_se(d, v); r[v] = 100 * m; r[v + "_se"] = 100 * se
    desc.append(r)
pd.DataFrame(desc).round(2).to_csv(RES / "P0_policy_prevalence.csv", index=False)


def fit_all(d, tag):
    out = []
    specs = [("P1_digdist", ["DIGDIST"], POL + CTRL),
             ("P2_anxiety", ["ANXMAT"], POL + ["DIGDIST"] + CTRL),
             ("P3_math", PV_MATH, POL + ["DIGDIST"] + CTRL)]
    for name, y, X in specs:
        if d[y[0]].notna().mean() < .3:
            continue
        X = [x for x in X if d[x].notna().mean() > .5 and d[x].nunique() > 1]
        r = wls_brr_pv(d, y, X)
        r.insert(0, "model", name); r.insert(0, "CNT", tag)
        out.append(r)
    return out


res = []
for cnt in REGION:
    res += fit_all(stu[stu.CNT == cnt], cnt)
    print("done", cnt)

# Pooled: system fixed effects, each system weighted to the same total (senate weights)
pool = stu.copy()
scale = 5000 / pool.groupby("CNT")[W].transform("sum")
for c in [W] + RW:
    pool[c] = pool[c] * scale
for cnt in REGION[1:]:
    pool[f"FE_{cnt}"] = (pool.CNT == cnt).astype(float)
FE = [f"FE_{c}" for c in REGION[1:]]


def fit_pooled(d):
    out = []
    for name, y, X in [("P1_digdist", ["DIGDIST"], POL + CTRL),
                       ("P3_math", PV_MATH, POL + ["DIGDIST"] + CTRL)]:
        r = wls_brr_pv(d, y, X + FE)
        r.insert(0, "model", name); r.insert(0, "CNT", "POOLED_13")
        out.append(r)
    a = d[d.ANXMAT.notna()]
    fe = [f for f in FE if a[f].sum() > 0]
    r = wls_brr_pv(a, ["ANXMAT"], POL + ["DIGDIST"] + CTRL + fe)
    r.insert(0, "model", "P2_anxiety"); r.insert(0, "CNT", "POOLED_ANX")
    out.append(r)
    return out


res += fit_pooled(pool)
res = pd.concat(res)
res = res[~res.term.str.startswith("FE_")]
res.to_csv(RES / "P1_P3_policy_models.csv", index=False)
k = res[res.term.isin(POL)]
star = np.select([k.p < .001, k.p < .01, k.p < .05], ["***", "**", "*"], "")
k = k.assign(cell=k.coef.map("{:.2f}".format) + star + " (" + k.se.map("{:.2f}".format) + ")")
k.pivot_table(index=["model", "term"], columns="CNT", values="cell", aggfunc="first").to_csv(RES / "P_key_policy_coefs.csv")
print(k[k.CNT.isin(["VNM", "POOLED_13", "POOLED_ANX"])][["CNT", "model", "term", "cell", "n"]].to_string(index=False))
