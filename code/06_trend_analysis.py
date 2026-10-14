"""A decade of change, PISA 2012-2022 (10 East/Southeast Asian systems; Viet Nam in all four cycles).

TR1  indicator trends by system x cycle (BRR SE; PVs + Rubin for maths)
TR2  changing relationships: digital access, belonging and anxiety vs maths, by cycle
TR3  interpretable ML by cycle for Viet Nam (LightGBM + SHAP on harmonised features)
Figures: FigT1 small multiples of trends, FigT2 2012->2022 dumbbells, FigT3 SHAP importance by cycle.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))
from pisa_utils import NAMES, weighted_mean_se, wls_brr_pv, pv_mean_se

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RES = OUT / "results_trend"
RES.mkdir(exist_ok=True)
SYSTEMS = ["VNM", "SGP", "JPN", "KOR", "TAP", "HKG", "MAC", "THA", "MYS", "IDN"]
CYCLES = [2012, 2015, 2018, 2022]
ACCENT, GREY, INK, MUTED = "#2a6fdb", "#b4bcc6", "#1f2328", "#59636e"

df = pd.read_parquet(OUT / "pisa_trend_2012_2022.parquet")


def pvs(d):
    return [f"PV{i}MATH" for i in range(1, 11) if d[f"PV{i}MATH"].notna().any()]


# ---------------- TR1: indicator trends ----------------
IND = ["COMPUTER", "INTERNET", "SMARTPHONE3", "BELONG_AGREE", "LONELY_AGREE", "BELONG_RAW",
       "ANX_RAW", "TENSE_AGREE", "DISC_RAW", "NOISE_MOST", "ESCS"]
rows = []
for (cnt, cyc), d in df.groupby(["CNT", "CYCLE"]):
    r = {"CNT": cnt, "CYCLE": cyc, "n": len(d)}
    has_pv = len(pvs(d)) > 0  # Viet Nam 2018: performance data not released in the public file
    if has_pv:
        m, se = pv_mean_se(d, pvs(d)); r["MATH"], r["MATH_se"] = m, se
    for v in IND:
        if d[v].notna().mean() > .3:
            m, se = weighted_mean_se(d, v)
            k = 100 if v in {"COMPUTER", "INTERNET", "SMARTPHONE3", "BELONG_AGREE", "LONELY_AGREE",
                             "TENSE_AGREE", "NOISE_MOST"} else 1
            r[v], r[v + "_se"] = m * k, se * k
    if has_pv:
        g = wls_brr_pv(d, pvs(d), ["ESCS"], standardize=False)
        r["ESCS_SLOPE"], r["ESCS_SLOPE_se"] = g.coef[1], g.se[1]
        g = wls_brr_pv(d, pvs(d), ["FEMALE"], standardize=False)
        r["GENDER_GAP_F_minus_M"], r["GENDER_GAP_se"] = g.coef[1], g.se[1]
    rows.append(r)
    print("TR1", cnt, cyc)
tr1 = pd.DataFrame(rows)
tr1.round(3).to_csv(RES / "TR1_indicator_trends.csv", index=False)

# change 2012 -> 2022 for questionnaire indicators (independent samples)
chg = []
for cnt, g in tr1.groupby("CNT"):
    a, b = g[g.CYCLE == 2012].iloc[0], g[g.CYCLE == 2022].iloc[0]
    for v in ["COMPUTER", "INTERNET", "BELONG_AGREE", "LONELY_AGREE", "BELONG_RAW", "ANX_RAW",
              "TENSE_AGREE", "DISC_RAW", "NOISE_MOST", "ESCS_SLOPE", "GENDER_GAP_F_minus_M"]:
        se_name = v + "_se" if v + "_se" in g else ("GENDER_GAP_se" if v.startswith("GENDER") else None)
        if pd.notna(a.get(v)) and pd.notna(b.get(v)):
            d_ = b[v] - a[v]; s_ = np.sqrt(a[se_name] ** 2 + b[se_name] ** 2)
            chg.append(dict(CNT=cnt, var=v, y2012=a[v], y2022=b[v], change=d_, se=s_, z=d_ / s_))
chg = pd.DataFrame(chg)
chg.round(3).to_csv(RES / "TR1b_change_2012_2022.csv", index=False)

# ---------------- TR2: changing relationships ----------------
rel = []
for (cnt, cyc), d in df.groupby(["CNT", "CYCLE"]):
    specs = {"R1_digital_access": ["COMPUTER", "INTERNET", "ESCS", "FEMALE", "BOOKS"],
             "R2_belonging": ["BELONG_RAW", "ESCS", "FEMALE"],
             "R3_anxiety": ["ANX_RAW", "ESCS", "FEMALE"]}
    for name, preds in specs.items():
        if not pvs(d) or any(d[p].notna().mean() < .3 for p in preds):
            continue
        r = wls_brr_pv(d, pvs(d), preds)
        r.insert(0, "model", name); r.insert(0, "CYCLE", cyc); r.insert(0, "CNT", cnt)
        rel.append(r)
rel = pd.concat(rel)
rel.to_csv(RES / "TR2_relationships_by_cycle.csv", index=False)
key = rel[rel.term.isin(["COMPUTER", "INTERNET", "BELONG_RAW", "ANX_RAW"])]
key.assign(cell=key.coef.round(1).astype(str) + " (" + key.se.round(1).astype(str) + ")") \
   .pivot_table(index=["term", "CNT"], columns="CYCLE", values="cell", aggfunc="first") \
   .to_csv(RES / "TR2_key_coefs_wide.csv")

# ---------------- TR3: ML by cycle (Viet Nam + all systems for comparison) ----------------
FEATS = {"ESCS": "Socio-economic status", "FEMALE": "Female", "GRADE": "Grade", "REPEAT": "Grade repetition",
         "BOOKS": "Books at home", "COMPUTER": "Computer at home", "INTERNET": "Internet at home",
         "SMARTPHONE3": "3+ smartphones at home", "BELONG_RAW": "Sense of belonging",
         "ANX_RAW": "Mathematics anxiety", "DISC_RAW": "Disciplinary climate (maths)"}
PARAMS = dict(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
ml_perf, ml_imp = [], []
for (cnt, cyc), d in df.groupby(["CNT", "CYCLE"]):
    d = d.reset_index(drop=True)
    if not pvs(d):
        continue
    feats = [f for f in FEATS if d[f].notna().mean() > .3]
    X, w, groups = d[feats], d["W_FSTUWT"].to_numpy(float), d["SCHID"].to_numpy()
    r2s, imps = [], []
    for k, pv in enumerate(pvs(d)[:5]):
        y = d[pv].to_numpy(float)
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X, y, groups):
            pred[te] = lgb.LGBMRegressor(**PARAMS, random_state=k).fit(X.iloc[tr], y[tr], sample_weight=w[tr]).predict(X.iloc[te])
        r2s.append(r2_score(y, pred, sample_weight=w))
        m = lgb.LGBMRegressor(**PARAMS, random_state=k).fit(X, y, sample_weight=w)
        sv = shap.TreeExplainer(m).shap_values(X)
        imps.append(np.average(np.abs(sv), axis=0, weights=w))
    ml_perf.append(dict(CNT=cnt, CYCLE=cyc, n=len(d), r2_cv=np.mean(r2s), n_features=len(feats)))
    imp = pd.DataFrame({"CNT": cnt, "CYCLE": cyc, "feature": feats, "mean_abs_shap": np.mean(imps, 0)})
    imp["share"] = imp.mean_abs_shap / imp.mean_abs_shap.sum()
    ml_imp.append(imp)
    print("TR3", cnt, cyc, round(np.mean(r2s), 3))
pd.DataFrame(ml_perf).to_csv(RES / "TR3_ml_performance_by_cycle.csv", index=False)
ml_imp = pd.concat(ml_imp)
ml_imp.to_csv(RES / "TR3_shap_by_cycle.csv", index=False)

# ---------------- Figures ----------------
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e8eb", "grid.linewidth": .6})
panels = [("MATH", "Mathematics (mean score)"), ("COMPUTER", "% computer for school work at home"),
          ("INTERNET", "% internet at home"), ("BELONG_AGREE", "% 'I feel like I belong at school'"),
          ("LONELY_AGREE", "% 'I feel lonely at school'"), ("ESCS_SLOPE", "Socio-economic gradient (points per ESCS unit)")]
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
for ax, (v, title) in zip(axes.flat, panels):
    for cnt in SYSTEMS:
        g = tr1[tr1.CNT == cnt].sort_values("CYCLE").dropna(subset=[v])
        if cnt == "VNM":
            continue
        ax.plot(g.CYCLE, g[v], color=GREY, lw=1, marker="o", ms=2.5)
    g = tr1[tr1.CNT == "VNM"].sort_values("CYCLE").dropna(subset=[v])
    ax.errorbar(g.CYCLE, g[v], yerr=1.96 * g[v + "_se"] if v + "_se" in g else None, color=ACCENT, lw=2,
                marker="o", ms=4, capsize=2, label="Viet Nam")
    ax.set_title(title, fontsize=9, color=INK); ax.set_xticks(CYCLES)
axes.flat[0].plot([], [], color=GREY, label="Other 9 systems"); axes.flat[0].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(RES / "FigT1_trends_small_multiples.png", dpi=300); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)
order = tr1[tr1.CYCLE == 2022].sort_values("ANX_RAW").CNT.tolist()
for ax, (v, title) in zip(axes, [("ANX_RAW", "Mathematics anxiety (1-4)"), ("TENSE_AGREE", "% tense doing maths homework"),
                                 ("NOISE_MOST", "% noise/disorder in most lessons")]):
    for i, cnt in enumerate(order):
        a = tr1[(tr1.CNT == cnt) & (tr1.CYCLE == 2012)][v].iloc[0]
        b = tr1[(tr1.CNT == cnt) & (tr1.CYCLE == 2022)][v].iloc[0]
        col = ACCENT if cnt == "VNM" else INK
        ax.plot([a, b], [i, i], color=col, lw=1.2)
        ax.plot(a, i, "o", mfc="white", mec=col, ms=5)
        ax.plot(b, i, "o", color=col, ms=5)
    ax.set_title(title, fontsize=9); ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(order)), [NAMES[c] for c in order])
axes[0].plot([], [], "o", mfc="white", mec=INK, label="2012"); axes[0].plot([], [], "o", color=INK, label="2022")
axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig(RES / "FigT2_dumbbell_2012_2022.png", dpi=300); plt.close(fig)

v = ml_imp[ml_imp.CNT == "VNM"].pivot(index="feature", columns="CYCLE", values="share").fillna(0)
v = v.loc[v.mean(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(6, 4.2))
im = ax.imshow(v.values, cmap="Blues", aspect="auto", vmin=0)
ax.set_xticks(range(v.shape[1]), v.columns); ax.set_yticks(range(v.shape[0]), [FEATS[f] for f in v.index])
for i in range(v.shape[0]):
    for j in range(v.shape[1]):
        val = v.values[i, j]
        ax.text(j, i, "–" if val == 0 else f"{100 * val:.0f}%", ha="center", va="center", fontsize=7.5,
                color="white" if val > .25 else INK)
ax.grid(False); ax.set_title("Viet Nam: share of SHAP importance by PISA cycle", fontsize=9)
fig.colorbar(im, ax=ax, fraction=.04)
fig.tight_layout(); fig.savefig(RES / "FigT3_vnm_shap_by_cycle.png", dpi=300); plt.close(fig)
print("done")
