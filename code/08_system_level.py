"""System-level lens: national digitalisation (World Bank WDI), population mental-health context
(WHO GHO suicide mortality), national phone policy (UNESCO GEM PEER coding) and PISA aggregates.

Output: results_system/S1_system_panel.csv (system x year), S2_system_2022.csv (cross-section),
        FigS1 (internet use vs digital distraction / anxiety), FigS2 (internet use 2012-2022 trajectories).
With 10-13 systems this is a descriptive, hypothesis-generating layer (Spearman rho reported).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from pisa_utils import NAMES

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "outputs"
RES = OUT / "results_system"
RES.mkdir(exist_ok=True)
ACCENT, INK, MUTED, GREY = "#2a6fdb", "#1f2328", "#59636e", "#b4bcc6"

# World Bank WDI
wb = []
for f in (RAW / "worldbank").glob("*.json"):
    j = json.loads(f.read_text(encoding="utf-8"))
    if len(j) < 2 or not j[1]:
        continue
    for r in j[1]:
        wb.append(dict(CNT=r["countryiso3code"], year=int(r["date"]), indicator=r["indicator"]["id"], value=r["value"]))
wb = pd.DataFrame(wb).pivot_table(index=["CNT", "year"], columns="indicator", values="value").reset_index()

# WHO GHO suicide mortality (both sexes, all ages, per 100,000)
who = pd.DataFrame(json.loads((RAW / "who/SDGSUICIDE.json").read_text(encoding="utf-8"))["value"])
who = who[(who.Dim1 == "SEX_BTSX") & (who.Dim2 == "AGEGROUP_YEARSALL")][["SpatialDim", "TimeDim", "NumericValue"]]
who.columns = ["CNT", "year", "suicide_rate"]
panel = wb.merge(who, on=["CNT", "year"], how="outer").sort_values(["CNT", "year"])
panel.to_csv(RES / "S1_system_panel.csv", index=False)

# PEER policy coding
pol = pd.read_csv(RAW / "policy/peer_phone_policy_coding.csv")[["CNT", "policy_2022", "policy_code_2022", "later_change"]]

# PISA 2022 aggregates (from Paper 1 results) + 2012-2022 changes
t1 = pd.read_csv(OUT / "results/T1_descriptives_by_system.csv")[["CNT", "DIST_SELF_MOST", "ANXMAT", "MATH", "BELONG"]]
sch = pd.read_csv(OUT / "results_policy/P0_policy_prevalence.csv")[["CNT", "SCH_PHONE_BAN", "SCH_RESP_PROG"]]
w22 = panel[panel.year == 2022].drop(columns="year")
w21 = panel[panel.year == 2021][["CNT", "suicide_rate"]].rename(columns={"suicide_rate": "suicide_rate_2021"})
x = t1.merge(sch, on="CNT").merge(pol, on="CNT", how="left").merge(w22.drop(columns=["suicide_rate"], errors="ignore"), on="CNT", how="left").merge(w21, on="CNT", how="left")
trend = OUT / "results_trend/TR1b_change_2012_2022.csv"
if trend.exists():
    ch = pd.read_csv(trend).pivot(index="CNT", columns="var", values="change").add_prefix("chg_").reset_index()
    x = x.merge(ch, on="CNT", how="left")
x.round(3).to_csv(RES / "S2_system_2022.csv", index=False)

pairs = [("IT.NET.USER.ZS", "DIST_SELF_MOST"), ("IT.NET.USER.ZS", "ANXMAT"), ("SCH_PHONE_BAN", "DIST_SELF_MOST"),
         ("NY.GDP.PCAP.PP.KD", "DIST_SELF_MOST"), ("suicide_rate_2021", "ANXMAT"), ("DIST_SELF_MOST", "ANXMAT")]
cor = []
for a, b in pairs:
    s = x[[a, b]].dropna()
    rho, p = spearmanr(s[a], s[b])
    cor.append(dict(x=a, y=b, n=len(s), spearman_rho=rho, p=p))
pd.DataFrame(cor).round(3).to_csv(RES / "S3_system_correlations.csv", index=False)
print(pd.DataFrame(cor).round(3))

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e8eb"})
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
for ax, (yv, lab) in zip(axes, [("DIST_SELF_MOST", "% distracted in most/every maths lesson (PISA 2022)"),
                                ("SCH_PHONE_BAN", "% students in schools banning phones (PISA 2022)")]):
    s = x.dropna(subset=["IT.NET.USER.ZS", yv])
    for _, r in s.iterrows():
        col = ACCENT if r.CNT == "VNM" else INK
        ax.scatter(r["IT.NET.USER.ZS"], r[yv], color=col, s=28, zorder=3)
        ax.annotate(NAMES[r.CNT].split(" (")[0], (r["IT.NET.USER.ZS"], r[yv]), fontsize=7, xytext=(3, 3),
                    textcoords="offset points", color=col)
    ax.set_xlabel("Internet users, % of population, 2022 (World Bank WDI)"); ax.set_ylabel(lab, fontsize=8)
fig.tight_layout(); fig.savefig(RES / "FigS1_system_scatter.png", dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.8))
for cnt, g in panel.dropna(subset=["IT.NET.USER.ZS"]).groupby("CNT"):
    if cnt not in NAMES:
        continue
    g = g[g.year.between(2010, 2023)]
    ax.plot(g.year, g["IT.NET.USER.ZS"], color=ACCENT if cnt == "VNM" else GREY, lw=2 if cnt == "VNM" else 1)
    ax.text(g.year.iloc[-1] + .2, g["IT.NET.USER.ZS"].iloc[-1], NAMES[cnt].split(" (")[0], fontsize=6.5,
            color=ACCENT if cnt == "VNM" else MUTED, va="center")
for yr in [2012, 2015, 2018, 2022]:
    ax.axvline(yr, color=MUTED, lw=.6, ls=":")
ax.set_ylabel("Internet users (% of population)"); ax.set_xlim(2010, 2025.5)
ax.set_title("Digitalisation context, 2010-2023 (dotted lines = PISA cycles)", fontsize=9)
fig.tight_layout(); fig.savefig(RES / "FigS2_internet_trajectories.png", dpi=300); plt.close(fig)
print(x[["CNT", "IT.NET.USER.ZS", "DIST_SELF_MOST", "ANXMAT", "SCH_PHONE_BAN", "policy_2022", "suicide_rate_2021"]].round(1).to_string(index=False))
