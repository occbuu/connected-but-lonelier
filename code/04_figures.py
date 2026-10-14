"""Publication figures for Paper 1 (static PNG, 300 dpi). Viet Nam highlighted, other systems grey."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from pisa_utils import NAMES

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs/results"
ACCENT, GREY, INK, MUTED = "#2a6fdb", "#9aa3ad", "#1f2328", "#59636e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e8eb", "grid.linewidth": .6})

def save(fig, name):
    fig.tight_layout()
    for directory in [RES, ROOT / 'latex/figures']:
        fig.savefig(directory / name, dpi=300)
        fig.savefig(directory / Path(name).with_suffix('.pdf'))
    plt.close(fig)

# Figure 1: prevalence of frequent digital distraction
t1 = pd.read_csv(RES / "T1_descriptives_by_system.csv").sort_values("DIST_SELF_MOST")
fig, ax = plt.subplots(figsize=(6.2, 4.2))
cols = [ACCENT if c == "VNM" else GREY for c in t1.CNT]
ax.barh(t1.system, t1.DIST_SELF_MOST, xerr=1.96 * t1.DIST_SELF_MOST_se, color=cols, height=.62,
        error_kw=dict(ecolor=INK, lw=.8, capsize=2))
for y, (v, se, c) in enumerate(zip(t1.DIST_SELF_MOST, t1.DIST_SELF_MOST_se, t1.CNT)):
    ax.text(v + 1.96 * se + 1, y, f"{v:.1f}%", va="center", fontsize=8, color=INK, fontweight="bold" if c == "VNM" else None)
ax.set_xlabel("% distracted by digital devices in most/every maths lesson (95% CI)")
ax.set_xlim(0, 50); ax.grid(axis="y", visible=False)
save(fig, "Fig1_prevalence_digital_distraction.png")


def forest(tab, models, labels, xlabel, fname, legend_loc="upper left"):
    sub = tab[(tab.term == "DIGDIST") & tab.model.isin(models)]
    order = sub[sub.model == models[0]].sort_values("coef").CNT.tolist()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for k, (m, lab) in enumerate(zip(models, labels)):
        s = sub[sub.model == m].set_index("CNT").reindex(order)
        y = np.arange(len(order)) + (k - .5) * .28
        for yi, cnt, c, se in zip(y, order, s.coef, s.se):
            col = ACCENT if cnt == "VNM" else INK
            ax.errorbar(c, yi, xerr=1.96 * se, fmt="o", ms=5, color=col, lw=1,
                        mfc=col if k == 0 else "white", mec=col, label=lab if cnt == order[0] else None)
    ax.axvline(0, color=MUTED, lw=.8)
    ax.set_yticks(range(len(order)), [NAMES[c] for c in order])
    ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc=legend_loc)
    save(fig, fname)


t3 = pd.read_csv(RES / "T3_math_regressions.csv")
forest(t3, ["M1_controls", "M2_+climate"],
       ["Controls only (M1)", "+ non-digital disciplinary climate (M2)"],
       "Change in maths score per 1 SD of digital distraction (95% CI)", "Fig2_forest_math.png")
t4 = pd.read_csv(RES / "T4_anxiety_regressions.csv")
forest(t4, ["A1_controls", "A2_+climate_relations"],
       ["Controls only (A1)", "+ climate & relationships (A2)"],
       "Change in maths anxiety (index) per 1 SD of digital distraction (95% CI)", "Fig3_forest_anxiety.png", legend_loc="lower right")

# Figure 4: digital self-regulation (turning off notifications) vs maths, M4, systems with CBA module
s = t3[(t3.model == "M4_habits") & t3.term.isin(["NOTIF_OFF_CLASS", "ONLINE_PRESSURE", "DEVICE_ANXIETY", "DIGDIST"])]
lab = {"NOTIF_OFF_CLASS": "Turns off notifications in class", "ONLINE_PRESSURE": "Feels pressure to be online in class",
       "DEVICE_ANXIETY": "Nervous without device", "DIGDIST": "Digital distraction (classroom)"}
fig, ax = plt.subplots(figsize=(6.2, 3.4))
for i, (term, g) in enumerate(s.groupby("term", sort=False)):
    y = np.full(len(g), i) + np.linspace(-.25, .25, len(g))
    ax.errorbar(g.coef, y, xerr=1.96 * g.se, fmt="o", ms=3.5, color=INK, lw=.7, alpha=.85)
    ax.plot(np.average(g.coef, weights=1 / g.se ** 2), i, "D", color=ACCENT, ms=7, zorder=5,
            label="Inverse-variance mean" if i == 0 else None)
ax.set_yticks(range(s.term.nunique()), [lab[t] for t in s.drop_duplicates("term").term])
ax.axvline(0, color=MUTED, lw=.8); ax.grid(axis="y", visible=False)
ax.set_xlabel("Maths score change per 1 SD (95% CI), one dot per system")
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
save(fig, "Fig4_digital_habits_math.png")
print("figures saved")
