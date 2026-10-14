"""Re-draw the multi-panel figures of Paper 1 at print size (text width ~6.6 in) from the saved result CSVs,
so that fonts stay legible in the PDF (>= 8 pt after scaling)."""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from pisa_utils import NAMES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TEX = (ROOT / 'latex' if (ROOT / 'latex/main.tex').exists() else ROOT / 'Paper1_PISA2022') / 'figures'
ACCENT, GREY, INK, MUTED = "#2a6fdb", "#b4bcc6", "#1f2328", "#59636e"
SYSTEMS = ["VNM", "SGP", "JPN", "KOR", "TAP", "HKG", "MAC", "THA", "MYS", "IDN"]
CYCLES = [2012, 2015, 2018, 2022]
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 10.5,
                     "ytick.labelsize": 10.5, "legend.fontsize": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e6e8eb", "grid.linewidth": .6})


def save(fig, name):
    panels = [a for a in fig.axes if a.get_label() != '<colorbar>']
    if len(panels) > 1:
        for i, ax in enumerate(panels):
            ax.set_title(f"({chr(97+i)}) {ax.get_title()}")
    fig.tight_layout()
    for d in [OUT / "results_trend", TEX]:
        fig.savefig(d / name, dpi=300)
        fig.savefig(d / Path(name).with_suffix('.pdf'))
    plt.close(fig)


tr1 = pd.read_csv(OUT / "results_trend/TR1_indicator_trends.csv")

# FigT1: 3 x 2 small multiples, print size
panels = [("MATH", "Mathematics (mean)"), ("COMPUTER", "Computer at home (%)"), ("INTERNET", "Internet at home (%)"),
          ("BELONG_AGREE", "'I belong at school' (%)"), ("LONELY_AGREE", "'I feel lonely at school' (%)"),
          ("ESCS_SLOPE", "SES gradient (points/ESCS)")]
fig, axes = plt.subplots(3, 2, figsize=(7.0, 8.2))
for ax, (v, title) in zip(axes.flat, panels):
    for cnt in SYSTEMS[1:]:
        g = tr1[tr1.CNT == cnt].sort_values("CYCLE").dropna(subset=[v])
        ax.plot(g.CYCLE, g[v], color=GREY, lw=1.1, marker="o", ms=3)
    g = tr1[tr1.CNT == "VNM"].sort_values("CYCLE").dropna(subset=[v])
    if v in ['MATH','ESCS_SLOPE']:
        # OECD explicitly disallows a Vietnamese 2022 performance trend comparison.
        label='performance trend' if v=='MATH' else 'gradient comparison'
        ax.text(.5,.52,'Viet Nam: no linked\n'+label,transform=ax.transAxes,ha='center',fontsize=10,color=ACCENT,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
    else:
        ax.errorbar(g.CYCLE, g[v], yerr=1.96 * g[v + "_se"] if v + "_se" in g else None, color=ACCENT, lw=2.4,
                    marker="o", ms=5, capsize=2.5, label="Viet Nam")
    ax.set_title(title, color=INK); ax.set_xticks(CYCLES)
axes.flat[0].plot([], [], color=GREY, label="Other 9 systems")
axes.flat[1].legend(frameon=False, loc="lower left")
save(fig, "FigT1_trends_small_multiples.png")

# FigT2: dumbbells, three panels stacked horizontally at print width
order = tr1[tr1.CYCLE == 2022].sort_values("ANX_RAW").CNT.tolist()
fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.9), sharey=True)
for ax, (v, title) in zip(axes, [("ANX_RAW", "Maths anxiety (1-4)"), ("TENSE_AGREE", "Tense doing\nmaths homework (%)"),
                                 ("NOISE_MOST", "Noise/disorder in\nmost lessons (%)")]):
    for i, cnt in enumerate(order):
        a = tr1[(tr1.CNT == cnt) & (tr1.CYCLE == 2012)][v].iloc[0]
        b = tr1[(tr1.CNT == cnt) & (tr1.CYCLE == 2022)][v].iloc[0]
        col = ACCENT if cnt == "VNM" else INK
        ax.plot([a, b], [i, i], color=col, lw=1.4)
        ax.plot(a, i, "o", mfc="white", mec=col, ms=6)
        ax.plot(b, i, "o", color=col, ms=6)
    ax.set_title(title); ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(order)), [NAMES[c].replace(" (China)", "") for c in order])
axes[0].plot([], [], "o", mfc="white", mec=INK, label="2012"); axes[0].plot([], [], "o", color=INK, label="2022")
axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)
save(fig, "FigT2_dumbbell_2012_2022.png")

# FigT3: SHAP share heatmap
FEATS = {"ESCS": "Socio-economic status", "FEMALE": "Female", "GRADE": "Grade", "REPEAT": "Grade repetition",
         "BOOKS": "Books at home", "COMPUTER": "Computer at home", "INTERNET": "Internet at home",
         "SMARTPHONE3": "3+ smartphones at home", "BELONG_RAW": "Sense of belonging",
         "ANX_RAW": "Mathematics anxiety", "DISC_RAW": "Disciplinary climate"}
imp = pd.read_csv(OUT / "results_trend/TR3_shap_by_cycle.csv")
v = imp[imp.CNT == "VNM"].pivot(index="feature", columns="CYCLE", values="share").fillna(0)
v = v.loc[v.mean(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(6.4, 4.8))
im = ax.imshow(v.values, cmap="Blues", aspect="auto", vmin=0)
ax.set_xticks(range(v.shape[1]), v.columns); ax.set_yticks(range(v.shape[0]), [FEATS[f] for f in v.index])
for i in range(v.shape[0]):
    for j in range(v.shape[1]):
        val = v.values[i, j]
        ax.text(j, i, "–" if val == 0 else f"{100 * val:.0f}%", ha="center", va="center", fontsize=9.5,
                color="white" if val > .25 else INK)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=.045); cb.set_label("Share of mean |SHAP|")
save(fig, "FigT3_vnm_shap_by_cycle.png")

# FigS1: system scatter
x = pd.read_csv(OUT / "results_system/S2_system_2022.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
for ax, (yv, lab) in zip(axes, [("DIST_SELF_MOST", "% distracted in most lessons"),
                                ("SCH_PHONE_BAN", "% in schools banning phones")]):
    s = x.dropna(subset=["IT.NET.USER.ZS", yv])
    for _, r in s.iterrows():
        col = ACCENT if r.CNT == "VNM" else INK
        ax.scatter(r["IT.NET.USER.ZS"], r[yv], color=col, s=30, zorder=3)
        ax.annotate(NAMES[r.CNT].split(" (")[0], (r["IT.NET.USER.ZS"], r[yv]), fontsize=8, xytext=(3, 3),
                    textcoords="offset points", color=col)
    ax.set_xlabel("Internet users, % of population (2022)"); ax.set_ylabel(lab)
save(fig, "FigS1_system_scatter.png")

panel = pd.read_csv(OUT / 'results_system/S1_system_panel.csv')
fig, ax = plt.subplots(figsize=(7, 4.4))
for cnt, g in panel.dropna(subset=['IT.NET.USER.ZS']).groupby('CNT'):
    if cnt not in NAMES: continue
    g = g[g.year.between(2010, 2023)].sort_values('year')
    ax.plot(g.year,g['IT.NET.USER.ZS'],color=ACCENT if cnt=='VNM' else GREY,lw=2 if cnt=='VNM' else 1)
    ax.text(g.year.iloc[-1]+.2,g['IT.NET.USER.ZS'].iloc[-1],NAMES[cnt].split(' (')[0],fontsize=7,
            color=ACCENT if cnt=='VNM' else MUTED,va='center')
for yr in CYCLES: ax.axvline(yr,color=MUTED,lw=.6,ls=':')
ax.set_ylabel('Internet users (% of population)'); ax.set_xlim(2010,2027)
ax.set_title('Digitalisation context, 2010–2023 (dotted lines = PISA cycles)')
save(fig,'FigS2_internet_trajectories.png')
print("replotted")
