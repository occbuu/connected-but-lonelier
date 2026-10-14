"""Write LaTeX tables (complete tabular environments) and copy figures for Paper 1."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pisa_utils import NAMES, REGION

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TEX = ROOT / 'latex' if (ROOT / 'latex/main.tex').exists() else ROOT / 'Paper1_PISA2022'
(TEX / "tables").mkdir(parents=True, exist_ok=True)
(TEX / "figures").mkdir(parents=True, exist_ok=True)
NL = r" \\"


def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def zstar(z):
    return star(2 * (1 - __import__("scipy.stats").stats.norm.cdf(abs(z))))


def w(name, spec, head, rows):
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule", *head, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (TEX / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


f = lambda x, d=1: "--" if pd.isna(x) else f"{x:.{d}f}"

# T1: Viet Nam over four cycles
tr = pd.read_csv(OUT / "results_trend/TR1_indicator_trends.csv")
v = tr[tr.CNT == "VNM"].set_index("CYCLE")
spec = [("n", "Students (N)", 0), ("MATH", "Mathematics mean", 1), ("ESCS_SLOPE", "Socio-economic gradient (points per ESCS unit)", 1),
        ("GENDER_GAP_F_minus_M", "Gender gap, girls minus boys (points)", 1), ("COMPUTER", r"Computer for school work at home (\%)", 1),
        ("INTERNET", r"Internet at home (\%)", 1), ("SMARTPHONE3", r"Three or more smartphones at home (\%)", 1),
        ("BELONG_AGREE", r"`I feel like I belong at school' (\%)", 1), ("LONELY_AGREE", r"`I feel lonely at school' (\%)", 1),
        ("BELONG_RAW", "Sense of belonging, six items (1--4)", 2), ("ANX_RAW", "Mathematics anxiety, five items (1--4)", 2),
        ("TENSE_AGREE", r"`I get very tense doing maths homework' (\%)", 1), ("DISC_RAW", "Disciplinary climate in maths (1--4, higher = better)", 2),
        ("NOISE_MOST", r"Noise and disorder in most/every maths lesson (\%)", 1)]
rows = []
for col, lab, d in spec:
    cells = []
    for c in [2012, 2015, 2018, 2022]:
        x = v.loc[c, col] if col in v else np.nan
        se = v.loc[c, col + "_se"] if col + "_se" in v and not pd.isna(x) else np.nan
        cells.append(f"{x:,.0f}" if col == "n" else (f(x, d) + ("" if pd.isna(se) else f" ({f(se, d)})")))
    rows.append(lab + " & " + " & ".join(cells) + NL)
w("tab_vnm_trend", "@{}p{6.6cm}rrrr@{}", ["Indicator & 2012 & 2015 & 2018 & 2022" + NL], rows)

# T2: change 2012 -> 2022, ten systems
ch = pd.read_csv(OUT / "results_trend/TR1b_change_2012_2022.csv")
vars_ = [("INTERNET", 1), ("LONELY_AGREE", 1), ("BELONG_AGREE", 1), ("ANX_RAW", 2), ("TENSE_AGREE", 1), ("NOISE_MOST", 1)]
rows = []
for cnt in ["VNM", "SGP", "JPN", "KOR", "TAP", "HKG", "MAC", "THA", "MYS", "IDN"]:
    cells = []
    for vv, d in vars_:
        r = ch[(ch.CNT == cnt) & (ch["var"] == vv)]
        cells.append("--" if r.empty else f"{r.change.iloc[0]:+.{d}f}{zstar(r.z.iloc[0])}")
    name = NAMES[cnt] if cnt != "VNM" else r"\textbf{Viet Nam}"
    rows.append(name + " & " + " & ".join(cells) + NL)
w("tab_change", "@{}lrrrrrr@{}",
  [r" & Internet & Lonely & Belong & Maths anxiety & Tense & Noise" + NL,
   r" & (pp) & (pp) & (pp) & (1--4) & (pp) & (pp)" + NL], rows)

# T3: relationships by cycle, Viet Nam
rel = pd.read_csv(OUT / "results_trend/TR2_relationships_by_cycle.csv")
lab = {"COMPUTER": "Computer at home (vs none)", "INTERNET": "Internet at home (vs none)",
       "BELONG_RAW": "Sense of belonging (per SD)", "ANX_RAW": "Mathematics anxiety (per SD)"}
rows = []
for term, model in [("COMPUTER", "R1_digital_access"), ("INTERNET", "R1_digital_access"), ("BELONG_RAW", "R2_belonging"), ("ANX_RAW", "R3_anxiety")]:
    cells = []
    for c in [2012, 2015, 2022]:
        r = rel[(rel.CNT == "VNM") & (rel.CYCLE == c) & (rel.term == term) & (rel.model == model)]
        cells.append("--" if r.empty else f"{r.coef.iloc[0]:.1f}{star(r.p.iloc[0])} ({r.se.iloc[0]:.1f})")
    rows.append(lab[term] + " & " + " & ".join(cells) + NL)
w("tab_vnm_rel", "@{}lrrr@{}", ["Predictor of mathematics & 2012 & 2015 & 2022" + NL], rows)

# T4: 2022 descriptives, 13 systems
t1 = pd.read_csv(OUT / "results/T1_descriptives_by_system.csv")
rows = []
for _, r in t1.iterrows():
    name = r"\textbf{Viet Nam}" if r.CNT == "VNM" else r.system
    rows.append(f"{name} & {int(r.n):,} & {f(r.DIST_SELF_MOST)} ({f(r.DIST_SELF_MOST_se)}) & {f(r.DIST_PEERS_MOST)} & "
                f"{f(r.ONLINE_PRESSURE_HIGH) if 'ONLINE_PRESSURE_HIGH' in r else '--'} & {f(r.DEVICE_ANXIETY_HIGH) if 'DEVICE_ANXIETY_HIGH' in r else '--'} & "
                f"{f(r.MATH)} ({f(r.MATH_se)}) & {f(r.ANXMAT, 2)}" + NL)
w("tab_2022_desc", "@{}lrrrrrrr@{}",
  [r"System & $N$ & \multicolumn{2}{c}{Distracted, most lessons (\%)} & Online & Nervous & Maths & ANXMAT" + NL,
   r" & & own devices & peers' devices & pressure (\%) & w/o device (\%) & & " + NL], rows)

# T5: 2022 key coefficients, 13 systems
t3 = pd.read_csv(OUT / "results/T3_math_regressions.csv")
t4 = pd.read_csv(OUT / "results/T4_anxiety_regressions.csv")
rows = []
for cnt in REGION:
    cells = []
    for tab, model in [(t3, "M1_controls"), (t3, "M2_+climate"), (t4, "A1_controls"), (t4, "A2_+climate_relations")]:
        r = tab[(tab.CNT == cnt) & (tab.model == model) & (tab.term == "DIGDIST")]
        d = 1 if tab is t3 else 2
        cells.append("--" if r.empty else f"{r.coef.iloc[0]:.{d}f}{star(r.p.iloc[0])} ({r.se.iloc[0]:.{d}f})")
    name = r"\textbf{Viet Nam}" if cnt == "VNM" else NAMES[cnt]
    rows.append(name + " & " + " & ".join(cells) + NL)
w("tab_2022_models", "@{}lrrrr@{}",
  [r" & \multicolumn{2}{c}{Mathematics (points per SD)} & \multicolumn{2}{c}{Maths anxiety (index per SD)}" + NL,
   r"System & M1 & M2 (+climate) & A1 & A2 (+climate, relations)" + NL], rows)

# T6: school policies
p0 = pd.read_csv(OUT / "results_policy/P0_policy_prevalence.csv")
pm = pd.read_csv(OUT / "results_policy/P1_P3_policy_models.csv")
rows = []
for cnt in REGION:
    r = p0[p0.CNT == cnt].iloc[0]
    name = r"\textbf{Viet Nam}" if cnt == "VNM" else NAMES[cnt]
    rows.append(f"{name} & {f(r.SCH_PHONE_BAN)} & {f(r.SCH_CO_RULES)} & {f(r.SCH_RESP_PROG)} & {f(r.SCH_SEL_POLICY)}" + NL)
w("tab_policy_prev", "@{}lrrrr@{}",
  [r"System & Phone ban & Co-created rules & Responsible-internet & SEL policy" + NL,
   r" & (\% students) & (\%) & programme (\%) & (\%)" + NL], rows)
rows = []
lab = {"SCH_PHONE_BAN": "Phones not allowed on premises", "SCH_CO_RULES": "Rules co-created with students",
       "SCH_RESP_PROG": "Responsible-internet programme", "SCH_SEL_POLICY": "SEL policy"}
for term in lab:
    cells = []
    for cnt, model, d in [("POOLED_13", "P1_digdist", 2), ("POOLED_ANX", "P2_anxiety", 2), ("POOLED_13", "P3_math", 1),
                          ("VNM", "P1_digdist", 2), ("VNM", "P2_anxiety", 2), ("VNM", "P3_math", 1)]:
        r = pm[(pm.CNT == cnt) & (pm.model == model) & (pm.term == term)]
        cells.append("--" if r.empty else f"{r.coef.iloc[0]:.{d}f}{star(r.p.iloc[0])} ({r.se.iloc[0]:.{d}f})")
    rows.append(lab[term] + " & " + " & ".join(cells) + NL)
w("tab_policy_models", "@{}lrrrrrr@{}",
  [r" & \multicolumn{3}{c}{Pooled, 13 systems (system FE)} & \multicolumn{3}{c}{Viet Nam}" + NL,
   r"School policy & Distraction & Anxiety & Maths & Distraction & Anxiety & Maths" + NL], rows)

# T7: system level
s2 = pd.read_csv(OUT / "results_system/S2_system_2022.csv")
rows = []
for _, r in s2.iterrows():
    name = r"\textbf{Viet Nam}" if r.CNT == "VNM" else NAMES[r.CNT]
    pol = str(r.policy_2022) if isinstance(r.policy_2022, str) else "--"
    rows.append(f"{name} & {f(r['IT.NET.USER.ZS'])} & {f(r['NY.GDP.PCAP.PP.KD'] / 1000)} & {pol} & {f(r.suicide_rate_2021)} & {f(r.DIST_SELF_MOST)} & {f(r.SCH_PHONE_BAN)}" + NL)
w("tab_system", "@{}lrrp{4.2cm}rrr@{}",
  [r"System & Internet & GDP pc & National phone policy (PEER) & Suicide & Distracted & School ban" + NL,
   r" & users (\%) & (PPP, \$k) & & rate & (\%) & (\%)" + NL], rows)

# ML performance 2022
ml = pd.read_csv(OUT / "results/T5_ml_performance.csv")
ml[["CNT", "outcome"]] = ml.tag.str.split("_", n=1, expand=True)
rows = []
for cnt in REGION:
    a = ml[(ml.CNT == cnt) & (ml.outcome == "MATH")]
    b = ml[(ml.CNT == cnt) & (ml.outcome == "ANXMAT")]
    name = r"\textbf{Viet Nam}" if cnt == "VNM" else NAMES[cnt]
    g = lambda d, c: "--" if d.empty else f"{d[c].iloc[0]:.3f}"
    rows.append(f"{name} & {g(a, 'r2_gbm')} & {g(a, 'r2_linear')} & {g(b, 'r2_gbm')} & {g(b, 'r2_linear')}" + NL)
w("tab_ml", "@{}lrrrr@{}", [r" & \multicolumn{2}{c}{Mathematics} & \multicolumn{2}{c}{Maths anxiety}" + NL,
                            r"System & LightGBM & Linear & LightGBM & Linear" + NL], rows)

figs = ["results_trend/FigT1_trends_small_multiples.png", "results_trend/FigT2_dumbbell_2012_2022.png",
        "results_trend/FigT3_vnm_shap_by_cycle.png", "results/Fig1_prevalence_digital_distraction.png",
        "results/Fig2_forest_math.png", "results/Fig3_forest_anxiety.png", "results/Fig4_digital_habits_math.png",
        "results/F_shap_bar_VNM_MATH.png", "results/F_shap_dependence_VNM_ANXMAT.png",
        "results_system/FigS1_system_scatter.png", "results_system/FigS2_internet_trajectories.png"]
for fp in figs:
    shutil.copy(OUT / fp, TEX / "figures" / Path(fp).name)
print("ok")

# ---------------- robustness tables ----------------
RB = OUT / "results_robust"
inv = pd.read_csv(RB / "R2_invariance_fit.csv")
rows, prev = [], {}
for _, r in inv.iterrows():
    key = r.scale
    d_cfi = "" if r.model == "configural" else f"{r.CFI - prev[key]:+.3f}"
    if r.model in ("configural", "metric", "scalar"):
        prev[key] = r.CFI
    rows.append(f"{r.scale} & {r.model} & {r.chi2:,.1f} & {int(r.df)} & {r.CFI:.3f} & {d_cfi} & {r.RMSEA:.3f}" + NL)
w("tab_invariance", "@{}llrrrrr@{}", [r"Scale (Viet Nam) & Model & $\chi^2$ & df & CFI & $\Delta$CFI & RMSEA" + NL], rows)

lat = pd.read_csv(RB / "R2_latent_means.csv")
rows = []
for (sc, mod), g in lat.groupby(["scale", "model"], sort=False):
    g = g.set_index("CYCLE")
    cells = [f"{g.loc[c, 'latent_mean_2012SD']:+.2f}" if c in g.index else "--" for c in [2015, 2018, 2022]]
    rows.append(f"{sc} & {mod} & " + " & ".join(cells) + NL)
w("tab_latent", "@{}llrrr@{}", [r"Scale & Model & 2015 & 2018 & 2022" + NL], rows)

it = pd.read_csv(RB / "R1_belonging_items.csv")
lab_it = {"B1": "Not an outsider (reversed)", "B2": "Make friends easily", "B3": "Feel I belong",
          "B4": "Not awkward (reversed)", "B5": "Other students like me", "B6": "Not lonely (reversed)"}
v = it[it.CNT == "VNM"].pivot(index="item", columns="CYCLE", values="pct_belonging_response")
w("tab_items", "@{}lrrrrr@{}", [r"Item (\% giving the belonging response) & 2012 & 2015 & 2018 & 2022 & $\Delta$ 2012--22" + NL],
  [f"{lab_it[i]} & " + " & ".join(f"{v.loc[i, c]:.1f}" for c in [2012, 2015, 2018, 2022]) + f" & {v.loc[i, 2022] - v.loc[i, 2012]:+.1f}" + NL for i in v.index])

r3 = pd.read_csv(RB / "R3_access_wellbeing.csv")
rows = []
for y, lab_y, d in [("LONELY_AGREE", "Lonely (pp)", 1), ("BELONG_RAW", "Belonging (1--4)", 3), ("ANX_RAW", "Maths anxiety (1--4)", 3)]:
    for term in ["COMPUTER", "INTERNET", "SMARTPHONE3"]:
        cells = []
        for c in [2012, 2015, 2018, 2022]:
            r = r3[(r3.CNT == "VNM") & (r3.CYCLE == c) & (r3.outcome == y) & (r3.term == term)]
            cells.append("--" if r.empty else f"{r.coef.iloc[0]:.{d}f}{star(r.p.iloc[0])} ({r.se.iloc[0]:.{d}f})")
        rows.append(f"{lab_y} & {term.title().replace('3', ' 3+')} & " + " & ".join(cells) + NL)
reg = r3[r3.outcome == "LONELY_AGREE"]
w("tab_access", "@{}llrrrr@{}", [r"Outcome & Home access & 2012 & 2015 & 2018 & 2022" + NL], rows)
(TEX / "tables" / "access_region_summary.tex").write_text(
    f"{len(reg)} system-cycle-indicator tests in the region, {int((reg.p < .05).sum())} significant at 5\\%, "
    f"all {('negative' if (reg[reg.p < .05].coef < 0).all() else 'mixed in sign')}", encoding="utf-8")

r4 = pd.read_csv(RB / "R4_dfl_decomposition_vnm.csv")
lab4 = {"LONELY_AGREE": r"Lonely at school (\%)", "BELONG_RAW": "Belonging (1--4)", "ANX_RAW": "Maths anxiety (1--4)",
        "TENSE_AGREE": r"Tense doing maths homework (\%)", "NOISE_MOST": r"Noise in most maths lessons (\%)"}
rows = []
for _, r in r4.iterrows():
    d = 1 if "%" in lab4[r.outcome] else 3
    rows.append(f"{lab4[r.outcome]} & {r.y2012:.{d}f} & {r.y2022:.{d}f} & {r.change:+.{d}f} & {r.explained_composition:+.{d}f} & "
                f"{r.explained_access:+.{d}f} & {r.unexplained:+.{d}f}" + NL)
w("tab_dfl", "@{}lrrrrrr@{}", [r" & 2012 & 2022 & Change & Composition & Home access & Unexplained" + NL], rows)

r5 = pd.read_csv(RB / "R5_ipw_phone_ban.csv")
lab5 = {"DIGDIST": "Digital distraction (SD)", "ANXMAT": "Maths anxiety (SD)", "MATH": "Mathematics (points)"}
rows = []
for y in ["DIGDIST", "ANXMAT", "MATH"]:
    cells = []
    for sp in ["unadjusted", "IPW + controls"]:
        r = r5[(r5.outcome == y) & (r5.spec == sp)].iloc[0]
        d = 1 if y == "MATH" else 3
        cells.append(f"{r.coef:.{d}f}{star(r.p)} ({r.se:.{d}f})")
    rows.append(f"{lab5[y]} & " + " & ".join(cells) + f" & {int(r.n):,}" + NL)
w("tab_ipw", "@{}lrrr@{}", [r"Outcome & Unadjusted & IPW + student controls & $N$" + NL], rows)

r0 = pd.read_csv(RB / "R0_admin_mode_share_computer.csv").set_index("CNT")
w("tab_mode", "@{}lrrrr@{}", ["System & 2012 & 2015 & 2018 & 2022" + NL],
  [f"{NAMES[c]} & " + " & ".join("--" if pd.isna(r0.loc[c, str(y)]) else ("computer" if r0.loc[c, str(y)] > .5 else "paper")
                                  for y in [2012, 2015, 2018, 2022]) + NL for c in r0.index])

r6 = pd.read_csv(RB / "R6_bh_qvalues.csv")
(TEX / "tables" / "bh_summary.tex").write_text("; ".join(
    f"{m.split('_')[0]}: {int((g.q_BH < .05).sum())}/{len(g)}" for m, g in r6.groupby("model", sort=False)), encoding="utf-8")
print("robustness tables ok")
