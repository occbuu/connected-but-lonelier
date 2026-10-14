"""Interpretable machine learning: LightGBM + SHAP on PISA 2022 (Viet Nam focus, regional comparison).

For each system and outcome:
  * outcome = each of the 10 maths PVs (results averaged across PVs) or ANXMAT;
  * 5-fold cross-validation grouped by school (no school appears in both train and test);
  * student final weights used as sample weights; LightGBM handles missing values natively;
  * baseline = weighted linear regression (median-imputed) for comparison;
  * SHAP TreeExplainer on the model fitted to all data -> mean |SHAP| importance.
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))
from pisa_utils import REGION, NAMES, PV_MATH, W, recode

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RES = OUT / "results"
RES.mkdir(exist_ok=True)

FEATURES = {
    # digital
    "DIGDIST": "Digital distraction in maths lessons",
    "ONLINE_PRESSURE": "Pressure to be online in class",
    "DEVICE_ANXIETY": "Nervous without device",
    "NOTIF_OFF_CLASS": "Turns off notifications in class",
    "NOTIF_OFF_SLEEP": "Turns off notifications at bedtime",
    "DEVICE_NEAR_HOME": "Keeps device near to answer messages",
    "DEVICE_OPEN_CLASS": "Device open in class for notes/search",
    "EASILY_DISTRACTED": "Easily distracted (self)",
    "ICTRES": "ICT resources at home",
    # background
    "ESCS": "Socio-economic status (ESCS)", "FEMALE": "Female", "GRADE": "Grade (rel. modal)",
    "REPEAT": "Grade repetition", "SKIPPING": "Skipping classes", "TARDYSD": "Arriving late",
    "EXERPRAC": "Exercise/sport", "STUDYHMW": "Homework time", "WORKPAY": "Paid work",
    "WORKHOME": "Household work",
    # school climate / relationships
    "BELONG": "Sense of belonging", "BULLIED": "Being bullied", "FEELSAFE": "Feeling safe",
    "RELATST": "Teacher-student relationship", "TEACHSUP": "Teacher support", "DISCLIM_ND": "Disciplinary climate (non-digital)",
    "FAMSUP": "Family support",
    # socio-emotional / maths dispositions
    "ANXMAT": "Mathematics anxiety", "MATHEFF": "Mathematics self-efficacy", "MATHPERS": "Effort & persistence (maths)",
    "GROSAGR": "Growth mindset", "PERSEVAGR": "Perseverance", "CURIOAGR": "Curiosity",
    "STRESAGR": "Stress resistance", "EMOCOAGR": "Emotional control", "COOPAGR": "Cooperation",
    "EMPATAGR": "Empathy", "ASSERAGR": "Assertiveness",
}
DIGITAL = list(FEATURES)[:9]
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def run(d, target_cols, feats, tag):
    X = d[feats]
    groups = d["CNTSCHID"].to_numpy()
    w = d[W].to_numpy(float)
    gkf = GroupKFold(n_splits=5)
    r2_gbm, r2_lin, shap_abs, shap_store = [], [], [], None
    for k, y_col in enumerate(target_cols):
        y = d[y_col].to_numpy(float)
        pred_g, pred_l = np.zeros_like(y), np.zeros_like(y)
        for tr, te in gkf.split(X, y, groups):
            m = lgb.LGBMRegressor(**PARAMS, random_state=k).fit(X.iloc[tr], y[tr], sample_weight=w[tr])
            pred_g[te] = m.predict(X.iloc[te])
            med = X.iloc[tr].median()
            lin = LinearRegression().fit(X.iloc[tr].fillna(med), y[tr], sample_weight=w[tr])
            pred_l[te] = lin.predict(X.iloc[te].fillna(med))
        r2_gbm.append(r2_score(y, pred_g, sample_weight=w))
        r2_lin.append(r2_score(y, pred_l, sample_weight=w))
        full = lgb.LGBMRegressor(**PARAMS, random_state=k).fit(X, y, sample_weight=w)
        sv = shap.TreeExplainer(full).shap_values(X)
        shap_abs.append(np.average(np.abs(sv), axis=0, weights=w))
        if k == 0:
            shap_store = (sv, X)
    imp = pd.DataFrame({"feature": feats, "label": [FEATURES[f] for f in feats],
                        "mean_abs_shap": np.mean(shap_abs, 0), "sd_across_pv": np.std(shap_abs, 0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp["rank"] = range(1, len(imp) + 1)
    imp["is_digital"] = imp.feature.isin(DIGITAL)
    perf = dict(tag=tag, n=len(d), r2_gbm=np.mean(r2_gbm), r2_gbm_sd=np.std(r2_gbm),
                r2_linear=np.mean(r2_lin), r2_linear_sd=np.std(r2_lin))
    return imp, perf, shap_store


def plots(imp, store, tag, title):
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top.label, top.mean_abs_shap, xerr=top.sd_across_pv,
            color=np.where(top.is_digital, "#d1495b", "#4c72b0"))
    ax.set_xlabel("Mean |SHAP value| (outcome units)"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(RES / f"F_shap_bar_{tag}.png", dpi=200); plt.close(fig)
    sv, X = store
    shap.summary_plot(sv, X.rename(columns=FEATURES), max_display=15, show=False, plot_size=(8, 6))
    plt.title(title, fontsize=10); plt.tight_layout()
    plt.savefig(RES / f"F_shap_beeswarm_{tag}.png", dpi=200); plt.close("all")
    # dependence plots for the two key digital variables
    dep = [f for f in ["DIGDIST", "EASILY_DISTRACTED", "ONLINE_PRESSURE", "NOTIF_OFF_CLASS"] if f in X][:3]
    fig, axes = plt.subplots(1, len(dep), figsize=(4 * len(dep), 3.6), squeeze=False)
    for ax, f in zip(axes[0], dep):
        if f in X:
            j = list(X.columns).index(f)
            x = X[f].to_numpy(float); ok = np.isfinite(x)
            jitter = np.random.default_rng(0).uniform(-.15, .15, ok.sum())
            ax.scatter(x[ok] + jitter, sv[ok, j], s=3, alpha=.25)
            means = pd.Series(sv[ok, j]).groupby(x[ok]).mean()
            if f == "DIGDIST":
                means = pd.Series(sv[ok, j]).groupby(np.round(x[ok] * 2) / 2).mean()
            ax.plot(means.index, means.values, "k-o", ms=4)
            ax.axhline(0, c="grey", lw=.8); ax.set_xlabel(FEATURES[f]); ax.set_ylabel("SHAP")
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    fig.savefig(RES / f"F_shap_dependence_{tag}.png", dpi=200); plt.close(fig)


def main(countries):
    df = recode(pd.read_parquet(OUT / "pisa22_analysis.parquet"))
    perfs, imps = [], []
    for cnt in countries:
        d = df[df.CNT == cnt].reset_index(drop=True)
        feats = [f for f in FEATURES if f in d and d[f].notna().mean() > .5]
        for outcome, targets, fs in [("MATH", PV_MATH, feats),
                                     ("ANXMAT", ["ANXMAT"], [f for f in feats if f != "ANXMAT"])]:
            dd = d.dropna(subset=targets)
            if len(dd) < 500:  # outcome not administered in this system (e.g. ANXMAT in THA/IDN/PHL)
                continue
            tag = f"{cnt}_{outcome}"
            imp, perf, store = run(dd, targets, fs, tag)
            imp.insert(0, "tag", tag); imps.append(imp); perfs.append(perf)
            if cnt == "VNM" or len(countries) <= 3:
                plots(imp, store, tag, f"{NAMES[cnt]} – {'Mathematics (10 PVs)' if outcome == 'MATH' else 'Mathematics anxiety'}")
            print(perf)
    sfx = "" if set(countries) == set(REGION) else "_" + "-".join(countries)
    pd.DataFrame(perfs).to_csv(RES / f"T5_ml_performance{sfx}.csv", index=False)
    imp_all = pd.concat(imps)
    imp_all.to_csv(RES / f"T6_shap_importance{sfx}.csv", index=False)
    if sfx:
        return
    # rank of digital features across systems
    rank = imp_all[imp_all.is_digital].pivot_table(index="label", columns="tag", values="rank")
    rank.to_csv(RES / "T7_digital_feature_ranks.csv")


if __name__ == "__main__":
    main(sys.argv[1:] or REGION)
