"""Regenerate the Vietnamese anxiety SHAP dependence figure as vector PDF and 300 dpi PNG."""
import importlib.util
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
spec=importlib.util.spec_from_file_location('ml',ROOT/'code/03_ml_shap.py'); ml=importlib.util.module_from_spec(spec); spec.loader.exec_module(ml)
d=ml.recode(pd.read_parquet(ROOT/'outputs/pisa22_analysis.parquet',filters=[('CNT','==','VNM')])).dropna(subset=['ANXMAT']).reset_index(drop=True)
features=[f for f in ml.FEATURES if f!='ANXMAT' and f in d and d[f].notna().mean()>.5]
X=d[features].astype(float); w=d['W_FSTUWT'].to_numpy(float)
model=lgb.LGBMRegressor(**dict(ml.PARAMS,n_jobs=4)).fit(X,d.ANXMAT,sample_weight=w)
sv=shap.TreeExplainer(model).shap_values(X)
fig,axes=plt.subplots(1,2,figsize=(7,3.5))
for i,(ax,f) in enumerate(zip(axes,['DIGDIST','EASILY_DISTRACTED'])):
    j=features.index(f); x=X[f].to_numpy(float); ok=np.isfinite(x)
    ax.scatter(x[ok]+np.random.default_rng(0).uniform(-.15,.15,ok.sum()),sv[ok,j],s=2,alpha=.2,rasterized=True)
    grouping=np.round(x[ok]*2)/2 if f=='DIGDIST' else x[ok]
    means=pd.Series(sv[ok,j]).groupby(grouping).mean()
    ax.plot(means.index,means.values,'k-o',ms=3)
    ax.axhline(0,color='grey',lw=.7); ax.set_ylabel('SHAP (anxiety units)')
    ax.set_xlabel(ml.FEATURES[f]); ax.set_title(f'({chr(97+i)})')
fig.tight_layout()
figdir=(ROOT/'latex' if (ROOT/'latex/main.tex').exists() else ROOT/'Paper1_PISA2022')/'figures'
for directory in [ROOT/'outputs/results',figdir]:
    for ext in ['pdf','png']: fig.savefig(directory/f'F_shap_dependence_VNM_ANXMAT.{ext}',dpi=300)
print('SHAP regenerated',len(d),'students',d.CNTSCHID.nunique(),'schools')
