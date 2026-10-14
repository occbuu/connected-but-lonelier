"""Additional, explicitly exploratory robustness checks; no causal identification."""
import ast, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).parent))
from pisa_utils import W,RW,BRR_FACTOR,weighted_mean_se,wls_brr_pv,recode
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/results_upgrade'; OUT.mkdir(exist_ok=True)
FIG=(ROOT/'latex' if (ROOT/'latex/main.tex').exists() else ROOT/'Paper1_PISA2022')/'figures'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
tr=pd.read_parquet(ROOT/'outputs/pisa_trend_2012_2022.parquet',filters=[('CNT','==','VNM')],use_threads=False)
def save(fig,name):
    fig.tight_layout()
    for target in (OUT,FIG):
        fig.savefig(target/(name+'.png'),dpi=300,bbox_inches='tight')
        fig.savefig(target/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

if '--calibration-only' not in sys.argv:
    # Weighted item distributions on available cases, not complete-case CFA samples.
    rows=[]
    for cycle,d in tr.groupby('CYCLE'):
        for item in [f'B{i}' for i in range(1,7)]+[f'A{i}' for i in range(1,6)]:
            g=d.dropna(subset=[item,W]); x=g[item].to_numpy(float); w=g[W].to_numpy(float)
            if not len(g): continue
            mean=np.average(x,weights=w)
            row=dict(cycle=cycle,item=item,n=len(g),missing_pct=100*(1-len(g)/len(d)),mean=mean,
                     sd=np.sqrt(np.average((x-mean)**2,weights=w)))
            row.update({f'p{k}':100*np.average(x==k,weights=w) for k in range(1,5)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'U1_item_descriptives.csv',index=False)
    
    # Reuse tested moment-CFA functions only, without executing old analysis workflow.
    source=Path(__file__).with_name('10_robustness.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    namespace=dict(np=np,optimize=optimize)
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('wmoments','mgcfa','null_chi2','fit_idx'):
            code=ast.get_source_segment(source,node)
            code=code.replace('N=N, est=est)','N=N, est=est, converged=bool(r.success), message=str(r.message))')
            exec(code,namespace)
    fit_rows=[]; latent_rows=[]
    # Positive B3 is the marker, allowing ALL three negative intercepts to be freed.
    items=['B3','B2','B5','B1','B4','B6']; rc=((3,4),(3,5),(4,5))
    for cycles in ([2012,2015,2018,2022],[2012,2015,2022]):
        groups=[]
        for cycle in cycles:
            d=tr[tr.CYCLE==cycle].dropna(subset=items+[W])
            m,s=namespace['wmoments'](d[items].to_numpy(float),d[W].to_numpy(float))
            groups.append((m,s,len(d)))
        null=namespace['null_chi2'](groups)
        for level,free in [('configural',()),('metric',()),('scalar',()),('scalar',(3,4,5))]:
            result=namespace['mgcfa'](groups,level,free,rc)
            cfi,rmsea=namespace['fit_idx'](result,null)
            label='partial scalar: B1/B4/B6 free' if free else level
            fit_rows.append(dict(cycles='/'.join(map(str,cycles)),model=label,n=result['N'],
                                 chi2=result['chi2'],df=result['df'],cfi=cfi,rmsea=rmsea,
                                 converged=result['converged'],optimizer_message=result['message']))
            if level=='scalar':
                ref=result['est'][0]
                for cycle,est in zip(cycles,result['est']):
                    latent_rows.append(dict(cycles='/'.join(map(str,cycles)),model=label,cycle=cycle,
                                            mean_2012SD=(est[5]-ref[5])/np.sqrt(ref[4])))
            print('CFA',cycles,label,cfi,rmsea,result['converged'],flush=True)
    pd.DataFrame(fit_rows).to_csv(OUT/'U2_partial_scalar_fit.csv',index=False)
    pd.DataFrame(latent_rows).to_csv(OUT/'U2_partial_scalar_means.csv',index=False)
    
    # DFL: independently vary each cycle's 80 replicates; refit propensity each time.
    v=tr[tr.CYCLE.isin([2012,2022])].copy()
    composition=['FEMALE','GRADE','REPEAT','BOOKS']; exposure=['COMPUTER','INTERNET']
    v['GRADE']=v.GRADE.clip(-2,1)
    for col in composition+exposure: v[col]=v[col].fillna(-1)
    d12=v[v.CYCLE==2012].reset_index(drop=True); d22=v[v.CYCLE==2022].reset_index(drop=True)
    pool=pd.concat([d12,d22],ignore_index=True)
    designs=[pd.get_dummies(pool[cols].astype('category'),drop_first=True).to_numpy(float)
             for cols in (composition,composition+exposure)]
    ycols=['LONELY_AGREE','BELONG_RAW','ANX_RAW','TENSE_AGREE','NOISE_MOST']
    strategies=['untrimmed','ps_01_99','ps_05_95','ratio_01_99','ratio_05_95']
    def estimates(w12,w22,diagnostics=False):
        probabilities=[]
        weights=np.r_[w12,w22]; normalized=weights/weights.mean()
        target=np.r_[np.zeros(len(d12)),np.ones(len(d22))]
        for X in designs:
            model=LogisticRegression(C=1e4,max_iter=4000).fit(X,target,sample_weight=normalized)
            probabilities.append(np.clip(model.predict_proba(X[len(d12):])[:,1],1e-8,1-1e-8))
        result=[]; diag=[]
        for strategy in strategies:
            cf_weights=[]
            for stage,p in enumerate(probabilities):
                pp=np.clip(p,.01,.99) if strategy=='ps_01_99' else np.clip(p,.05,.95) if strategy=='ps_05_95' else p
                ratio=(1-pp)/pp
                if strategy.startswith('ratio_'):
                    q=.01 if strategy=='ratio_01_99' else .05
                    ratio=np.clip(ratio,*np.quantile(ratio,[q,1-q]))
                cw=w22*ratio; cf_weights.append(cw)
                if diagnostics:
                    diag.append(dict(strategy=strategy,stage=stage+1,ess=cw.sum()**2/np.sum(cw**2),
                                     n=len(cw),min_ps=p.min(),max_ps=p.max(),outside_05_95=np.mean((p<.05)|(p>.95))))
            for y in ycols:
                a=d12[y].to_numpy(float); b=d22[y].to_numpy(float)
                good12=np.isfinite(a); good22=np.isfinite(b)
                mu12=np.average(a[good12],weights=w12[good12]); mu22=np.average(b[good22],weights=w22[good22])
                cf=[np.average(b[good22],weights=w[good22]) for w in cf_weights]
                sc=100 if y.endswith('AGREE') or y=='NOISE_MOST' else 1
                result.append(np.array([mu12,mu22,mu22-mu12,mu22-cf[0],cf[0]-cf[1],cf[1]-mu12])*sc)
        return np.array(result),diag
    base,diag=estimates(d12[W].to_numpy(),d22[W].to_numpy(),True)
    variance=np.zeros_like(base)
    for side in [2012,2022]:
        for k,rw in enumerate(RW):
            w12=d12[rw if side==2012 else W].to_numpy()
            w22=d22[rw if side==2022 else W].to_numpy()
            replicate,_=estimates(w12,w22)
            variance+=BRR_FACTOR*(replicate-base)**2
            if (k+1)%20==0: print('DFL replicates',side,k+1,flush=True)
    rows=[]
    for i,(strategy,y) in enumerate((s,y) for s in strategies for y in ycols):
        row=dict(strategy=strategy,outcome=y,n2012=int(d12[y].notna().sum()),n2022=int(d22[y].notna().sum()))
        for j,key in enumerate(['mean2012','mean2022','change','composition','access','unexplained']):
            row[key]=base[i,j]; row[key+'_se']=np.sqrt(variance[i,j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'U3_dfl_sensitivity.csv',index=False)
    pd.DataFrame(diag).to_csv(OUT/'U3_dfl_weight_diagnostics.csv',index=False)
    
    # Pandemic-spanning contrasts are temporal descriptions, never a COVID effect.
    rows=[]
    for a,b in [(2012,2018),(2018,2022)]:
        for y in ['LONELY_AGREE','BELONG_AGREE','BELONG_RAW']:
            ma,sa=weighted_mean_se(tr[tr.CYCLE==a],y); mb,sb=weighted_mean_se(tr[tr.CYCLE==b],y)
            sc=1 if y=='BELONG_RAW' else 100
            rows.append(dict(start=a,end=b,outcome=y,change=sc*(mb-ma),se=sc*np.hypot(sa,sb),annual_change=sc*(mb-ma)/(b-a)))
    pd.DataFrame(rows).to_csv(OUT/'U4_pandemic_interval.csv',index=False)
    
    # Conditional permutation of the joint access vector: an estimator diagnostic.
    rng=np.random.default_rng(20261014); rows=[]
    for cycle in [2012,2022]:
        for y in ['LONELY_AGREE','ANX_RAW']:
            controls=['ESCS','FEMALE','GRADE','REPEAT','BOOKS']; access=['COMPUTER','INTERNET']
            d=tr[tr.CYCLE==cycle].dropna(subset=[y]+controls+access+[W]).reset_index(drop=True)
            Z=d[controls].to_numpy(float); Z=(Z-Z.mean(0))/np.where(Z.std(0)>0,Z.std(0),1)
            X=np.c_[np.ones(len(d)),Z,d[access].to_numpy(float)]; w=d[W].to_numpy(float); target=d[y].to_numpy(float)
            def coef(xx): return np.linalg.lstsq(xx*np.sqrt(w[:,None]),target*np.sqrt(w),rcond=None)[0][-2:]
            observed=coef(X); null=[]
            blocks=d.groupby([d.FEMALE,pd.qcut(d.ESCS,5,duplicates='drop')],observed=True).indices
            for k in range(199):
                perm=X.copy()
                for idx in blocks.values(): perm[idx,-2:]=X[rng.permutation(idx),-2:]
                null.append(coef(perm))
            null=np.array(null)
            for j,a in enumerate(access):
                rows.append(dict(cycle=cycle,outcome=y,access=a,n=len(d),coef=observed[j],
                                 permutation_mean=null[:,j].mean(),permutation_sd=null[:,j].std(ddof=1),
                                 permutation_p=(1+np.sum(np.abs(null[:,j])>=abs(observed[j])))/200))
    pd.DataFrame(rows).to_csv(OUT/'U5_permutation_falsification.csv',index=False)
    
df=recode(pd.read_parquet(ROOT/'outputs/pisa22_analysis.parquet',filters=[('CNT','==','VNM')],use_threads=False))
comparisons=[]
for domain in ['READ','MATH']:
    result=wls_brr_pv(df,[f'PV{i}{domain}' for i in range(1,11)],['ICTRES','ESCS','FEMALE','GRADE','REPEAT'])
    result.insert(0,'domain',domain); comparisons.append(result[result.term=='ICTRES'])
pd.concat(comparisons).to_csv(OUT/'U5b_reading_comparison.csv',index=False)

# OOF regression calibration plus a separate fixed-threshold classification task.
spec=importlib.util.spec_from_file_location('ml',Path(__file__).with_name('03_ml_shap.py'))
ml=importlib.util.module_from_spec(spec); spec.loader.exec_module(ml)
params=dict(ml.PARAMS,n_jobs=4)
features=[f for f in ml.FEATURES if f in df and df[f].notna().mean()>.5]
performance=[]; regression_bins=[]; probability_bins=[]
for outcome,targets in [('MATH',[f'PV{i}MATH' for i in range(1,11)]),('ANXMAT',['ANXMAT'])]:
    d=df.dropna(subset=targets).reset_index(drop=True)
    feats=[f for f in features if outcome!='ANXMAT' or f!='ANXMAT']
    X=d[feats]; w=d[W].to_numpy(float); groups=d.CNTSCHID.to_numpy()
    for k,target_col in enumerate(targets):
        y=d[target_col].to_numpy(float); pred=np.zeros(len(d)); probability=np.zeros(len(d)); baseline=np.zeros(len(d))
        for train,test in GroupKFold(5).split(X,y,groups):
            model=lgb.LGBMRegressor(**params,random_state=k).fit(X.iloc[train],y[train],sample_weight=w[train])
            pred[test]=model.predict(X.iloc[test])
            if outcome=='MATH':
                binary=(y<420.07).astype(int)
                classifier=lgb.LGBMClassifier(**params,random_state=k).fit(X.iloc[train],binary[train],sample_weight=w[train])
                probability[test]=classifier.predict_proba(X.iloc[test])[:,1]
                baseline[test]=np.average(binary[train],weights=w[train])
        slope=np.linalg.lstsq(np.c_[np.ones(len(y)),pred]*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
        row=dict(outcome=outcome,pv=target_col,n=len(d),schools=len(np.unique(groups)),r2=r2_score(y,pred,sample_weight=w),
                 rmse=np.sqrt(np.average((y-pred)**2,weights=w)),calibration_intercept=slope[0],calibration_slope=slope[1])
        bins=pd.qcut(pred,10,labels=False,duplicates='drop')
        for b in np.unique(bins):
            mask=bins==b
            regression_bins.append(dict(outcome=outcome,pv=target_col,bin=int(b),n=int(mask.sum()),
                                        predicted=np.average(pred[mask],weights=w[mask]),observed=np.average(y[mask],weights=w[mask])))
        if outcome=='MATH':
            binary=(y<420.07).astype(int)
            row.update(brier=np.average((binary-probability)**2,weights=w),brier_baseline=np.average((binary-baseline)**2,weights=w))
            bins=pd.qcut(probability,10,labels=False,duplicates='drop')
            for b in np.unique(bins):
                mask=bins==b
                probability_bins.append(dict(pv=target_col,bin=int(b),n=int(mask.sum()),predicted=np.average(probability[mask],weights=w[mask]),observed=np.average(binary[mask],weights=w[mask])))
        performance.append(row)
        print('Calibration',target_col,row,flush=True)
pd.DataFrame(performance).to_csv(OUT/'U6_calibration_performance.csv',index=False)
pd.DataFrame(regression_bins).to_csv(OUT/'U6_regression_calibration_bins.csv',index=False)
pd.DataFrame(probability_bins).to_csv(OUT/'U6_probability_calibration_bins.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(10,3.2))
reg=pd.DataFrame(regression_bins)
for ax,outcome,panel in zip(axes[:2],['MATH','ANXMAT'],['(a)','(b)']):
    points=reg[reg.outcome==outcome].groupby('bin')[['predicted','observed']].mean()
    ax.plot(points.predicted,points.observed,'o-',color='#2a6fdb'); lim=[min(points.min()),max(points.max())]
    ax.plot(lim,lim,'--',color='grey'); ax.set_title(panel+' '+outcome); ax.set_xlabel('OOF prediction'); ax.set_ylabel('Observed mean')
points=pd.DataFrame(probability_bins).groupby('bin')[['predicted','observed']].mean()
axes[2].plot(points.predicted,points.observed,'o-',color='#2a6fdb'); axes[2].plot([0,1],[0,1],'--',color='grey')
axes[2].set(title='(c) Below mathematics Level 2',xlabel='OOF probability',ylabel='Observed proportion',xlim=(0,1),ylim=(0,1))
save(fig,'FigU_calibration')
print('Upgrade analyses complete',flush=True)
