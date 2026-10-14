"""Generate manuscript supplement directly from the saved local robustness analyses."""
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'outputs/results_upgrade'; TEX=ROOT/'latex'
def read(name): return pd.read_csv(RES/name)
def tab(cols, rows, spec=None):
    return '\\begin{tabular}{@{}'+(spec or 'l'+'r'*(len(cols)-1))+'@{}}\n\\toprule\n'+' & '.join(cols)+r'\\'+'\n\\midrule\n'+'\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'
def table(caption,label,body,note):
    return '\\begin{table}[H]\\centering\\scriptsize\n\\caption{'+caption+'}\\label{'+label+'}\n'+body+'\\par\\smallskip\\raggedright\\footnotesize '+note+'\n\\end{table}\n'
parts=[]
fit=read('U2_partial_scalar_fit.csv'); means=read('U2_partial_scalar_means.csv')
assert fit.converged.all()
rows=[]
for _,r in fit.iterrows():
    metric=fit[(fit.cycles==r.cycles)&(fit.model=='metric')].cfi.iloc[0]
    rows.append(['Four' if '2018' in r.cycles else 'Three', 'Partial (negative free)' if r.model.startswith('partial') else r.model.capitalize(),f'{r.n:,}',r.df,f'{r.cfi:.3f}',f'{r.rmsea:.3f}',f'{r.cfi-metric:+.3f}' if 'scalar' in r.model else '--'])
parts.append(table('Belonging: targeted partial-scalar invariance, Viet Nam','tab:partial',tab(['Cycles','Model','$n$','df','CFI','RMSEA','$\\Delta$CFI'],rows,'llrrrrr'),
    'Final student weights; complete cases on all six items. Three-cycle models omit 2018. B3 (positive belonging) anchors the factor; B1, B4 and B6 intercepts are free across groups, with correlated residuals among these negative items. All models converged. Scalar and partial-scalar $\\Delta$CFI are relative to the metric model; neither meets the $-0.010$ criterion. Weighted-moment ML treats the four response categories as continuous; fit statistics are not survey-adjusted chi-square tests.'))
rows=[]
for _,r in means[means.cycle==2022].iterrows(): rows.append(['Four' if '2018' in r.cycles else 'Three','Partial (negative free)' if r.model.startswith('partial') else 'Full scalar',f'{r.mean_2012SD:.3f}'])
parts.append(table('Belonging: sensitivity of the 2022 latent mean relative to 2012','tab:partialmeans',tab(['Cycles','Specification','2012 SD units'],rows,'llr'),'The full-scalar specification is a diagnostic comparison and fails invariance. The targeted partial-scalar models also fail; these point estimates do not establish a common latent-mean change. No design-based confidence interval is claimed.'))
items=read('U1_item_descriptives.csv')
assert np.allclose(items[['p1','p2','p3','p4']].sum(axis=1),100)
for prefix,title in [('B','Belonging'),('A','Mathematics anxiety')]:
    rows=[]
    for _,r in items[items.item.str.startswith(prefix)].iterrows():
        rows.append([r['item'],int(r.cycle),f'{r.n:,}',f'{r.missing_pct:.1f}',f'{r["mean"]:.2f}',f'{r.sd:.2f}']+[f'{r[c]:.1f}' for c in ['p1','p2','p3','p4']])
    parts.append(table(title+': item distributions by cycle, Viet Nam','tab:items'+prefix,tab(['Item','Year','$n$','Missing\\%','Mean','SD','1\\%','2\\%','3\\%','4\\%'],rows),
        'Available-case means, population SDs and category percentages use final student weights; missing percentage is unweighted relative to the Vietnamese cycle sample. B1 outsider, B2 friends, B3 belonging, B4 awkward, B5 liked, B6 lonely; B1/B4/B6 are reversed, so higher B responses indicate more belonging. A1 worry about difficulty, A2 tense with homework, A3 nervous with problems, A4 helpless, A5 worry about poor grades; higher A responses indicate more anxiety. Items A1--A5 are available in 2012/2022 only. SD is dispersion, not a standard error.'))
dfl=read('U3_dfl_sensitivity.csv'); assert np.allclose(dfl.change,dfl.composition+dfl.access+dfl.unexplained)
names={'LONELY_AGREE':'Loneliness (pp)','BELONG_RAW':'Belonging (1--4)','ANX_RAW':'Anxiety (1--4)','TENSE_AGREE':'Tension (pp)','NOISE_MOST':'Noise (pp)'}
strategies={'untrimmed':'Untrimmed','ps_01_99':'PS .01/.99','ps_05_95':'PS .05/.95','ratio_01_99':'Ratio 1/99 pct','ratio_05_95':'Ratio 5/95 pct'}
rows=[]
for _,r in dfl.iterrows(): rows.append([names[r.outcome],strategies[r.strategy]]+[f'{r[c]:.3f} ({r[c+"_se"]:.3f})' for c in ['composition','access','unexplained']])
parts.append(table('DFL sensitivity to propensity and density-ratio clipping, Viet Nam','tab:dflsens',tab(['Outcome','Weights','Composition (SE)','Access (SE)','Residual (SE)'],rows,'llrrr'),
    'Final student weights multiplied by sequential reweighting ratios. PS clips probabilities; ratio clipping winsorises the density ratio at empirical percentiles, without discarding observations. Composition: gender, grade, repetition, books; access adds computer/internet. Missing covariates form separate categories; outcomes use available cases. All 80 Fay BRR replicates are refitted separately in each independent cycle (160 fits per stage/specification); variances are summed across cycles. pp = percentage points. Residual is not a causal effect.'))
base=dfl[dfl.strategy=='untrimmed']; rows=[]
for _,r in base.iterrows(): rows.append([names[r.outcome]]+[f'{r[c]:.2f} ({r[c+"_se"]:.2f})' for c in ['change','composition','access','unexplained']])
(TEX/'tables/tab_dfl.tex').write_text(tab(['Outcome','Change','Composition','Access','Residual'],rows,'lrrrr'),encoding='utf-8')
diag=read('U3_dfl_weight_diagnostics.csv')
rows=[]
for _,r in diag.iterrows(): rows.append([strategies[r.strategy],int(r.stage),f'{r.n:,}',f'{r.ess:.0f}',f'{r.min_ps:.3f}',f'{r.max_ps:.3f}',f'{100*r.outside_05_95:.1f}'])
parts.append(table('DFL overlap and effective sample size diagnostics, 2022 cohort','tab:overlap',tab(['Weights','Stage','$n$','ESS','Min PS','Max PS','Outside .05/.95\\%'],rows,'lrrrrrr'),'Stage 1 uses composition, stage 2 adds access. ESS is $(\\sum w)^2/\\sum w^2$ after multiplying the final student weight by the reweighting ratio, across all 6,068 students before excluding outcome missingness. PS ranges and the outside percentage refer to original estimated probabilities; the latter is an unweighted percentage. Limited overlap can inflate uncertainty even when point estimates are stable.'))
pandemic=read('U4_pandemic_interval.csv'); rows=[]
for _,r in pandemic.iterrows(): rows.append([f'{int(r.start)}--{int(r.end)}',names.get(r.outcome,'Belonging agreement (pp)'),f'{r.change:.3f}',f'{r.se:.3f}',f'{r.annual_change:.3f}'])
parts.append(table('Trends before and across the pandemic interval, Viet Nam','tab:pandemic',tab(['Interval','Outcome','Change','SE','Per year'],rows,'llrrr'),'Final student weights and all 80 BRR replicates; independent-cycle variances are summed. PISA observations are not available in 2020/2021, so interval differences cannot identify a pandemic effect. Agreement proportions use available responses; raw scale means use the harmonised scale.'))
perm=read('U5_permutation_falsification.csv'); rows=[]
for _,r in perm.iterrows():
    factor=100 if r.outcome=='LONELY_AGREE' else 1
    rows.append([int(r.cycle),names[r.outcome],r.access.capitalize(),f'{r.n:,}',f'{r.coef*factor:.3f}',f'{r.permutation_mean*factor:.3f}',f'{r.permutation_sd*factor:.3f}',f'{r.permutation_p:.3f}'])
parts.append(table('Conditional permutation diagnostic for home access, Viet Nam','tab:perm',tab(['Year','Outcome','Access','$n$','Coefficient','Null mean','Null SD','$p$'],rows,'llrrrrrr'),'Joint computer/internet labels are shuffled 199 times within gender by ESCS-quintile strata, holding outcomes and survey weights fixed. WLS controls ESCS, gender, grade, repetition and books; complete cases are used. The two-sided empirical $p$ uses an add-one correction. This diagnostic is conditional on the chosen strata and does not test absence of causal effects or replace design-based inference.'))
comp=read('U5b_reading_comparison.csv'); rows=[]
for _,r in comp.iterrows(): rows.append([r.domain,f'{r.n:,}',f'{r.coef:.2f}',f'{r.se:.2f}',r'$<.001$'])
parts.append(table('Reading as a comparison outcome, Viet Nam, PISA 2022','tab:reading',tab(['Domain','$n$','ICT resources coefficient','SE','$p$'],rows,'lrrrr'),'ICTRES is standardised within system. Both outcomes use all ten plausible values, final weights, all 80 BRR replicates and Rubin combination; complete cases on ESCS, gender, grade, repetition and ICTRES. Reading is associated with ICT resources and is not a technology-unrelated negative control. Coefficients are achievement points per predictor SD.'))
cal=read('U6_calibration_performance.csv'); math=cal[cal.outcome=='MATH']; anxiety=cal[cal.outcome=='ANXMAT'].iloc[0]
assert len(math)==10 and math.brier.between(0,1).all()
rows=[]
for name,g in [('Mathematics',math),('Anxiety',cal[cal.outcome=='ANXMAT'])]:
    rows.append([name,f'{int(g.n.iloc[0]):,}',int(g.schools.iloc[0])]+[f'{g[c].mean():.3f}' for c in ['r2','rmse','calibration_intercept','calibration_slope']])
parts.append(table('School-grouped out-of-fold regression calibration, Viet Nam, 2022','tab:calibration',tab(['Outcome','$n$','Schools','$R^2$','RMSE','Intercept','Slope'],rows,'lrrrrrr'),'Final student weights; five-fold GroupKFold keeps each school in one fold; LightGBM handles missing predictors natively and missing targets are excluded. Mathematics results average ten PV-specific diagnostics; anxiety uses the observed index. Intercept/slope come from weighted observed-on-predicted regressions (ideal 0/1), without recalibrating the validation predictions. Intercepts and RMSE use the outcome units. These diagnostics are descriptive, not design-based confidence estimates.'))
mn=pd.read_csv(ROOT/'outputs/results/T3_n.csv').set_index('model')
an=pd.read_csv(ROOT/'outputs/results/T4_n.csv').set_index('model'); rows=[]
for cnt in mn.columns:
    rows.append([cnt]+[f'{int(mn.loc[m,cnt]):,}' for m in ['M1_controls','M2_+climate']]+[f'{int(an.loc[m,cnt]):,}' if cnt in an else '--' for m in ['A1_controls','A2_+climate_relations']]+[f'{int(mn.loc["M4_habits",cnt]):,}' if pd.notna(mn.loc['M4_habits',cnt]) else '--'])
parts.append(table('Analytic samples in distraction and digital-habit figures, PISA 2022','tab:modeln',tab(['System','M1','M2','A1','A2','M4'],rows),'Unweighted complete-case student counts. M1/M2/M4 model mathematics with ten plausible values; A1/A2 model anxiety. M4 includes digital habits in the computer-based systems. Dashes indicate no model for the corresponding outcome. These counts accompany the final-weighted BRR estimates.'))
parts.append(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{FigU_calibration.pdf}'+'\n'+r'\caption{Out-of-fold reliability, Viet Nam, PISA 2022: (a) mathematics regression ($n=6,068$, 178 schools); (b) anxiety regression ($n=5,934$, 178 schools); (c) separate classification of mathematics below Level 2, threshold 420.07 points \citep{oecd2023vol1}. Points are final-weighted means within predicted deciles; the dashed diagonal indicates perfect calibration. Mathematics panels average all ten plausible values. No confidence bands or significance tests are implied.}\label{fig:calibration}\end{figure}'+'\n')
parts.append(f'The separate below-Level-2 classifiers have mean weighted Brier score {math.brier.mean():.3f}, compared with {math.brier_baseline.mean():.3f} for each fold\'s training-prevalence baseline. Brier evaluates probabilities for this binary diagnostic, not the continuous regression. The predictions are retrospective associations using concurrent questionnaire data, not a validated prospective screening tool.\n')
(TEX/'upgrade_appendix.tex').write_text('\n'.join(parts),encoding='utf-8')
print('Generated upgrade appendix; numerical identities, frequencies, convergence and PV coverage verified.')
