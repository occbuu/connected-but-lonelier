"""Verified new references and link-error sensitivity for the peer-review revision."""
import json,time,urllib.request,urllib.parse
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/results_upgrade'
DOIS={
 'meredith1993':'10.1007/BF02294825',
 'byrne1989':'10.1037/0033-2909.105.3.456',
 'vandenberg2000':'10.1177/109442810031002',
 'millsap2004':'10.1207/S15327906MBR3903_4',
 'rutkowski2014':'10.1177/0013164413498257',
 'svetina2017':'10.1177/0022022117717028',
 'svetina2014':'10.1186/s40536-014-0004-5',
 'dienlin2020':'10.31887/DCNS.2020.22.2/tdienlin',
 'odgers2020review':'10.1111/jcpp.13190',
 'odgers2020screen':'10.1146/annurev-devpsych-121318-084815',
 'racine2021':'10.1001/jamapediatrics.2021.2482',
 'loades2020':'10.1016/j.jaac.2020.05.009',
 'samji2022':'10.1111/camh.12501',
 'goodyear2025':'10.1016/j.lanepe.2025.101211',
 'austin2009':'10.1002/sim.3697',
 'austin2015':'10.1002/sim.6607',
 'stuart2010':'10.1214/09-STS313',
 'crump2009':'10.1093/biomet/asn055',
 'li2018':'10.1080/01621459.2016.1260466',
 'aas2021':'10.1016/j.artint.2021.103502',
 'vietnam2024anxiety':'10.3389/fpubh.2024.1232856'
}
target=OUT/'peer_reference_audit.json'
records=json.loads(target.read_text(encoding='utf-8')) if target.exists() else {}
for key,doi in DOIS.items():
    if 'record' in records.get(key,{}): continue
    for attempt in range(3):
        try:
            req=urllib.request.Request('https://api.crossref.org/works/'+urllib.parse.quote(doi,safe=''),headers={'User-Agent':'ManuscriptVerification/1.0'})
            with urllib.request.urlopen(req,timeout=40) as r: record=json.load(r)['message']
            records[key]={'doi':doi,'record':record}; break
        except Exception as e:
            records[key]={'doi':doi,'error':str(e)}; time.sleep(3)
    time.sleep(.5)
target.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
for key,obj in records.items(): print(key,obj.get('record',{}).get('title'),obj.get('error',''),flush=True)
# Link error 3.58 points for mathematics 2012 vs 2022 (NCES technical notes Table 3, sourced to OECD).
# Viet Nam is deliberately excluded: OECD says a reliable performance trend is not established.
trend=pd.read_csv(ROOT/'outputs/results_trend/TR1_indicator_trends.csv')
rows=[]
for cnt in trend.CNT.unique():
    g=trend[trend.CNT==cnt].set_index('CYCLE')
    if not {2012,2022}.issubset(g.index): continue
    diff=g.loc[2022,'MATH']-g.loc[2012,'MATH']
    se=np.hypot(g.loc[2022,'MATH_se'],g.loc[2012,'MATH_se'])
    linked=np.sqrt(se**2+3.58**2)
    valid=cnt!='VNM'
    rows.append(dict(CNT=cnt,status='not comparable: OECD warning' if not valid else 'link-adjusted conditional on comparability',
                     change=diff if valid else np.nan,sampling_pv_se=se if valid else np.nan,
                     link_error=3.58 if valid else np.nan,total_se=linked if valid else np.nan,
                     ci_low=diff-1.96*linked if valid else np.nan,ci_high=diff+1.96*linked if valid else np.nan))
pd.DataFrame(rows).to_csv(OUT/'U7_link_error_math.csv',index=False)
