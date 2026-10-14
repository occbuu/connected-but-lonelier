"""Apply the review revision once to the local manuscript; keep the pre-review version."""
from pathlib import Path
import re,json,html,shutil
ROOT=Path(__file__).resolve().parents[1]
TEX=ROOT/'latex' if (ROOT/'latex/main.tex').exists() else ROOT/'Paper1_PISA2022'
backup=ROOT/'archive/pre_peer_review_2026-10-14'; backup.mkdir(parents=True,exist_ok=True)
for name in ['main.tex','main.pdf','refs.bib','cover_letter.md','highlights.txt','upgrade_appendix.tex']:
    if (TEX/name).exists() and not (backup/name).exists(): shutil.copy2(TEX/name,backup/name)
s=(TEX/'main.tex').read_text(encoding='utf-8')
if 'peer_review_literature' in s: raise SystemExit('Revision already applied; edit normally rather than applying twice.')
title='A decade of digital transition, educational equity and adolescent well-being: Evidence from PISA in Viet Nam and East/Southeast Asia, 2012--2022'
s=re.sub(r'\\title\{[^\n]*\}',lambda _:r'\title{'+title+'}',s,count=1)
s=s.replace(r'\newcommand{\tbd}[1]{\textcolor{red}{[#1]}}',r'\newif\ifblindreview'+'\n'+r'\blindreviewtrue')
start=s.index(r'\author[a]'); end=s.index(r'\begin{abstract}',start)
s=s[:start]+r'''\ifblindreview
\author{Anonymous manuscript for peer review}
\else
\input{author_metadata}
\fi

'''+s[end:]
(TEX/'author_metadata.tex').write_text(r'''% Author-completed title-page template; not used in anonymous-review mode.
\author[a]{Author name to be supplied\corref{cor1}}
\cortext[cor1]{Corresponding author email to be supplied}
\affiliation[a]{organization={Institution to be supplied},country={Viet Nam}}
''',encoding='utf-8')
s=s.replace(r'\tbd{Author 1}: Conceptualization, Methodology, Formal analysis, Writing -- original draft. \tbd{Author 2}: Validation, Writing -- review \& editing.',r'\ifblindreview Author contributions are withheld for anonymous review.\else\input{author_contributions}\fi')
(TEX/'author_contributions.tex').write_text('% Complete with actual author names and approved CRediT roles before submission.\nAuthor contribution information to be supplied.\n',encoding='utf-8')
s=s.replace('H1','RQ1').replace(r'\paragraph{Hypotheses}',r'\paragraph{Research question and hypotheses}')
s=s.replace(r'\textbf{RQ1}: Between 2012 and 2022, indicators of social exclusion and mathematics anxiety among Vietnamese students increased.',r'\textbf{RQ1 (descriptive)}: How did responses to school-connectedness items and measurement-tested mathematics anxiety differ between Vietnamese PISA cohorts from 2012 to 2022?')
s=s.replace('RQ1 is supported.', 'RQ1 documents changes in reported item responses and in the anxiety model; it does not establish a general deterioration in mental health.')
s=s.replace('The adjusted home-computer mathematics association increased to 23 points.', 'In 2022, the adjusted home-computer mathematics association was approximately 23 points; no cross-cycle growth in this association is inferred.')
s=s.replace('Chênh lệch điểm toán gắn với máy tính tại nhà tăng lên khoảng 23 điểm.', 'Năm 2022, chênh lệch điểm toán gắn với máy tính tại nhà khoảng 23 điểm; nghiên cứu không suy ra mức tăng của chênh lệch này qua chu kỳ.')
s=s.replace('Although measured digital access does not account for the well-being trend in this decomposition, its association with achievement has strengthened, raising an equity concern. This is an ancillary H2 finding; H5 concerns school phone policies and self-regulation rather than access inequality.', 'The 2022 within-cycle access--achievement association raises an equity question alongside the uncertain DFL sensitivity results. It does not establish that inequality increased over time. H5 concerns school phone policies and self-regulation.')
start=s.index(r'\paragraph{Access and achievement inequality}'); end=s.index(r'\subsection{Digital distraction',start)
old=s[start:end]
shapfigure=re.search(r'\\begin\{figure\}[\s\S]*?\\end\{figure\}',old).group()
shapfigure=shapfigure.replace('share of SHAP importance of harmonised predictors of mathematics, by cycle','separately normalised SHAP summaries for harmonised predictors of mathematics, by cycle')
shapfigure=shapfigure.replace('SHAP shares describe prediction, not causal importance.', 'Each model has its own feature covariance, availability and background distribution. These summaries are not a common cross-cycle importance scale and cannot demonstrate increasing digital inequality.')
s=s[:start]+r'''\paragraph{Access and achievement within the 2022 cohort} In Viet Nam in 2022, home-computer access was associated with 23.4 higher mathematics points after background adjustment. This association may reflect resources, learning opportunities and unmeasured selection; it is not the effect of providing a computer. Separately fitted cross-cycle SHAP summaries are retained only as exploratory model documentation in Figure~\ref{fig:shapcycle}. No change in an inequality parameter is estimated from them.

'''+s[end:]
insert=r'\subsection{Additional measurement, sensitivity and calibration checks}'
s=s.replace(insert,r'\subsection{Exploratory model documentation}'+ '\n'+shapfigure+'\n'+insert)
s=s.replace('An ecological mechanism remains a hypothesis to investigate. Meanwhile, the increasing adjusted home-computer achievement association raises an equity concern; repeated cross-sections cannot establish that devices caused the gap.', 'An ecological mechanism remains an untested possibility alongside examination pressure, family change and willingness to disclose distress. The 2022 home-computer achievement association motivates an equity question; its size must not be interpreted as growth in an inequality effect across non-comparable score scales.')
s=s.replace('its strengthening achievement association nevertheless raises an equity concern.', 'the 2022 within-cycle achievement association raises an equity question without establishing a temporal increase.')
s=s.replace('Belonging remains non-invariant, and weighted-moment continuous-item CFA is an approximation to ordinal responses.', 'Belonging remains non-invariant, and weighted-moment continuous-item CFA is an approximation to ordinal responses. The near-zero partial-scalar contrast neither proves stable true belonging nor proves that loneliness is a measurement artifact: both the full- and partial-scalar mean models fail. Changed negative-item response style is one plausible explanation, not an identified cause.')
s=s.replace('Mathematics trends omit link error; Viet Nam\'s 2018 performance values are unavailable.', 'Viet Nam\'s 2022 performance scores are not reliably linked to its earlier cycles, and its 2018 performance values are unavailable; no Vietnamese achievement trend is inferred. For eligible comparison systems, the separate mathematics-difference sensitivity adds squared link error to sampling/PV variance (Table~\\ref{tab:linkerror}).')
s=s.replace('Its strengthening achievement association', 'Its within-cycle achievement association')
s=s.replace('The contribution combines multi-cycle harmonisation with measurement-invariance tests, sequential DFL decomposition, and LightGBM/SHAP tracking of predictive importance across cycles', 'The contribution combines multi-cycle questionnaire harmonisation with measurement-invariance tests, sequential DFL sensitivity, and within-cycle LightGBM/SHAP model documentation')
s=s.replace('The same approach was applied by cycle with the harmonised features to track how the importance of predictors shifted over the decade.', 'Separate cycle models provide exploratory prediction summaries. Their SHAP percentages have model-specific denominators and background distributions; they are not harmonised importance estimates across cycles.')
s=s.replace('Viet Nam remained paper-based in all four cycles (Appendix Table~\\ref{tab:mode}), so Vietnamese trends are not confounded by changes in mode.', 'Viet Nam remained paper-based in all four cycles (Appendix Table~\\ref{tab:mode}); this removes an administration-mode transition but does not establish score comparability. OECD explicitly warns that its 2022 performance results are not comparable with previous cycles \\citep{oecd2023vnm}. We therefore analyse questionnaire responses over time and achievement associations within cycles; no Vietnamese mathematics trajectory is drawn.')
s=s.replace('Mathematics trends are descriptive (link error not added)', 'Mathematics means are separate cycle descriptions; Viet Nam 2022 is not comparable with earlier cycles')
# Disconnect all language assigning the unexplained component to a mechanism.
s=s.replace('These results do not support the proposed compositional explanation, but nonsignificance does not establish equivalence', 'These are overlap-limited sensitivity results, and imprecise access components cannot demonstrate absence of an association; nonsignificance does not establish equivalence')
s=s.replace('Sensitivity to five weighting choices strengthens this descriptive conclusion, without ruling out intensity of use, content, reverse causation or unobserved confounding.', 'Five clipping choices document sensitivity rather than solve identification. Stage-2 ESS falls from 6,068 to 367 without clipping; 17.5\\% of estimated 2022 probabilities lie outside .05/.95. Low effective information can mask contributions, and clipping trades variance for bias without recovering unsupported population contrasts. Intensity, content, reverse causation and unobserved confounding remain unresolved.')
s=s.replace('Principal-reported policies and the small number of systems further limit inference.', 'Principal-reported restrictions do not measure compliance or enforcement; school management, disciplinary culture and teacher quality are unmeasured possible confounders. IPW balances only included covariates and is not a quasi-experiment. The small number of systems further limits inference.')
s=s.replace(r'\section{Results}',r'\input{peer_review_literature}'+'\n'+r'\section{Results}')
literature=r'''\paragraph{Measurement and response processes} Latent mean comparison requires comparable measurement structure, not merely identical wording \citep{meredith1993,vandenberg2000}. Partial invariance can be useful when a defensible invariant subset anchors comparisons \citep{byrne1989}; freeing problematic parameters does not by itself validate a model. International-survey fit cutoffs depend on group structure and model conditions \citep{rutkowski2014,svetina2017}. Ordinal-invariance and differential-item-functioning approaches offer useful extensions beyond our continuous-moment approximation \citep{millsap2004,svetina2014}. We consequently distinguish directly observed response proportions from claims about an invariant psychological construct.

\paragraph{Overlap, selection and model interpretation} Propensity methods require balance and adequate support as well as assumptions about unmeasured selection \citep{stuart2010,austin2009,austin2015}. Severe distributional separation can make population reweighting unstable; restricting support or using overlap weights defines a different target population \citep{crump2009,li2018}. We do not relabel winsorised DFL ratios as an identified overlap-population effect. SHAP attributions depend on feature dependence and how the background distribution is specified \citep{aas2021,janzing2020}; within-model attribution therefore cannot be promoted to a standardised between-cycle inequality measure.

\paragraph{Digital use, pandemic context and Vietnamese schooling} Reviews distinguish access and duration from use patterns, individual vulnerability and social context \citep{dienlin2020,odgers2020review,odgers2020screen}. Pandemic-era syntheses document distress and isolation concerns but do not identify the source of our PISA cohort differences \citep{racine2021,loades2020,samji2022}. A mixed-method study of Hanoi high-school students reports academic, social and family concerns surrounding anxiety \citep{vietnam2024anxiety}; its local sample and GAD-7 outcome are not interchangeable with national PISA mathematics anxiety. The SMART Schools observational study adds recent evidence on restrictive policies, phone use and well-being \citep{goodyear2025}; policies on paper, enforcement and students' overall use are distinct exposures. These sources motivate alternatives to a uniquely digital ecological explanation, without assigning those alternatives causal effects in our data.

'''
(TEX/'peer_review_literature.tex').write_text(literature,encoding='utf-8')
(TEX/'main.tex').write_text(s,encoding='utf-8')
framework=(TEX/'figures/framework.tex').read_text(encoding='utf-8').replace('H1: cohort trend (ecological pathway)','RQ1: descriptive cohort comparison')
(TEX/'figures/framework.tex').write_text(framework,encoding='utf-8')
vnm=TEX/'tables/tab_vnm_trend.tex'
t=vnm.read_text(encoding='utf-8').replace('Mathematics mean &','Mathematics mean (not a linked trend) &')
vnm.write_text(t,encoding='utf-8')
# Bibliographic records: exact DOI retrieval; sentence case titles with protected proper nouns.
records=json.loads((ROOT/'outputs/results_upgrade/peer_reference_audit.json').read_text(encoding='utf-8'))
assert all('record' in x for x in records.values())
def sentence(text):
    text=html.unescape(text).replace('\xa0',' ').strip().rstrip('.').lower()
    text=re.sub(r'(^|[?:]\s+)([a-z])',lambda m:m[1]+m[2].upper(),text)
    for low,proper in {'covid-19':'COVID-19','iptw':'IPTW','pisa':'PISA','shap':'SHAP','vietnam':'Vietnam','hanoi':'Hanoi','smart schools':'SMART Schools'}.items():
        text=re.sub(re.escape(low),lambda _:'{'+proper+'}',text,flags=re.I)
    return text.replace('&',r'\&')
bib=(TEX/'refs.bib').read_text(encoding='utf-8').replace('ROSENBAUM, PAUL R. and RUBIN, DONALD B.','Rosenbaum, Paul R. and Rubin, Donald B.')
for key,obj in records.items():
    r=obj['record']; authors=[]
    for a in r['author']:
        family=a.get('family',''); given=a.get('given','')
        if family.isupper(): family=family.title()
        if given.isupper() and len(given)>5: given=given.title()
        authors.append(family+', '+given)
    year=r.get('published-print',r['published'])['date-parts'][0][0]
    title=sentence(r['title'][0]+(': '+r['subtitle'][0] if r.get('subtitle') else ''))
    journal=html.unescape(r['container-title'][0]).replace('&',r'\&')
    if journal.islower(): journal=journal.title()
    fields={'author':' and '.join(authors),'title':title,'journal':journal,'year':str(year),'doi':obj['doi']}
    for field in ['volume','issue','page','article-number']:
        if r.get(field): fields[{'issue':'number','page':'pages','article-number':'eid'}.get(field,field)]=str(r[field]).replace('-','--') if field=='page' else str(r[field])
    bib+='\n@article{'+key+',\n'+',\n'.join('  '+k+' = {'+v+'}' for k,v in fields.items())+'\n}\n'
bib+=r'''
@inproceedings{janzing2020,
 author = {Janzing, Dominik and Minorics, Lenon and Blöbaum, Patrick}, year = {2020},
 title = {Feature relevance quantification in explainable {AI}: A causal problem},
 booktitle = {Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics}, volume = {108}, pages = {2907--2916}, publisher = {PMLR}, url = {https://proceedings.mlr.press/v108/janzing20a.html}
}
@misc{oecd2023vnm,
 author = {{OECD}}, year = {2023}, title = {{PISA} 2022 results (Volume {I} and {II}): Country notes---Viet Nam}, url = {https://www.oecd.org/en/publications/pisa-2022-results-volume-i-and-ii-country-notes_ed6fbcc5-en/viet-nam_a727c3a8-en.html}
}
@misc{nces2023technical,
 author = {{National Center for Education Statistics}}, year = {2023}, title = {{PISA} 2022 U.S. results: Technical notes}, url = {https://nces.ed.gov/surveys/pisa/pisa2022/technical-notes/index.asp}
}
'''
bib=bib.replace('Thông tư 32/2018/{TT-BGDĐT} ban hành chương trình giáo dục phổ thông}', 'Thông tư 32/2018/{TT-BGDĐT} ban hành chương trình giáo dục phổ thông [Circular 32/2018/{TT-BGDĐT} promulgating the general education curriculum]}')
bib=bib.replace('trường phổ thông có nhiều cấp học}', 'trường phổ thông có nhiều cấp học [Circular 32/2020/{TT-BGDĐT} promulgating the charter of lower-secondary, upper-secondary and multi-level schools]}')
bib=bib.replace('Đề thi chung, làm thế nào để bảo đảm phân hóa trình độ thí sinh?}', 'Đề thi chung, làm thế nào để bảo đảm phân hóa trình độ thí sinh? [A common examination: How can differences in candidate proficiency be distinguished?]}')
bib=bib.replace('{{UNESCO GEM Report}}','{{UNESCO Global Education Monitoring Report}}')
# Continuously updated datasets: do not invent publication years from internal key names.
for key in ['peer2024','whogho2025','wdi2024']:
    pattern=r'@misc\{'+key+r',[\s\S]*?\n\}'
    def clean(m):
        block=re.sub(r'year\s*=\s*\{\d+\},?','',m[0])
        block=re.sub(r',?\s*note\s*=\s*\{[^}]*\}','',block)
        if 'urldate' not in block: block=block[:-1]+',\n  urldate = {2026-10-14}\n}'
        return block
    bib=re.sub(pattern,clean,bib)
(TEX/'refs.bib').write_text(bib,encoding='utf-8')
# Submission-support texts must agree with the revised claims.
(TEX/'highlights.txt').write_text('Home internet access among Vietnamese PISA respondents rose from 29% to 90%.\nMathematics anxiety passes scalar invariance; belonging remains non-invariant.\nOverlap-limited DFL sensitivity cannot establish absence of an access association.\nSchool policy and achievement analyses are observational within-cycle comparisons.\nSeparate SHAP summaries do not identify changes in inequality across cycles.\n',encoding='utf-8')
(TEX/'cover_letter.md').write_text('Dear Editors,\n\nPlease consider our manuscript, "'+title.replace('--','–')+'". It combines four PISA questionnaire cohorts with 2022 school-policy data and public context indicators. The focus is measurement-tested mathematics anxiety, item-response patterns and within-cycle educational associations. Belonging fails scalar and targeted partial-scalar invariance; the article does not claim a validated latent decline. DFL is presented as overlap-limited sensitivity, and IPW/SHAP are not interpreted as causal designs. OECD\'s warning on Vietnamese performance comparability precludes an achievement trend claim.\n\nAuthor identities, contribution roles, public code URL, exclusivity/originality declarations and any conference disclosure must be completed and confirmed by the authors before submission. This is a working cover-letter draft, not an assertion that those administrative conditions have already been verified.\n\nSincerely,\nCorresponding author to be supplied\n',encoding='utf-8')
print('Reframed manuscript and',len(records),'new DOI references; bibliography total',len(re.findall(r'@\w+\{',bib)))
