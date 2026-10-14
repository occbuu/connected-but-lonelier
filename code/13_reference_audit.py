"""Save publisher/Crossref records for manual bibliographic verification."""
import concurrent.futures, json, re, urllib.request, urllib.parse, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
bib = (ROOT/'latex/refs.bib').read_text(encoding='utf-8')
records = {}
for block in re.split(r'(?=@\w+\{)', bib):
    m = re.match(r'@(\w+)\{([^,]+),', block)
    if not m: continue
    d = re.search(r'doi\s*=\s*\{([^}]+)', block)
    if d: records[m[2]] = d[1]
records.update(wang2024='10.1016/j.compedu.2024.105003', martin2025='10.1007/s11423-025-10550-6',
    benjamini1995='10.1111/j.2517-6161.1995.tb02031.x', kenny2015='10.1177/0049124114543236',
    rosenbaum1983='10.1093/biomet/70.1.41', sweller1988='10.1207/s15516709cog1202_4',
    zimmerman2002='10.1207/s15430421tip4102_2', oecd2023vol2='10.1787/a97db61c-en',
    oecd2023vol1='10.1787/53f23881-en', oecd2024screen='10.1787/7c225af4-en')
def fetch(pair):
    key, doi = pair
    try:
        req = urllib.request.Request('https://api.crossref.org/works/'+urllib.parse.quote(doi,safe=''), headers={'User-Agent':'PaperBibliographyAudit/1.0'})
        with urllib.request.urlopen(req, timeout=45) as r: data = json.load(r)['message']
        return key, {'doi':doi, 'record':data}
    except Exception as e: return key, {'doi':doi, 'error':str(e)}
target=ROOT/'outputs/results_upgrade/reference_audit.json'
result=json.loads(target.read_text(encoding='utf-8')) if target.exists() else {}
for key, doi in records.items():
    if 'record' in result.get(key,{}): continue
    time.sleep(2)
    _, result[key]=fetch((key,doi))
target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
for key, obj in result.items():
    r=obj.get('record',{})
    print(key, r.get('title'), r.get('published-print',r.get('published')),r.get('volume'),r.get('page'),obj.get('error',''))
