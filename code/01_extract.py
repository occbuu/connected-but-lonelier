"""Extract the analysis variables from the PISA 2022 student file (SPSS) into a compact parquet.

Input : data/raw/pisa2022/STU_QQQ_SPSS.zip (OECD, https://webfs.oecd.org/pisa2022/)
Output: outputs/pisa22_analysis.parquet
"""
import zipfile
from pathlib import Path

import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/pisa2022"
OUT = ROOT / "outputs"

ID = ["CNT", "CNTSCHID", "CNTSTUID", "OECD", "ADMINMODE", "ST004D01T", "AGE", "GRADE", "ISCEDP"]
WEIGHTS = ["W_FSTUWT", "SENWT"] + [f"W_FSTURWT{i}" for i in range(1, 81)]
PVS = [f"PV{i}MATH" for i in range(1, 11)] + [f"PV{i}READ" for i in range(1, 11)]
DIGITAL_ITEMS = ["ST273Q01JA", "ST273Q02JA", "ST273Q03JA", "ST273Q04JA", "ST273Q05JA",  # non-digital climate
                 "ST273Q06JA", "ST273Q07JA",           # distraction by digital resources in maths class
                 "ST322Q01JA", "ST322Q02JA", "ST322Q03JA", "ST322Q04JA", "ST322Q06JA", "ST322Q07JA",
                 "ST309Q02JA",                          # "I get easily distracted"
                 "ST250Q04JA", "ST254Q06JA"]            # own smartphone; number of smartphones at home
INDICES = ["ESCS", "HOMEPOS", "ICTRES", "REPEAT", "SKIPPING", "TARDYSD", "MISSSC",
           "EXERPRAC", "STUDYHMW", "WORKPAY", "WORKHOME",
           "BELONG", "BULLIED", "FEELSAFE", "SCHRISK", "RELATST", "TEACHSUP", "DISCLIM",
           "COGACRCO", "COGACMCO", "FAMSUP",
           "ANXMAT", "MATHEFF", "MATHPERS", "MATHMOT", "GROSAGR",
           "PERSEVAGR", "CURIOAGR", "COOPAGR", "EMPATAGR", "ASSERAGR", "STRESAGR", "EMOCOAGR",
           # ICT familiarity questionnaire (optional; may be missing for some countries)
           "ICTSCH", "ICTHOME", "ICTQUAL", "ICTSUBJ", "ICTENQ", "ICTFEED", "ICTOUT",
           "ICTWKDY", "ICTWKEND", "ICTREG", "ICTINFO", "ICTDISTR", "ICTEFFIC",
           # Well-being questionnaire (optional)
           "LIFESAT", "PSYCHSYM", "SOCCON", "EXPWB", "BODYIMA"]
COLS = ID + WEIGHTS + PVS + DIGITAL_ITEMS + INDICES


def main():
    zf = zipfile.ZipFile(RAW / "STU_QQQ_SPSS.zip")
    sav = [n for n in zf.namelist() if n.lower().endswith(".sav")][0]
    target = RAW / Path(sav).name
    if not target.exists():
        print("unzipping", sav)
        with zf.open(sav) as src, open(target, "wb") as dst:
            while chunk := src.read(1 << 24):
                dst.write(chunk)
    _, meta = pyreadstat.read_sav(str(target), metadataonly=True)
    cols = [c for c in COLS if c in meta.column_names]
    missing = sorted(set(COLS) - set(cols))
    print("not in file:", missing)
    df, meta = pyreadstat.read_sav(str(target), usecols=cols, apply_value_formats=False,
                                   user_missing=False)
    df.to_parquet(OUT / "pisa22_analysis.parquet", index=False)
    labels = pd.DataFrame({"var": cols, "label": [meta.column_names_to_labels.get(c, "") for c in cols]})
    labels.to_csv(OUT / "variable_labels.csv", index=False, encoding="utf-8-sig")
    print(df.shape)
    print(df.CNT.value_counts().loc[lambda s: s.index.isin(["VNM", "SGP", "JPN", "KOR", "THA", "MYS"])])


if __name__ == "__main__":
    main()
