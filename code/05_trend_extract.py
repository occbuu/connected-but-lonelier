"""Harmonise PISA 2012, 2015, 2018 and 2022 student data (10 East/Southeast Asian systems) into one
long file for the 2012-2022 trend analysis.

Sources (public):
  2012  OECD INT_STU12_DEC03 (CSV conversion with text labels, data/raw/pisa2012/pisa2012.csv)
  2015  OECD CY6_MS_CMB_STU_QQQ.sav   (https://webfs.oecd.org/pisa/PUF_SPSS_COMBINED_CMB_STU_QQQ.zip)
  2018  OECD CY07_MSU_STU_QQQ.sav     (https://webfs.oecd.org/pisa2018/SPSS_STU_QQQ.zip)
  2022  OECD CY08MSP_STU_QQQ.SAV      (https://webfs.oecd.org/pisa2022/STU_QQQ_SPSS.zip)

Harmonised variables (same wording and response scale across cycles unless noted):
  MATH PVs, final + 80 replicate weights, FEMALE, ESCS (cycle-specific scale), GRADE, REPEAT,
  BOOKS (6 categories), COMPUTER (computer for school work at home), INTERNET (2022 wording excludes
  smartphones), SMARTPHONE3 (3+ internet phones at home; 2015-2022),
  BELONG_RAW (mean of the six ST034/ST87 items, 1-4, higher = more belonging; all cycles),
  BELONG_AGREE ("I feel like I belong at school", agree/strongly agree),
  LONELY_AGREE ("I feel lonely at school"),
  ANX_RAW (five common maths-anxiety items, 1-4, higher = more anxious; 2012 & 2022),
  TENSE_AGREE ("I get very tense when I have to do maths homework"; 2012 & 2022),
  DISC_RAW (five common maths-class disciplinary items, 1-4, higher = better climate; 2012 & 2022),
  NOISE_MOST ("noise and disorder" in most/every maths lesson; 2012 & 2022).
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "outputs"
SYSTEMS = ["VNM", "SGP", "JPN", "KOR", "TAP", "HKG", "MAC", "THA", "MYS", "IDN"]
NAME2012 = {"Vietnam": "VNM", "Singapore": "SGP", "Japan": "JPN", "Korea": "KOR", "Chinese Taipei": "TAP",
            "Hong Kong-China": "HKG", "Macao-China": "MAC", "Thailand": "THA", "Malaysia": "MYS",
            "Indonesia": "IDN"}
RW = [f"W_FSTURWT{i}" for i in range(1, 81)]
PV = [f"PV{i}MATH" for i in range(1, 11)]


def unzip_sav(zpath):
    zf = zipfile.ZipFile(zpath)
    sav = max((i for i in zf.infolist() if i.filename.lower().endswith(".sav")),
              key=lambda i: i.file_size).filename  # main file (2015 zip also has a QQ2 file)
    target = zpath.parent / Path(sav).name
    if not target.exists():
        with zf.open(sav) as src, open(target, "wb") as dst:
            while chunk := src.read(1 << 24):
                dst.write(chunk)
    return target


def agree4(s, positive=True):
    """1 SA .. 4 SD codes -> agreement 1..4 (4 = strongly agree) or reversed."""
    a = 5 - s.where(s.between(1, 4))
    return a if positive else 5 - a


def common(d):
    """Derived variables shared by 2015/2018/2022 (numeric codes)."""
    o = pd.DataFrame(index=d.index)
    o["FEMALE"] = (d["ST004D01T"] == 1).astype(float).where(d["ST004D01T"].isin([1, 2]))
    items = [f"ST034Q0{i}TA" for i in range(1, 7)]
    pos = {2, 3, 5}
    b = pd.concat([agree4(d[c], positive=(i + 1) in pos) for i, c in enumerate(items)], axis=1)
    o["BELONG_RAW"] = b.mean(axis=1).where(b.notna().sum(axis=1) >= 4)
    for i in range(6):  # item-level scores, 1-4, higher = more belonging (for invariance tests)
        o[f"B{i + 1}"] = b.iloc[:, i]
    o["ADMIN"] = d["ADMINMODE"] if "ADMINMODE" in d else np.nan  # 1 paper, 2 computer
    o["BELONG_AGREE"] = (d["ST034Q03TA"] <= 2).astype(float).where(d["ST034Q03TA"].between(1, 4))
    o["LONELY_AGREE"] = (d["ST034Q06TA"] <= 2).astype(float).where(d["ST034Q06TA"].between(1, 4))
    return o


def load_2012():
    cols = (["CNT", "SCHOOLID", "ST04Q01", "ST26Q04", "ST26Q06", "ST28Q01", "ESCS", "REPEAT", "GRADE",
             "W_FSTUWT"] + [f"W_FSTR{i}" for i in range(1, 81)] + [f"PV{i}MATH" for i in range(1, 6)]
            + [f"ST42Q{q:02d}" for q in (1, 3, 5, 8, 10)] + [f"ST81Q0{i}" for i in range(1, 6)]
            + [f"ST87Q0{i}" for i in range(1, 7)])
    parts = []
    for ch in pd.read_csv(RAW / "pisa2012/pisa2012.csv", usecols=cols, chunksize=100000,
                          low_memory=False, encoding="latin1"):
        parts.append(ch[ch.CNT.isin(NAME2012)])
    d = pd.concat(parts, ignore_index=True)
    ag = {"Strongly agree": 4, "Agree": 3, "Disagree": 2, "Strongly disagree": 1}
    fr = {"Every Lesson": 1, "Most Lessons": 2, "Some Lessons": 3, "Never or Hardly Ever": 4}
    o = pd.DataFrame({"CYCLE": 2012, "CNT": d.CNT.map(NAME2012),
                      "SCHID": "2012-" + d.CNT.map(NAME2012) + "-" + d.SCHOOLID.astype(str)})
    o["FEMALE"] = d.ST04Q01.map({"Female": 1.0, "Male": 0.0})
    yn = {"Yes": 1.0, "No": 0.0}
    o["COMPUTER"] = d.ST26Q04.map(yn)
    o["INTERNET"] = d.ST26Q06.map(yn)
    o["SMARTPHONE3"] = np.nan
    books = {"0-10 books": 1, "11-25 books": 2, "26-100 books": 3, "101-200 books": 4,
             "201-500 books": 5, "More than 500 books": 6}
    o["BOOKS"] = d.ST28Q01.astype(str).str.strip().map(books)
    o["ESCS"] = d.ESCS
    o["GRADE"] = d.GRADE
    o["REPEAT"] = d.REPEAT.map({"Did not repeat a <grade>": 0.0, "Repeated a <grade>": 1.0})
    a = pd.concat([d[f"ST42Q{q:02d}"].map(ag) for q in (1, 3, 5, 8, 10)], axis=1)
    o["ANX_RAW"] = a.mean(axis=1).where(a.notna().sum(axis=1) >= 3)
    o["TENSE_AGREE"] = (d.ST42Q03.map(ag) >= 3).astype(float).where(d.ST42Q03.map(ag).notna())
    c = pd.concat([d[f"ST81Q0{i}"].map(fr) for i in range(1, 6)], axis=1)
    o["DISC_RAW"] = c.mean(axis=1).where(c.notna().sum(axis=1) >= 3)
    o["NOISE_MOST"] = (d.ST81Q02.map(fr) <= 2).astype(float).where(d.ST81Q02.map(fr).notna())
    pos = {2, 3, 5}
    b = pd.concat([d[f"ST87Q0{i}"].map(ag) if i in pos else 5 - d[f"ST87Q0{i}"].map(ag)
                   for i in range(1, 7)], axis=1)
    o["BELONG_RAW"] = b.mean(axis=1).where(b.notna().sum(axis=1) >= 4)
    for i in range(6):
        o[f"B{i + 1}"] = b.iloc[:, i].to_numpy()
    o["ADMIN"] = 1.0  # PISA 2012 main survey was paper-based
    for k, q in enumerate((1, 3, 5, 8, 10)):
        o[f"A{k + 1}"] = a.iloc[:, k].to_numpy()
    o["BELONG_AGREE"] = (d.ST87Q03.map(ag) >= 3).astype(float).where(d.ST87Q03.map(ag).notna())
    o["LONELY_AGREE"] = (d.ST87Q06.map(ag) >= 3).astype(float).where(d.ST87Q06.map(ag).notna())
    o["W_FSTUWT"] = d.W_FSTUWT
    for i in range(1, 81):
        o[f"W_FSTURWT{i}"] = d[f"W_FSTR{i}"]
    for i in range(1, 11):  # 2012 has 5 PVs; PV6-10 left missing
        o[f"PV{i}MATH"] = d[f"PV{i}MATH"] if i <= 5 else np.nan
    return o


def load_sav(path, cycle, extra):
    base = (["CNT", "CNTSCHID", "ADMINMODE", "ST004D01T", "ESCS", "REPEAT", "GRADE", "W_FSTUWT"] + RW + PV
            + [f"ST034Q0{i}TA" for i in range(1, 7)])
    _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
    use = [c for c in base + extra if c in meta.column_names]
    missing = sorted(set(base + extra) - set(use))
    if missing:
        print(cycle, "missing:", missing)
    d, _ = pyreadstat.read_sav(str(path), usecols=use, apply_value_formats=False)
    d = d[d.CNT.isin(SYSTEMS)].reset_index(drop=True)
    o = common(d)
    o.insert(0, "CYCLE", cycle)
    o.insert(1, "CNT", d.CNT)
    o.insert(2, "SCHID", f"{cycle}-" + d.CNT + "-" + d.CNTSCHID.astype("Int64").astype(str))
    for c in ["ESCS", "GRADE", "REPEAT", "W_FSTUWT"] + RW + PV:
        o[c] = d[c] if c in d else np.nan
    return o, d


def load_2015_2018(cycle, path):
    o, d = load_sav(path, cycle, ["ST011Q04TA", "ST011Q06TA", "ST012Q05NA", "ST013Q01TA"])
    o["COMPUTER"] = (d.ST011Q04TA == 1).astype(float).where(d.ST011Q04TA.isin([1, 2]))
    o["INTERNET"] = (d.ST011Q06TA == 1).astype(float).where(d.ST011Q06TA.isin([1, 2]))
    o["SMARTPHONE3"] = (d.ST012Q05NA >= 4).astype(float).where(d.ST012Q05NA.between(1, 4))
    o["BOOKS"] = d.ST013Q01TA.where(d.ST013Q01TA.between(1, 6))
    for c in ["ANX_RAW", "TENSE_AGREE", "DISC_RAW", "NOISE_MOST", "A1", "A2", "A3", "A4", "A5"]:
        o[c] = np.nan
    return o


def load_2022(path):
    anx = [f"ST292Q0{i}JA" for i in range(1, 6)]
    disc = [f"ST273Q0{i}JA" for i in range(1, 6)]
    o, d = load_sav(path, 2022, ["ST250Q02JA", "ST250Q05JA", "ST254Q06JA", "ST255Q01JA"] + anx + disc)
    o["COMPUTER"] = (d.ST250Q02JA == 1).astype(float).where(d.ST250Q02JA.isin([1, 2]))
    o["INTERNET"] = (d.ST250Q05JA == 1).astype(float).where(d.ST250Q05JA.isin([1, 2]))
    # ST254 (2022): 1 none, 2 "1 or 2", 3 "3-5", 4 "more than 5", 5 "I don't know" -> 3+ = codes 3-4
    o["SMARTPHONE3"] = (d.ST254Q06JA >= 3).astype(float).where(d.ST254Q06JA.between(1, 4))
    # ST255: 1 none, 2 1-10, 3 11-25, ... 7 >500  -> 6 harmonised categories (0-10 merged)
    o["BOOKS"] = d.ST255Q01JA.where(d.ST255Q01JA.between(1, 7)).clip(lower=2) - 1
    a = pd.concat([agree4(d[c]) for c in anx], axis=1)
    o["ANX_RAW"] = a.mean(axis=1).where(a.notna().sum(axis=1) >= 3)
    for k in range(5):
        o[f"A{k + 1}"] = a.iloc[:, k].to_numpy()
    o["TENSE_AGREE"] = (d.ST292Q02JA <= 2).astype(float).where(d.ST292Q02JA.between(1, 4))
    c = pd.concat([d[x].where(d[x].between(1, 4)) for x in disc], axis=1)
    o["DISC_RAW"] = c.mean(axis=1).where(c.notna().sum(axis=1) >= 3)
    o["NOISE_MOST"] = (d.ST273Q02JA <= 2).astype(float).where(d.ST273Q02JA.between(1, 4))
    return o


def main():
    frames = [load_2012()]
    print("2012", frames[-1].groupby("CNT").size().to_dict())
    frames.append(load_2015_2018(2015, unzip_sav(RAW / "pisa2015/PUF_SPSS_COMBINED_CMB_STU_QQQ.zip")))
    print("2015", frames[-1].groupby("CNT").size().to_dict())
    frames.append(load_2015_2018(2018, unzip_sav(RAW / "pisa2018/SPSS_STU_QQQ.zip")))
    print("2018", frames[-1].groupby("CNT").size().to_dict())
    frames.append(load_2022(RAW / "pisa2022/CY08MSP_STU_QQQ.SAV"))
    print("2022", frames[-1].groupby("CNT").size().to_dict())
    cols = list(dict.fromkeys(c for f in frames for c in f.columns))
    frames = [f.reindex(columns=cols) for f in frames]
    df = pd.concat([f[cols] for f in frames], ignore_index=True)
    df.to_parquet(OUT / "pisa_trend_2012_2022.parquet", index=False)
    print(df.shape)
    print(df.groupby(["CNT", "CYCLE"])[["BELONG_RAW", "ANX_RAW", "DISC_RAW", "COMPUTER", "SMARTPHONE3"]]
          .apply(lambda g: g.notna().mean()).round(2).loc["VNM"])


if __name__ == "__main__":
    main()
