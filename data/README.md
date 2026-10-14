# Data acquisition

The student/school microdata and processed student parquets are not distributed in this repository. Obtain them from the original sources before rerunning extraction; aggregate result CSVs are included for manuscript inspection.

| Input path | Source |
| --- | --- |
| `raw/pisa2012/pisa2012.csv` | OECD PISA 2012 public file converted to labelled CSV; the conversion used here is https://s3.amazonaws.com/udacity-hosted-downloads/ud507/pisa2012.csv.zip (extract CSV). Consult the OECD questionnaire/codebook to verify labels. |
| `raw/pisa2015/PUF_SPSS_COMBINED_CMB_STU_QQQ.zip` | https://webfs.oecd.org/pisa/PUF_SPSS_COMBINED_CMB_STU_QQQ.zip |
| `raw/pisa2018/SPSS_STU_QQQ.zip` | https://webfs.oecd.org/pisa2018/SPSS_STU_QQQ.zip |
| `raw/pisa2022/STU_QQQ_SPSS.zip` and extracted `CY08MSP_STU_QQQ.SAV` | https://webfs.oecd.org/pisa2022/STU_QQQ_SPSS.zip |
| `raw/pisa2022/SCH_QQQ_SPSS.zip`, SAV extracted under `raw/pisa2022/sch/` | https://webfs.oecd.org/pisa2022/SCH_QQQ_SPSS.zip |

Official dataset landing page: https://www.oecd.org/en/data/datasets/pisa-2022-database.html . Earlier cycles and their documentation are linked by OECD. Retain original filenames/case. `01_extract.py` reads the 2022 ZIP and `05_trend_extract.py` reads the extracted 2022 SAV. Raw datasets are several GB; use enough disk/memory. No API credentials are required for these public sources.

Small context snapshots (`raw/worldbank/`, `raw/who/`) and policy coding/evidence (`raw/policy/`) are included; their providers' terms continue to apply. See source URLs and retrieval dates in those files. Population suicide indicators describe all ages, not the PISA student sample. The unused failed OWID download is not included.
