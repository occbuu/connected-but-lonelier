# Paper 1 — PISA and Student Well-Being

*A decade of digital transition, educational equity and adolescent well-being: Evidence from PISA in Viet Nam and East/Southeast Asia, 2012–2022*

## Project overview

This repository contains the complete research project for Paper 1, which examines digital access and use, educational equity, and adolescent well-being using PISA data from Viet Nam and East/Southeast Asia across the 2012–2022 cycles. It brings together the analysis and reproducibility code, data documentation, processed outputs, manuscript source, figures, tables, and supporting materials.

The manuscript is being prepared for initial submission to *Computers & Education* and has not yet undergone peer review. The current submission version includes 64 cited sources, link-error sensitivity analyses, and an anonymized PDF. Author names, affiliations, email addresses, and CRediT contributions should be completed as required for the intended submission files.

```text
repository/
  code/          Extraction, analysis, robustness checks, tables, and figures
  latex/         main.tex, main.pdf, refs.bib, tables/, figures/
  data/raw/      PISA 2012/2015/2018/2022, policy, WHO, World Bank
  outputs/       Processed data and analysis results
  requirements.txt
```

## Reproducing the analysis

Open a terminal in this paper's directory. Create a dedicated environment if needed:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
foreach ($script in @('01_extract','02_survey_analysis','03_ml_shap','04_figures','05_trend_extract','06_trend_analysis','07_school_policy','08_system_level','10_robustness','09_make_latex_tables','11_replot_figures','12_upgrade_analysis','14_upgrade_tables','15_print_shap')) {
    .\.venv\Scripts\python.exe -X utf8 "code/$script.py"
    if ($LASTEXITCODE -ne 0) { throw "Failed: $script" }
}
```

The scripts locate `data/`, `outputs/`, and `latex/` within this paper's directory and do not require data from Paper 2. See `data/README.md` for instructions on obtaining restricted data from the original sources. This repository includes only aggregate results and a small amount of contextual data.

## Building the manuscript

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File latex/build_pdf.ps1
```

Alternatively, run `latexmk -lualatex main.tex` from `latex/`. The build requires Biber/biblatex-apa and elsarticle. Select LuaLaTeX on Overleaf. Set `\vietnamesetrue` to include the Vietnamese abstract, or `\vietnamesefalse` for the English version. `figures/framework.tex` is the TikZ vector source; its helper can compile a standalone PDF when needed. Figures are exported at 300 dpi as PNG, while the manuscript uses PDF figures.

PISA sources: OECD; the PISA 2012 CSV is available at https://s3.amazonaws.com/udacity-hosted-downloads/ud507/pisa2012.csv.zip. PISA 2015/2018/2022 data are available from https://webfs.oecd.org/. System-level indicators come from the World Bank API and WHO GHO; mobile-phone policies come from https://education-profiles.org.

## Study and analytical overview

- The measurement analysis evaluates partial scalar invariance by freeing B1/B4/B6. The resulting CFI is 0.916, compared with 0.976 for the metric model; scalar invariance is not achieved. The estimated 2022 belonging difference is −0.035 SD, compared with −0.250 SD under the more restrictive specification.
- DFL sensitivity analysis uses five weight-trimming approaches and all 80 BRR replicates independently in each cycle; weight distributions and effective sample sizes (ESS) are reported. Under these specifications, digital access does not make a positive contribution to the increase in loneliness. This decomposition is not evidence that digital access has no causal effect.
- Conditional permutation diagnostics and a reading-comprehension comparison are included. Because reading comprehension is associated with ICTRES, it is not treated as a technology-unrelated negative control.
- Prediction is calibrated by school using all 10 plausible values for mathematics. The Brier score for classifying below-Level-2 mathematics performance is approximately 0.164, compared with a baseline of 0.203. This is an internal evaluation using concurrent variables, not a validated early-warning prediction tool.
- The manuscript covers item-level descriptions, comparisons of pre-pandemic and pandemic-period intervals, curriculum and examination reforms, the H1→H2→H3/H4→H5 hypothesis structure, methodological contributions, and relevant research published in 2024–2025.
- The reference list contains 64 cited sources. DOIs were checked through Crossref. Twenge has a 2026 issue (the existing BibTeX key `twenge2025` is retained). APA 7 formatting uses italics for journal titles and volume numbers and a 0.5-inch hanging indent.

See `UPGRADE_REPORT.md` for analytical limitations and items requiring author confirmation. Do not assign a distribution license without choosing one first. The authors should review the AI declaration and take responsibility for its accuracy before submission.

## Submission preparation

Do not compare Viet Nam's 2022 mathematics trend score with earlier cycles, in accordance with the OECD warning. Cross-cycle SHAP results are included only as supplementary model documentation. Interpret the DFL results as sensitivity analyses subject to overlap limitations, not as evidence that access is unrelated to loneliness. The default PDF is anonymized. Fill in `author_metadata.tex` and `author_contributions.tex`, and set `blindreviewfalse` when preparing a non-anonymized title page.

To regenerate the current figures and tables after the earlier stages, run `17_peer_review_checks` (uses a metadata cache or the Internet), then `14_upgrade_tables`, then `11_replot_figures`. `18_apply_peer_revision` is a legacy, one-time manuscript-conversion helper; do not rerun it if it has already been applied. CSV outputs U1–U6 retain the existing measures; U7 adds link variance.

## Running only the upgrade on existing processed inputs

Place `pisa_trend_2012_2022.parquet` and `pisa22_analysis.parquet` in `outputs/` (generated by extraction), then run `12_upgrade_analysis.py`, `14_upgrade_tables.py` and `15_print_shap.py`. `--calibration-only` resumes only the reading/calibration stage. The full sequence above ensures that older tables are generated **before** the upgrade tables, so the updated DFL table is not overwritten. `13_reference_audit.py` is an optional online metadata audit; saved publisher records accompany the current manuscript.

All computations execute locally. These survey/LightGBM analyses use CPU, with four threads for the upgrade; they do not require a GPU. The code does not publish anything. `16_prepare_repository.py` is the packaging helper and is not part of the analysis sequence.

## Before public release

Complete the author, affiliation, email, and CRediT fields in the appropriate submission files; choose a code license; and review the AI declaration and policy-coding evidence. Before making the repository public, add its confirmed URL (and any Zenodo DOI) to `latex/main.tex` under Data and code availability, then rebuild the manuscript. No placeholder is presented as a working URL. Data-provider terms apply separately. Package SHA256 checksums are in `MANIFEST_SHA256.json`.
