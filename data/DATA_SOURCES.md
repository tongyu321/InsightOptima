# Public data sources used by InsightOptima

All demo corpora shipped with this app are **publicly documented** datasets.
Do not replace them with private or fabricated “case study” text if you need
portfolio credibility.

## Mixed-methods portfolio pair

| Strand | Dataset | Local file |
| --- | --- | --- |
| **Quantitative** | UCI Drug Review Dataset (Drugs.com) | `data/case_study_drug_reviews.csv` |
| **Qualitative** | Zenodo PubPeer open-ended / coded answers | `data/case_study_pubpeer_qual.csv` |

Both are CC BY 4.0 with DOIs. The Case Study page loads either strand into the
same analysis workspace.

## 1. Quantitative case — Public health (Drugs.com / UCI)

| Field | Value |
| --- | --- |
| Title | Drug Review Dataset (Drugs.com) |
| Strand | Quantitative (rated reviews, large *n*) |
| Domain | Public health / patient experience |
| Publisher | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com) |
| DOI | [10.24432/C5SK5S](https://doi.org/10.24432/C5SK5S) |
| Content origin | Patient reviews originally published on [drugs.com](https://www.drugs.com/) |
| License | CC BY 4.0 (cite authors) |
| Local file | `data/case_study_drug_reviews.csv` |
| Provenance file | `data/case_study_drug_reviews.SOURCE.json` |
| Rebuild | `python scripts/prepare_case_study_dataset.py` |

**Citation:** Kallumadi, Surya and Gräßer, Felix. (2018). Drug Review Dataset (Drugs.com). UCI Machine Learning Repository. https://doi.org/10.24432/C5SK5S

## 2. Qualitative case — Researcher perceptions (PubPeer / Zenodo)

| Field | Value |
| --- | --- |
| Title | Qualitative-coded answers for PubPeer perceptions survey |
| Strand | Qualitative (open-ended verbatims; no star ratings) |
| Domain | Research integrity / post-publication peer review |
| Publisher | [Zenodo](https://zenodo.org/records/20413424) |
| DOI | [10.5281/zenodo.20413424](https://doi.org/10.5281/zenodo.20413424) |
| Content | Anonymized open-ended survey answers + published coding scheme |
| License | CC BY 4.0 (cite authors) |
| Local file | `data/case_study_pubpeer_qual.csv` |
| Coding scheme | `data/case_study_pubpeer_coding_scheme.csv` |
| Provenance file | `data/case_study_pubpeer_qual.SOURCE.json` |
| Rebuild | `python scripts/prepare_case_study_qual_dataset.py` |

**Citation:** Hepkema, Wytske, & Bordignon, Frederique. (2026). Data for "\"PubPeer is okay, but ...\": researchers' perceptions of post-publication reviews" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20413424

## 3. General sample — Amazon Fine Food Reviews

| Field | Value |
| --- | --- |
| Title | Amazon Fine Food Reviews |
| Domain | Consumer product reviews |
| Origin | Stanford SNAP / Amazon review corpus |
| HF mirror used by script | `PJ2005/amazon-fine-food-reviews` |
| License | Typically cited as CC BY-SA 4.0 for the HF redistribution — verify on the dataset card |
| Local files | `data/sample_reviews.csv`, `data/sample_reviews.xlsx` |
| Rebuild | `python scripts/prepare_sample_dataset.py` |

## 4. Quick demo — synthetic mock

Generated locally by `generate_mock_reviews()` for UI walkthrough only.
**Not** a public research corpus — label it as synthetic in any portfolio write-up.

## Removed from featured demos

`data/case_study_admin_feedback.csv` (if present) was a synthetic enterprise-admin
scenario for UX prototyping. It is **not** a public dataset and should not be
presented as a research case study.
