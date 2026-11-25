# Indus Epigraphy AI — Milestone 2 (complete package)

**Positional Bias & Markov Chain Modelling.** Everything produced for Milestone 2 is in this folder.

## Start here
**`1_Summary/Milestone_2_Detailed_Summary.pdf`** is the full write-up (21 pages): context, data decisions, every step with its results, figures and verification checkpoints, the deliverables mapped to the proposal, limitations, the figure guide for the deck, and full appendix tables.

## Folder map

| Folder | Contents |
|---|---|
| `1_Summary/` | `Milestone_2_Detailed_Summary.pdf` (print/share) · `.html` (the same report, opens in any browser) · `Milestone_2_Summary_short.md` (short version, same style as the M1 summary) |
| `2_Figures/` | All 9 charts (PNG, 200 dpi), numbered `m2_01` … `m2_09` |
| `3_Results_Tables/` | All result tables (CSV/JSON); see below |
| `4_Notebook/` | `m2_markov.ipynb` (executed notebook) · `m2_markov_view.html` (read-only view, no Jupyter needed) |
| `5_Runnable_Project/indus_m2/` | Self-contained project: notebook, build scripts, corpus snapshot, outputs, `README.md`, `requirements.txt`. Re-running reproduces every number exactly. |
| `6_Project_Inputs/` | The project proposal and the Milestone 1 summary this work builds on |

## Results tables (`3_Results_Tables/`)

| File | Contents |
|---|---|
| `markov_witten_bell.csv` | **Main deliverable.** 182×182 Witten-Bell–smoothed transition matrix (`<S>` = text start row, `</S>` = text end column) |
| `markov_counts.csv` · `markov_mle.csv` | Raw pair counts · unsmoothed probabilities |
| `directional_rules.csv` | Per-sign start/end rates, significance, and role (opener / ender / medial) |
| `collocations.csv` | All 529 sign pairs with PMI, NPMI, q-value, collocation flag |
| `pair_order_asymmetry.csv` | Frequent pairs: A→B vs B→A |
| `segments_rtl.json` · `texts_reading_order.csv` | Cleaned corpus in right-to-left reading order |
| `error_propagation_runs.csv` | All 700 error-propagation simulation runs |
| `results_summary.json` | Every headline number, computed by the notebook |

## Conventions
- **Reading order: right-to-left.** Position 0 is the first sign read.
- **Sign IDs: Parpola numbering** (`P324`, …), as in the CISI corpus.
- **`P000` = lost material**, treated as a break, never as a sign.

## Note for the deck
The Milestone 1 slides say `P324` *opens* texts. In reading order it **closes** them, and `P385` opens them. Please update M1 so that both milestones agree.
