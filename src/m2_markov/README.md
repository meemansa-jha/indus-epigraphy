# Indus Epigraphy AI — Milestone 2: Positional Bias & Markov Chain Modelling

## Contents

| Path | What it is |
|---|---|
| `m2_markov.ipynb` | **The notebook.** Steps 0–8, already executed, with all outputs visible. |
| `build_notebook.py` | Script that writes the notebook. Edit this, not the `.ipynb`, then rebuild. |
| `Indus_Epigraphy_AI_Milestone_2_Summary.md` | Short results summary for the team and the deck. |
| `make_report.py` | Builds the detailed summary report (HTML) from `outputs/`. |
| `data/cisi_corpus/` | Snapshot of the CISI corpus (`mayig/indus-valley-script-corpus`, commit `ad2f1e2`, MIT/Apache licence). |
| `outputs/` | All tables (CSV/JSON) and `outputs/figures/` (9 charts, 200 dpi). |

### Key output files

| File | Contents |
|---|---|
| `markov_witten_bell.csv` | **Main deliverable.** Smoothed 182×182 transition matrix. Rows = current sign (incl. `<S>` = text start); columns = next sign (incl. `</S>` = text end). |
| `markov_counts.csv`, `markov_mle.csv` | Raw pair counts and unsmoothed probabilities. |
| `directional_rules.csv` | Per-sign start/end rates, significance (BH-corrected q-values), and role (opener / ender / medial). |
| `collocations.csv` | Every sign→sign pair with PMI, NPMI, Fisher q-value, and collocation flag. |
| `pair_order_asymmetry.csv` | Frequent pairs: A→B vs B→A counts. |
| `segments_rtl.json`, `texts_reading_order.csv` | Cleaned corpus in right-to-left reading order. |
| `error_propagation_runs.csv` | All 700 simulation runs from Step 7. |
| `results_summary.json` | Every headline number, computed by the notebook. |

## Conventions

- **Reading order is right-to-left.** In all outputs, position 0 is the first sign read. The raw corpus stores signs left-to-right, and Step 0 reverses them.
- **Sign IDs use Parpola numbering** (`P324`, …), as in CISI. They are not Mahadevan numbers.
- **`P000` (lost material) is a break**, not a sign. No pair is counted across it, and an edge next to it is not treated as a genuine beginning or ending.

## Re-running

```bash
pip install -r requirements.txt
python build_notebook.py                                   # writes m2_markov.ipynb
jupyter nbconvert --to notebook --execute --inplace m2_markov.ipynb --ExecutePreprocessor.timeout=1800
```

A full run takes about 1–2 minutes and is deterministic (fixed random seeds).

To regenerate the detailed PDF/HTML report from the outputs:

```bash
python make_report.py Milestone_2_Detailed_Summary.html   # then print to PDF from any browser
```
