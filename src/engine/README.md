# Epigraphic Parsing Engine

One command through all four milestones (the proposal's "unified Python script ... to output
positional n-gram probability reports").

```bash
python src/engine/parse.py --image data/raw/isd/1/1text.jpg            # photo / strip -> M1 -> M2 -> M3 -> M4
python src/engine/parse.py --seq "P385 ? P324" --numbering parpola     # typed transcription, READING order
python src/engine/parse.py --seq "267 99 342"  --numbering im77
python src/engine/make_synthetic_inputs.py --n 100   # real test inscriptions drawn as synthetic strips
python src/engine/eval_synthetic.py                  # end-to-end scores against known truth
python src/engine/eval_real.py                       # verdicts on the 77 real strips
python src/engine/generate_and_read.py               # generated texts: novelty filter, render, read back (M4 E8b)
python demo/demo.py                                  # the screen-recorded demo (7 steps)
```

Each run writes `report.json`, `m2_positional_ngrams.csv`, `panels.png` and `index.html`.
The report always records its `source`: "M1 automatic reading (not verified)" or
"typed / transcribed sequence".

| Stage | What it reports |
|---|---|
| M1 | boxes, top-5 Mahadevan candidates per sign (image input only) |
| conversion | IM77 / Parpola -> Wells through `src/common/signs.py`; unmapped signs become gaps |
| M2 | reading-order Witten-Bell bigram with START/END (Milestone 2 definitions, fitted on the M3 v2 train split): P(opens), P(next given previous), P(closes), bits, known collocations, sign roles |
| M3 | each sign hidden in turn: calibrated ensemble probability and top-3 alternatives |
| M4 | pseudo-perplexity vs real held-out inscriptions and structural controls -> verdict |

Results are in `reports/engine/` (`synthetic_eval.json`, `real_eval.json`, `generated_roundtrip.json`, `generated_gallery.png`).
