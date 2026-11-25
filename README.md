# Indus Epigraphy AI

Computational epigraphy of the undeciphered Indus Valley script (BITS Pilani course project).
We treat the script as a **notational information system** and measure its structure; we do
**not** claim a decipherment, and sign IDs are opaque labels throughout.

| Milestone | Question | Code | Write-up |
|---|---|---|---|
| **M1** Visual grapheme extraction | Can we turn a seal image into an ordered list of sign IDs? | `src/m1_vision/` | `reports/MILESTONE1_SUMMARY.md` |
| **M2** Positional bias & Markov chains | Is the writing direction fixed? Which signs open/close texts? Which pairs go together? | `src/m2_markov/` | `reports/m2/summary/` |
| **M3** Masked-token encoder | How well can a missing sign be predicted from its context, with honest confidence? | `src/m3_masking/` | `src/m3_masking/README.md` |
| **M4** Benchmarking semantic hallucinations | Do claims (readings, generated text, model confidence) beat structural noise? | `src/m4_guardrails/` | `reports/M4_REPORT.md` |
| **Engine** | All four milestones in one command | `src/engine/` | `src/engine/README.md` |

Shared code: `src/common/signs.py` converts between the three sign numberings used by the data
(Mahadevan IM77 for M1 images, Parpola for CISI/M2, Wells for the M3/M4 corpus; validated at
91.8% agreement on 785 aligned sign slots).

**Reading order.** The script reads right-to-left. Both corpora store signs left-to-right as they
appear on the impression, so the first *stored* sign is the last one *read*. Every result states
which order it uses.

## Headline results

| | Result |
|---|---|
| M1 detection (77 inscription strips) | boxes on 99% of strips, 86% of boxes are real signs |
| M1 recognition (417 signs, hand-judged samples of 70) | 17% top-1, 41% top-5 on real crops (88% on synthetic: a sim-to-real gap) |
| M2 direction (CISI, 179 texts) | flip-test divergence 0.89 bits vs ≤0.42 for random directions (p < 0.0001); replicated on the larger 5,679-row corpus from all sites |
| M3 gap filling (5-fold cross-validation, 12,525 predictions) | ensemble 33.3% top-1, perplexity 28.8 vs Witten–Bell bigram 32.5%, 31.8 |
| M4 structural controls | shuffling a real inscription's signs raises perplexity ~11×; calibrated "High confidence" still fires on 8% of pure noise (0% correct) |
| M4 on our own pipeline | 74% of M1's real readings are flagged as unlike real text; letting M3 re-rank them lowers that to 35% without any evidence they became more correct |

Full numbers, caveats and limitations are in each milestone's write-up.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data is not in the repository. See **`DATA_SOURCES.md`** for every dataset, where to get it and
where to put it.

**Quickest way to run it:** download `indus_demo.zip` from this repository's
[Releases](https://github.com/meemansa-jha/indus-epigraphy/releases) page. It contains the trained
weights and the data the demo needs. Unzip it and follow `START_HERE.md`; `python demo/demo.py` runs
the full engine in about 30 seconds.

## Reproducing

```bash
bash run_all.sh            # everything except the slow model training (uses saved results)
bash run_all.sh --train    # also retrains the M1 detector/recognizer and the M3 models
```

Individual steps are listed inside `run_all.sh`. Example engine runs:

```bash
python src/engine/parse.py --seq "P385 ? P324" --numbering parpola   # typed transcription, reading order
python src/engine/parse.py --image data/raw/isd/1/1text.jpg           # image -> M1 -> M2 -> M3 -> M4
```

## Repository layout

```
src/common/        shared code: sign-numbering crosswalk, corpus loader, reading order, Witten-Bell smoothing
src/m1_vision/     synthetic data, YOLO detector, recognizers, strip/seal readers
src/m2_markov/     M2 notebook + replication on the larger corpus, word2vec, LLM2Vec, Mahadevan Table I check
src/m3_masking/    masked-sign Transformer, baselines, cross-validation, predict.py
src/m4_guardrails/ structural controls, perplexity benchmark, Naive Bayes guardrail, experiments E1-E8
src/engine/        the unified parsing engine
notebooks/         one notebook per milestone (01-04); they load saved results and run in seconds
reports/           write-ups, tables and figures
```
