# Milestone 3 — Sequential Entropy & Masked Token Encoders (Indus script)

Predicts hidden/damaged signs in an inscription from the surrounding signs, with calibrated confidence.
Trained **only** on the `text` (sign sequence) column of `inscriptions.csv`; the `sanskrit` / `translation` /
`notes` columns are never read.

## Quick start
```bash
pip install -r requirements.txt
python predict.py "+740-000-590+" --top-k 5        # fill a damaged sign
python predict.py "+740-220-590+" --mask 1         # hide sign #1 to test the model
python predict.py "342 ? 59" --numbering im77 --order reading   # Milestone 1 output (IM77, right-to-left)
python predict.py "P385 ? P324" --numbering parpola --order reading   # Milestone 2 / CISI IDs
python run_all.py                                  # re-run split + training + evaluation (~20 min on 1 CPU)
```
```python
from predict import SignCompleter
sc = SignCompleter()
sc.complete("+740-000-590+")          # string in corpus format; 000, ? or [MASK] mark gaps
sc.complete(["740", "?", "590"])      # list of sign IDs (e.g. Milestone 1 output); edges="complete"|"broken"
```
Returns, per gap: top-k signs with calibrated probabilities and a **High (≥0.5) / Medium (0.2–0.5) / Low (<0.2)** rating.

## Pipeline
1. **Parse** `+400-740-405-590[` → `<S> 400 740 405 590 <E?>`. `+` = complete edge, `[`/`]` = broken edge
   (explicit edge tokens replace [CLS]/[SEP]). `/` alternatives → first readable one. `000` (illegible) is fed as a
   `[MASK]` and is never a training/eval target.
2. **Deduplicate on the sign sequence** (5,679 rows → 3,491 unique) *before* splitting, so no sign sequence appears in
   two splits. *(v2 fix: v1 deduplicated on the full token string including edge markers, so `+740-220-590+` and
   `[740-220-590+` counted as different texts and 37 of 373 test inscriptions shared their signs with training data.
   Each sequence now keeps its most frequent edge variant.)* Split 80/10/10 (2,792 / 349 / 350), seed 0.
   Verified: 0 sign sequences shared between any two splits.
3. **Vocabulary**: signs seen ≥2× in train (426); others → `[UNK]` (4.3% of test signs; excluded from metrics).
4. **Model**: Transformer encoder, tied embeddings, trained from scratch with BERT-style dynamic masking (25% per
   sequence, ≥1 target, 80/10/10). **Grid search actually run** over 5 configurations (one seed each, validation NLL,
   results in `artifacts/search.json`; v1 had typed-in numbers): chosen **2 layers, d=64, 4 heads, ff=128, dropout 0.2**
   (val NLL 3.294; others 3.372–3.469). Final = **ensemble of 3 seeds** + **temperature scaling (T=1.15) fit on validation**.
   Each seed's learning curve is saved (`artifacts/history_seed{0,1,2}.json`), so re-running `run_all.py` reproduces
   `results.json` exactly and never wipes the learning-curve panel.
5. **Baselines** under the identical protocol: unigram frequency; bidirectional Witten-Bell bigram.

## Evaluation protocol (test set, 350 sequences)
* **LOO**: every readable sign masked once (1,304 predictions). Existing `000`s remain masked context.
* **Multi**: ~40% of signs masked at once, one example per sequence (557 predictions).

## Results (test)
| Model (LOO) | Top-1 | Top-3 | Top-5 | Perplexity | ECE |
|---|---|---|---|---|---|
| Unigram | 0.110 | 0.190 | 0.244 | 111.9 | 0.012 |
| Witten-Bell bigram | **0.335** | 0.496 | 0.563 | 31.2 | 0.028 |
| Transformer, single seed (mean±std of 3) | 0.327±0.005 | 0.485±0.005 | 0.549±0.006 | 32.1±0.2 | 0.085±0.004 |
| Transformer ensemble | 0.334 | 0.504 | 0.570 | 27.3 | 0.064 |
| **Transformer ensemble + calibration** | 0.334 | **0.504** | **0.570** | **26.1** | 0.035 |

Multi-mask: unigram 0.113 / bigram 0.275 / ensemble 0.282 top-1 (perplexity 117 / 43.8 / 38.5 calibrated).

**Transformer vs bigram, tested:** top-1 is **not** different (33.4% vs 33.5%; on the 140 predictions where exactly one
of them is right, the Transformer wins 69 and the bigram 71; McNemar p = 0.93). Perplexity **is** lower for the
Transformer (26.1 vs 31.2; difference 5.1, bootstrap 95% CI 2.9–7.4, resampling predictions). Entropy: unigram 6.8,
bigram 5.0, ensemble 4.7 bits per sign.

**5-fold cross-validation (`cv.py`, results in `reports/m3/cv.json`).** All 3,491 unique sequences are tested once
(12,525 hidden-sign predictions). Mean ± sd over the 5 folds:

| Model | Top-1 | Top-5 | Perplexity | ECE |
|---|---|---|---|---|
| Unigram | 10.3% ± 0.3 | 24.1% ± 0.6 | 112.0 ± 3.8 | 0.005 |
| Witten-Bell bigram | 32.5% ± 1.2 | 55.2% ± 0.9 | 31.8 ± 1.0 | 0.036 |
| Transformer, one model (calibrated) | 32.2% ± 0.8 | 54.1% ± 0.7 | 32.1 ± 1.0 | 0.027 |
| **Transformer, ensemble of 3 (calibrated) = the deliverable** | **33.3% ± 0.6** | **55.7% ± 0.4** | **28.8 ± 0.5** | **0.019** |

* A **single** Transformer is **no better than the bigram** on either measure (top-1 wins 1/5 folds, perplexity 3/5).
* The **3-model ensemble** beats the bigram consistently but modestly: top-1 +0.8 points (33.3% vs 32.5%; better in 4/5 folds;
  on pooled predictions 761 vs 660 disagreements, McNemar p = 0.008) and perplexity 28.8 vs 31.8 (better in 5/5 folds;
  gap 3.0, bootstrap 95% CI 2.3–3.7). The McNemar test treats predictions as independent; signs from the same inscription are
  not fully independent, so the p-value is somewhat optimistic.
* So the honest claim is: **ensembling, not the Transformer architecture by itself, is what beats the smoothed bigram, and by a
  small margin.** The single split above (1,304 predictions) was too small to detect a 0.8-point top-1 difference.

**Confidence bands (test, calibrated)**: High n=294, accuracy 0.75; Medium n=459, accuracy 0.35; Low n=551, accuracy 0.10 —
the ratings track real accuracy. See `artifacts/figures.png` (reliability diagram, accuracy by length).

## What to take from this honestly
* **The neural model is not better than the smoothed bigram at picking the right sign** (top-1 33.4% vs 33.5%,
  p = 0.93). It assigns its probability better (perplexity 26.1 vs 31.2), so it is the better *probability model*,
  not the better *guesser*. (v1 reported +2 points top-1 for the Transformer; on the leak-free split that gap is gone.)
  5-fold cross-validation (below) shows the 3-model ensemble does beat the bigram, by a small margin.
* The data is small (≈3,000 short sequences, median 3 signs), which is the real limit. Context-poor cases are weak:
  1-sign sequences ≈5% top-1; accuracy rises to ~37–42% at 5+ signs.
* Accuracy depends heavily on sign frequency: 62% when the true sign is among the 10 most common, 8% for signs
  outside the top 50. The model largely learns frequent seal formulas and positional habits (first *stored*
  position 51% = the last sign READ, e.g. the jar sign; last stored position 16% = the first sign read), not deep
  structure.
* Masked-sign accuracy measures pattern completion on known signs. There is **no ground truth** for genuinely damaged
  signs, so predictions for real `000`s are plausible completions, not verified readings.

## Assumptions / limitations
* Sign IDs follow the CSV's own numbering, which is **Wells** numbering (up to 999; **not** Mahadevan's 417 list).
  The crosswalk in `src/common/signs.py` converts Milestone 1's IM77 IDs (validated at 91.8% agreement).
  `predict.py` converts on request (v2): `numbering="im77"|"parpola"`, `order="reading"`; results come back in the
  input numbering and order. Signs with no crosswalk entry are reported as `unmapped` and treated as holes, never guessed.
* Text is used in **stored order**, which is the reverse of reading order (the corpus is transcribed left-to-right on
  the impression; the script reads right-to-left). The CSV already normalises all texts to this orientation, so the
  `dir.` column is not needed.
* Duplicate deduplication changes the true frequency distribution (common seals appear once); Milestone 2 may prefer to
  count duplicates separately.
* Hyper-parameters were chosen with one seed per configuration; seed-to-seed spread in validation NLL (3.29–3.34) is
  smaller than the gap between the chosen and the next-best configuration (3.29 vs 3.37), but not by a wide margin.
* The headline table uses one split; 5-fold cross-validation (above) confirms it within about one standard deviation.
* The corpus is a derivative CSV; the script itself remains undeciphered in the academic literature.

## Files
`data.py` parsing/splits/masking · `baselines.py` · `model.py` · `evaluate.py` · `run_all.py` · `predict.py` ·
`artifacts/` (`ckpt_seed{0,1,2}.pt`, `splits.json` incl. vocabulary, `config.json`, `results.json`, `figures.png`)
