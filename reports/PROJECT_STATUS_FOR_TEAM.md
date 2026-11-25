# Indus Epigraphy AI: where each milestone stands

*Status note for the team. All numbers come from the saved results in the repository (`reports/`, `src/*/artifacts`, `notebooks/01–04`). Every number was re-checked by running `run_all.sh` end to end. The repository's documentation and the notebooks were checked too.*

---

## TL;DR

| Milestone | What it does | Does it deliver what the proposal asked? |
|---|---|---|
| **M1** Vision | Photo → ordered sign IDs | **Partly.** It finds the signs but cannot identify them: 12–17% correct on real photos. |
| **M2** Markov / positions | Direction, positional preferences, smoothing, collocations | **Yes.** The most solid milestone, and it replicates on a corpus 20× larger. |
| **M3** Masked encoder | Predicts damaged signs with calibrated confidence | **Yes, but the gain is small.** It barely beats a simple bigram model. |
| **M4** Hallucination benchmark | Tests every claim against fake inscriptions | **Yes.** It is conceptually the strongest milestone. |

**Overall:** the project is honest and defensible, but it is not a story of success. The two questions we should expect are:

1. Why doesn't M1 read real seals?
2. Why does deep learning (M3) barely beat a bigram?

We have measured answers to both.

---

## Milestone 1: photo → list of sign IDs

**What it does.** It takes a photo of an inscription strip, boxes each sign, and identifies which of the 417 Mahadevan signs each one is. It then orders the signs right-to-left.

**How.**
- The only sign dataset has **one image per sign**, and there are no labelled real photos.
- So all training data is synthetic: we rendered the 417 templates as carved stone, with random lighting, wear and blur.
- A YOLO detector finds the signs, and a MobileNet classifier identifies them.

**Results.**

| | Synthetic data | Real strips |
|---|---|---|
| Finding signs | mAP50 0.995 | 76/77 strips get boxes; 77–86% of boxes are real signs; 4.9 signs per strip vs a true 5.6 |
| Identifying the sign (top-1) | 87.9% | **17%** best measured [95% CI 9–29%]; **12%** for the model actually saved |
| Right sign in the top 5 | 97.7% | 40–41% |
| Whole inscription exactly right | 70% | not measurable (no ground truth) |

**Verdict.**
- **Detection works. Recognition does not.**
- 17% is about 70× better than chance (0.24%), so the model learned something. But 4–5 out of every 6 signs are wrong, so the output is not a usable transcription.
- The defensible finding is this: **every model scored near-perfectly on synthetic data and failed on real data.**
- For comparison, the published work (Dixit et al. 2025) hand-labelled 245 real images and handled only 40 sign classes. We had zero labelled real images and attempted all 417.

**Known problems.**
- Real accuracy comes from two hand-judged samples of 70 boxes each, so the confidence intervals are wide (±10–13 points).
- The 17% model's weights were overwritten. The repository contains the 12% model, and the engine and M4 use that one.
- The two reviews' detection precision (77% vs 86%) came from the **same detector**, so the difference is sampling noise; pooled, it is 81%.
- Detector mAP50-95 is **0.875** (from `results.csv`), not the 0.855 quoted in an earlier draft. The summary has been corrected.

---

## Milestone 2: positional statistics and Markov chains

**What it does.** It treats signs as opaque symbols and asks four questions:
- Is there a fixed writing direction?
- Do some signs prefer the start or end of a text?
- Which signs follow which (a bigram Markov chain), and how should it be smoothed?
- Which pairs of signs occur together more often than chance?

**Results on the CISI corpus (179 inscriptions).**
- **Endings are far more predictable than beginnings.**
  - The ending position carries 2.4 bits less uncertainty than the beginning.
  - The flip test, which checks whether beginnings and endings differ more than chance, gives JSD 0.89 against at most 0.42 under random flipping (p < 0.0001).
- **The jar sign P324 ends 45% of complete texts.**
- **Smoothing is necessary.** With no smoothing, 46% of held-out sign pairs get probability zero. Witten–Bell perplexity is ≈41 vs ≈77 for add-one. (Perplexity is roughly "how many signs the model is choosing between"; lower is better.)
- **41 significant collocations** were found.
- **Error propagation.** At M1's accuracy almost nothing survives: direction is detected in only about 12% of simulated runs.

**Replication on the larger corpus (3,491 unique inscriptions, the M3/M4 data).**
- Every finding reproduces:
  - the jar sign ends 31% of texts
  - flip test 0.63 vs 0.13
  - beginning-to-end gap 2.15 bits
  - 453 collocations
- **One finding changes.** M2 says forward and backward models are indistinguishable ("only smoothing noise"). On the larger corpus, **right-to-left reading order predicts held-out text better in 9 of 10 folds** (p ≈ 0.02). The two directions are exactly equal only on the training data.
- **External check against Mahadevan (1977).** The jar sign comes last in a text 69.6% of the time in his published table and 68.4% in our corpus.

**Verdict.** It meets the proposal. It is the most solid milestone, because every claim is tested against a shuffled null.

**For the M2 owner, two changes:**
1. The "only smoothing noise" statement should be softened, for the reason above.
2. Sign frequencies with equal counts were sorted in an order that depended on the library version, so the Step 7 simulation gave different numbers on different machines. Ties are now broken by sign label (`build_notebook.py`, Step 1). As a result, "direction detected at M1's accuracy" moved from **10% to 12%**, which is within simulation noise. No other headline number changed. The summary carries a note.

---

## Milestone 3: masked-sign model (filling in damaged signs)

**What it does.** It hides one sign and predicts it from the rest, with calibrated confidence: "70% sure" should be right about 70% of the time.

**How.**
- A small Transformer (2 layers, 64 dimensions), chosen by a real model search.
- Three training runs with different random seeds, combined as an ensemble.
- Temperature calibration.
- **The duplicate leak is fixed:** the same inscription no longer appears in both train and test.

**Results (5-fold cross-validation on all 3,491 inscriptions, 12,525 predictions).**

| Model | Top-1 | Top-5 | Perplexity |
|---|---|---|---|
| Frequency only | 10.3% | 24.1% | 112 |
| Witten–Bell bigram (M2's simple model) | 32.5% | 55.2% | 31.8 |
| Single Transformer | 32.2% | 54.1% | 32.1 |
| **Ensemble of 3 (deliverable)** | **33.3%** | **55.7%** | **28.8** |

**How we know the ensemble's lead is real.**
- Ensemble vs bigram, McNemar test (which model wins the cases where they disagree): p = 0.008.
- Lower perplexity in all 5 folds.

**Calibration.** "High" confidence predictions are right 75% of the time, against a stated 72%.

**Verdict.**
- **A single Transformer is no better than a bigram that takes one line to compute.** The ensemble wins by about 1 point of top-1: statistically real, but practically small.
- This is probably not a bug. Inscriptions are about 5 signs long, so there is little context beyond the neighbouring signs, and a bigram already uses those.
- Present it as "short texts limit what context models can add". Do not present it as "deep learning works".

**Faculty suggestions (word2vec, LLM2Vec), 3 runs each.**
- word2vec: 22.0% top-1.
- LLM2Vec: 27.6%.
- Both are below our encoder (32.7%) and the bigram (33.5%).
- LLM2Vec was trained from scratch at small size, not the pretrained version from the paper. See `reports/m2/embedding_comparison.md` for its 14 listed assumptions.

---

## Milestone 4: benchmarking semantic hallucinations

**What it does.** It tests every claim against fake inscriptions: shuffled real texts, signs drawn at random by frequency, uniformly random signs, and Markov-generated texts. A method that cannot tell real from fake is making empty claims.

**Results.**

**E1: Perplexity.**
- Real text scores 19.6, shuffled text 224, random text 504–2,995. Both models separate real text from noise easily.
- Markov fakes:
  - The Transformer scores them at 54, which separates them from real text.
  - The bigram scores them at 24.7 vs 25.2 for real text, so it cannot tell them apart. **A model cannot detect fakes that are made of its own statistics.**

**E2: Confidence on noise.** On pure noise, the calibrated model still marks **8–10%** of its guesses as "High" confidence, and those are almost never right. **Confidence is not truth.**

**E3: Guardrails.**
- **Naive Bayes guardrail.** Its AUC (0.5 = coin flip, 1 = perfect) is 0.98 against random signs and 0.85 against shuffled text, but **0.51 against Markov fakes**: useless against realistic fakes.
- **Perplexity guardrail.** It flags 6% of real texts (≈5% by design) and only 8% of Markov fakes.

**E4: The synthetic corpus we found (hellosindh) is unrealistic.**
- Average length is 7.5 signs vs 4.1 in real texts.
- The same sign repeats next to itself 25.5% of the time, vs 0.4% in real texts.

**E5: The Sanskrit readings in the CSV.**
- They are internally consistent: the first sign predicts the first sound of the reading 71% of the time, vs 18% for shuffled readings.
- **This shows the readings were assigned systematically, sign by sign. It does not show they are correct.** No test we can run decides correctness, and we say so.

**E6: Error propagation.** At M1's measured error rate:
- M3's accuracy falls from 34% to 8%.
- The guardrail flags 70% of texts.
- The direction signal falls to near chance (54% vs 50%).

**E7: The guardrail on M1's real output.**
- It flags **74%** of strips, so it correctly recognises M1's readings as noise.
- Letting M3 re-rank M1's top-5 guesses brings the flag rate down to 35%. **This is a hallucination we created ourselves.** The output became plausible, not correct: 158 signs changed, with no ground truth to check against.

**E8: Generating new inscriptions (the professor's request).**
- Texts generated by M3 pass the guardrail, with perplexity 18.2, lower than real text.
- But **23% are exact copies of training inscriptions.**

**E8b: Novelty filter and round trip** (`src/engine/generate_and_read.py`).
- About **half** of the generated texts are new: 52% for Markov and 46% for Gibbs are at least two edits from every training text. Another ~31% are only one sign away.
- About 45–50% are novel *and* pass the guardrail. Gibbs's accepted texts are closer to real text than Markov's (pseudo-perplexity 27 vs 46; real text 19.6).
- 60 accepted texts were drawn as carved strips and read back by the engine. The sign count was right on 90%. On those, 91% of signs and 67% of whole sequences were read back exactly (synthetic images, so this is an upper bound).
- **The guardrail passed 96% of the *misread* read-backs.** It catches noise, not single-sign errors.

**The engine (M1 → M2 → M3 → M4).**
- On synthetic photos:
  - correct sign count 80% of the time
  - 87.9% of signs right
  - 70% of inscriptions exactly right
- On real photos, 74% are flagged and 23 of 77 strips are unreadable. **The guardrail catches M1's failure, which is the correct behaviour.**

**Verdict.** It meets the proposal and is the strongest milestone conceptually: M1's failure becomes evidence for M4's argument.

---

## If we leave M1 out, is it a success?

**Mostly yes, as an honest analysis rather than a working tool.**

**What succeeds:**
- **M2:** every finding holds against shuffled data, replicates on a 20× larger corpus, and matches Mahadevan's published counts within about 1 point. It also corrects an over-strong claim in our own M2 write-up.
- **M4:** it does what the proposal asked. Its key findings stand on their own:
  - a model cannot catch fakes built from its own statistics
  - confidence is not truth
  - the Sanskrit readings are systematic but not shown to be correct
  - generated text copies its training data

**What only partly succeeds:**
- **M3** is well tested but its headline is modest: a single model ties the bigram, and the ensemble wins by about 1 point.
- **Nothing here reads the script.** M2–M4 show the script has structure and that our guardrails catch nonsense. We must not let the framing drift toward "we understood the inscriptions".

**Two reasons not to drop M1:**
1. **We cannot remove it.** It is one of the four milestones and the proposal's first deliverable. What we can change is the framing: from "our reader" to "an experiment whose failure we measured".
2. **Some of M4's strongest evidence uses M1:** E6, E7 and the engine's real-photo test. Without them, M4 shows the guardrail working only on fakes we built ourselves, not on a real failure.

**Recommended story:**
- **M1** shows how easily AI produces confident wrong output.
- **M2 and M4** show how to detect it.
- **M3** shows the honest ceiling on 5-sign texts.

M1's failure is part of the result, not something to hide.

---

## Every assumption made during integration

1. **The CSV source is undocumented.** It appears to come from Yajnadevam's dataset, but this is unconfirmed. **Someone needs to find and cite it.**
2. **Sign-numbering crosswalk.**
   - It is built from the CISI feature files and agrees on 91.8% of 785 checked positions.
   - Where a sign has several candidates, the most frequent is picked.
   - 83 of the 417 Mahadevan signs have no equivalent and become "unknown".
3. **Reading order.**
   - Both corpora are stored left-to-right; we reverse them to get reading order.
   - The CSV's `<S>`, `<E>` and `000` are assumed to mean start, end and damage. The statistics support this (the jar sign lands at the end), but it is not documented.
4. **Duplicates.** Exact duplicate sign sequences are removed. Near-duplicates (differing by one sign) remain, so test results may be slightly optimistic.
5. **M3 vocabulary.** Signs seen fewer than 2 times are treated as unknown; this affects 4.3% of test signs.
6. **Guardrail threshold.** It is set at the 95th percentile of validation perplexity (597), so it flags about 5% of real texts by construction. A different cut-off changes every flag rate.
7. **Fake inscriptions are assumed not to be language.** The Markov fakes are deliberately hard. The shuffled and random ones are easy, so beating them is a low bar.
8. **E6 error model.**
   - Errors are independent, and substitutions are drawn from M1's predictions.
   - The rates mix two hand reviews: 17% recognition from one, 86% detection precision from the other. This keeps them consistent with M2's Step 7.
   - The saved model's numbers would mean slightly worse corruption, so E6 slightly understates the damage.
9. **E5.** The CSV readings are taken as given, and the first sound is read from the transliteration's first letter. A different sound system would change the numbers.
10. **LLM2Vec** was trained from scratch at small size, not the pretrained version in the paper.
11. **The engine's synthetic test is optimistic.** It uses the same renderer family that generated M1's training data, so 87.9% overstates real performance.
12. **M1 orientation.** Photos are assumed to show the impression, not the mirror-image seal. A test on 262 crops found no evidence of mirroring.
13. **Hand reviews.** We should state who judged them and whether the judge was blind to the model's guess. If one person judged without blinding, the reviews could be biased either way.

---

## Open items

- [ ] Find and cite the source of the inscription CSV (`data/raw/m3_csv/inscriptions.csv`).
- [ ] M2 owner: soften the "only smoothing noise" claim; note the Step 7 change (10% → 12%).
- [ ] Record who did the M1 hand reviews and how.
- [ ] Decide on third-party images (`reports/deck_figures/` is excluded from the repository by default).
- [ ] Create the public GitHub repository (required for submission).
- [ ] Record the demo video using `demo/demo.py`; the shot list is in `demo/README.md`.
- [ ] Submission: a 4-page double-column report in the unmodified LaTeX template, with member contributions, a generative-AI disclosure and the GitHub link; plus a video of at most 3 minutes.

**Where to look:**
- `README.md`
- `reports/M4_REPORT.md`
- `reports/MILESTONE1_SUMMARY.md`
- `src/m1_vision/README.md`
- `src/m3_masking/README.md`
- `reports/m2/FACULTY_SUGGESTIONS_RESULTS.md`
- `notebooks/01–04`
