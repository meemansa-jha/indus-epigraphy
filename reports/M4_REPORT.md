# Milestone 4 — Benchmarking Semantic Hallucinations

**Indus Epigraphy AI · Empirical research report**

> **Proposal objective:** "Build validation guardrails to test and statistically disprove absolute language claims … plot generative perplexities against random noise baselines … evaluate proposed readings using a multinomial Naive Bayes classifier or a controlled verification harness."
>
> **Deliverable:** "An empirical research report disproving arbitrary linguistic translations by evaluating confidence scoring metrics against empirical structural controls."

All numbers in this report come from scripts in `src/m4_guardrails/`. Their outputs are in `reports/m4/`. Section 15 lists the commands that reproduce every number.

---

## Summary of findings

We built a **guardrail**: a test that asks whether a sign sequence, or a claim about one, behaves like real Indus text. It does this by comparing the sequence with **structural controls**, which are fake inscriptions with known defects. We then applied the guardrail to every kind of "claim" this project produced or encountered: our own models' outputs, our own vision pipeline's readings, a published synthetic corpus, our own generated text, and a published Sanskrit decipherment.

1. **Sign order carries real structure.** Shuffling the signs of a real inscription raises its perplexity from a median of 19.6 to 224.3, about 11 times higher. Signs drawn at random by frequency reach 504, and uniformly random signs reach 2,995.
2. **Model confidence does not equal truth.** Milestone 3's calibrated model still rates 8–10% of its guesses on pure noise as "High confidence", and those High-confidence guesses are correct 0% of the time on uniform noise. On real text the same rating is correct 75% of the time.
3. **A guardrail is only as good as the structure it looks at.** A Naive Bayes classifier using sign and sign-pair counts separates real text from uniform noise (AUC 0.98) but cannot separate it from bigram-generated text (AUC 0.51). Our Transformer ranks those same fakes as less real-like than real text (median 54.2 vs 19.6, non-overlapping 95% intervals), but as a classifier it separates them only weakly (AUC 0.63).
4. **A published synthetic corpus that claims to be "indistinguishable" from real seals is easy to tell apart.** It is almost twice as long (7.53 vs about 4 signs), uses 170 instead of more than 640 distinct signs, and has a doubled sign in 68% of inscriptions versus 1.1% of real ones.
5. **Our own vision pipeline (M1) produces text that the guardrail flags as unlike real inscriptions 74% of the time.** Letting the language model (M3) re-rank M1's candidates cuts that to 35%. The output then *looks* real, but nothing shows it is more *correct*. This is a semantic hallucination produced inside our own pipeline, and it is the central result of the project.
6. **Digitization errors destroy the statistical findings.** At M1's measured error rates, the direction signal falls from 90% to 54% (chance is 50%), and M3's accuracy falls from 34% to 8%. This extends Milestone 2's own error-propagation study (its Step 7), which found the same collapse for M2's statistics.
7. **Milestone 2's conclusions replicate on 20 times more text.** Re-running M2's tests on the larger corpus used by M3/M4 (5,679 rows from all sites) confirms a fixed direction (flip test JSD 0.63 vs. a null maximum of 0.13) and the same dominant ender (the jar sign, P324 = W740).
8. **Generated text passes the guardrail and memorizes.** Our Markov and Transformer generators produce text the guardrail accepts. Even so, 16–23% of their outputs are exact copies of training inscriptions, and Naive Bayes still detects them weakly (AUC 0.56 and 0.66).
9. **The published Sanskrit decipherment is systematic, but its correctness was not tested.** Its readings follow a consistent sign-to-sound mapping in reading order: the first sign read predicts the first sound 71% of the time, against 18% for shuffled readings (p < 0.005). Consistency is required of any cipher key and is not evidence that the key is correct. The decisive test, whether the key reads real text better than shuffles of its own values, needs the key in machine-readable form, which we could not obtain (Section 9).

---

## 1. Question and scope

The Indus script is undeciphered. There is no bilingual text, inscriptions average about 5 signs, and no reading can be checked against a known language. That makes it an ideal setting for **semantic hallucination**: a human or a model can produce a confident "meaning" that nothing can verify.

Milestone 4 does **not** attempt a decipherment. It asks a narrower, testable question:

> *Does a given claim (a sequence, a reading, a confidence score, a generated text) behave differently from what structural noise would produce?*

A claim that noise satisfies equally well carries no evidence.

Throughout this report, signs are opaque labels. We never state what a sign means.

---

## 2. Data, sign numbering and reading order

### 2.1 Corpora

| Corpus | Size | Sign numbering | Used for |
|---|---|---|---|
| Inscription CSV (the M3 corpus) | 5,679 rows; 3,491 unique sign sequences (v2 deduplication) | **Wells** (3-digit, e.g. `740`) | M3 model, all M4 experiments |
| CISI digital corpus (Parpola et al.) | 179 sides, Mohenjo-daro | Parpola (`P###`) | Milestone 2; building and validating the crosswalk |
| IM-417-150 sign images | 417 templates | Mahadevan **IM77** (1–417) | M1 recognition |
| `hellosindh/indus-script-synthetic` | 5,000 generated sequences | Its own (`T###`) | Section 8 |

### 2.2 Crosswalk between the three numbering systems

Each Parpola grapheme in the CISI `features/` files lists its Wells and Mahadevan equivalents. We built a crosswalk from these files (`src/common/signs.py` → `data/processed/crosswalk.csv`).

To validate it, we aligned the 148 inscriptions that appear in both the CISI corpus and the Wells CSV. **91.8% of 785 aligned sign slots agree.** By comparison, only 9.8% agree when one sequence is reversed, which is what chance alignment looks like.

Coverage:

- 334 of 417 IM77 signs have a Wells equivalent.
- The mapping is many-to-many for some signs. For example, Wells `002` corresponds to IM77 99 and 100. In those cases we pick the candidate that is most frequent in the corpus and flag the choice as ambiguous.

### 2.3 Reading order versus stored order

Both corpora store signs **left to right as they appear on the impression**. The script reads **right to left**, so the first stored sign is the last sign read.

Evidence: the jar sign (Wells `740` = IM77 342 = P324), a well-known text-ender, is stored first in 71% of right-to-left inscriptions and last in only 3%.

Milestone 1 outputs signs in reading order, so its output is reversed before it is scored.

### 2.4 Milestone 2 replicated on the M3/M4 corpus

Milestone 2 established reading direction, positional rules and collocations on the CISI corpus (179 texts from one site, Parpola numbering). M3 and M4 use the larger Wells corpus. To connect the two, `src/m2_markov/replicate_wells.py` applies M2's **exact definitions** to both corpora:

- reading order = reversed stored order
- damage = a break
- START/END markers only at genuine edges
- Witten–Bell smoothing with a Witten–Bell-interpolated unigram
- JSD flip test
- positional entropy
- Fisher + Benjamini–Hochberg collocations

On CISI the script **reproduces M2's published numbers** (JSD 0.891, P324 ending 45.3% of texts, 41 collocations, 17 of 22 frequent pairs in one order only), which confirms the re-implementation is faithful.

| | CISI (M2) | Wells corpus, unique sign sequences (same policy as M3) |
|---|---|---|
| Complete texts | 153 | 2,106 |
| Distinct signs | 181 | 712 |
| Dominant ender (share of genuine endings) | P324, 45.3% | **W740 (= P324)**, 31.0% |
| Dominant beginner | P385 (= W817/861), 17.5% | W820 (= P378), 5.5% |
| Flip test JSD (null max) | 0.891 (0.418) | **0.633 (0.128)** |
| Position-1 entropy gap (null max) | 2.37 (1.31) bits | **2.15 (0.43) bits** |
| Held-out perplexity, reading order vs. reversed | 40.6 vs. 40.5 | 40.5 vs. 41.8 (reading order better in 9 of 10 folds) |
| Significant collocations | 41 | 453 |
| Frequent pairs seen in one order only | 17 of 22 | 210 of 358 |

**What this shows:**

- **M2's direction findings hold on a corpus about 20 times larger, drawn from all sites.** The jar sign is again the dominant ender.
- The beginner side is spread more thinly across sites. The top beginner, W820, is P378, which M2 had already identified as a significant opener.
- **The "perplexity trap" holds exactly only on training data**, where the two directions always score identically. On held-out data with 20 times more text, reading order is slightly but consistently better: 9 of 10 folds, perplexity 40.5 vs 41.8 (3%), sign test p ≈ 0.02. So overall perplexity is not completely blind to direction, but the signal is tiny compared with the flip test and the positional-entropy test, and on CISI's 179 texts it is invisible (4 of 10 folds). M2's warning stands in its practical form: overall perplexity is a poor direction test.

---

## 3. What is being tested: the models and the guardrail

**M3 Transformer ensemble.** This is the Milestone 3 masked-sign encoder: 3 seeds with temperature calibration. On held-out text it reaches 33.4% top-1, perplexity 26.1, and calibration error (ECE) 0.035. The Witten–Bell bigram reaches 33.5% top-1 and perplexity 31.2 on the same test: the Transformer is **not** better at top-1 (McNemar p = 0.93) but has clearly lower perplexity (gap 5.1, bootstrap 95% CI 2.9–7.4). Five-fold cross-validation over all 3,491 sequences (12,525 predictions) gives the 3-model ensemble 33.3% ± 0.6 top-1 and perplexity 28.8 ± 0.5, against 32.5% ± 1.2 and 31.8 ± 1.0 for the bigram: a small but consistent advantage (McNemar p = 0.008; better perplexity in 5 of 5 folds). A single Transformer is not better than the bigram. These are the numbers after the Milestone 3 deduplication fix (v2 split: 2,792 / 349 / 350 unique inscriptions, no sign sequence shared between splits).

**Witten–Bell bigram.** A Markov model with Witten–Bell smoothing, trained on the M3 train split.

**Pseudo-perplexity.** For each sign in a sequence, we hide it, predict it from the remaining signs, and record how surprised the model is. Averaging those surprises over the sequence and exponentiating gives the pseudo-perplexity. Lower means more like real Indus text.

**The guardrail.** A sequence is **flagged** when its M3 pseudo-perplexity exceeds **597**, which is the 95th percentile of real inscriptions in the **validation** split. Because the threshold is set on validation data, about 5% of real test inscriptions are flagged by design, and the test set plays no part in setting it.

---

## 4. Structural controls

For each of the 308 held-out test inscriptions with at least 2 readable signs, we built four fakes. Each fake keeps the original's length, edge markers and damaged positions (`controls.py`):

| Control | How it is made | What it tests |
|---|---|---|
| Shuffled | Same signs, random order | Does **order** matter? |
| Frequency-random | Signs drawn by corpus frequency | Do **combinations** matter? |
| Uniform-random | Any sign, equally likely | Pure noise |
| Markov-generated | Sampled from the bigram | Are sign **pairs** enough to fake text? |

---

## 5. Perplexity versus noise baselines (E1)

*Figure: `reports/m4/perplexity_vs_noise.png`*

Median pseudo-perplexity per inscription, with bootstrap 95% confidence intervals for the median (n = 308 per set; models trained on the train split only):

| Set | M3 Transformer | Bigram |
|---|---|---|
| **Real (held-out)** | **19.6** [16.5–26.9] | **25.2** [18.6–34.0] |
| Markov-generated | 54.2 [47.7–67.8] | **24.7** [20.3–30.8] |
| Shuffled | 224.3 [190.1–257.6] | 306.2 [254.1–383.3] |
| Frequency-random | 504.1 [433.1–591.4] | 496.8 [431.2–573.8] |
| Uniform-random | 2,995.2 [2,655–3,571] | 3,890.9 [3,541–4,296] |

**What this shows:**

- **Order carries information.** The same signs in random order are about 11 times more surprising than in their real order.
- **The bigram cannot see its own kind.** It scores Markov fakes the same as real inscriptions (24.7 vs 25.2; the intervals overlap almost completely). The Transformer, which sees the whole sequence, rates them as clearly less natural than real text (54.2 vs 19.6; the intervals do not overlap).

**Lesson for any guardrail:** a judge that shares the generator's assumptions cannot catch the generator.

---

## 6. Confidence scoring against structural controls (E2)

The Milestone 3 tool attaches a calibrated rating to every prediction: High (≥ 0.5), Medium, or Low. On each set we counted how often the rating is "High" and how often those High predictions are actually correct:

| Set | Predictions rated High | Of those, correct | Overall top-1 |
|---|---|---|---|
| **Real (held-out)** | 23.2% | **75%** | 34.2% |
| Markov-generated | 18.2% | 53% | 23.7% |
| Shuffled | 10.0% | 17% | 8.1% |
| Frequency-random | 9.4% | 6% | 3.0% |
| Uniform-random | 7.9% | **0%** | 0.3% |

**What this shows:**

- The calibration is good on real inscriptions (ECE 0.035).
- On noise, the model still says "High confidence" for about 1 guess in 12 and is never right.
- **A calibrated confidence score is only trustworthy on input that resembles its training data.** This is the "confidence scoring vs. structural controls" result the proposal asks for, and it applies directly to any AI system that reports confidence about Indus readings.

---

## 7. The Naive Bayes guardrail (E3)

We used multinomial Naive Bayes on counts of single signs and adjacent pairs, with edge markers included as tokens. Each fake type gets its own balanced binary task, evaluated with 5-fold stratified cross-validation (n = 3,047 real inscriptions vs. 3,047 fakes per task).

*Figure: `reports/m4/nb_guardrail.png`*

| Real vs. … | Accuracy | AUC |
|---|---|---|
| Uniform-random | 0.927 | 0.980 |
| Frequency-random | 0.801 | 0.895 |
| Shuffled | 0.760 | 0.850 |
| Markov-generated | 0.502 | **0.506** |

**The Markov result is correct, not a failure.** Naive Bayes sees only sign and pair counts, and a bigram generator reproduces exactly those counts. A guardrail cannot detect a fake that matches every statistic it measures.

For comparison, the same calibration done with the perplexity guardrail at its fixed threshold (`f7_guardrail_on_m1.py`; Wilson 95% intervals for flag rates, bootstrap 95% intervals for AUC):

| Set | Flagged | AUC vs. real |
|---|---|---|
| Real (held-out) | 6.2% [4.0–9.4] (about 5% by design) | — |
| Markov-generated | 8.1% [5.6–11.7] | 0.633 [0.587–0.675] |
| Shuffled | 21.8% [17.5–26.7] | 0.809 [0.775–0.844] |
| Frequency-random | 44.5% [39.0–50.1] | 0.885 [0.857–0.912] |
| Uniform-random | 93.8% [90.6–96.0] | 0.978 [0.966–0.988] |

The perplexity guardrail is conservative. It catches nearly all noise but only a minority of shuffles at the 95th-percentile threshold. Measured by AUC, the two guardrails perform about equally well, except that only the perplexity guardrail sees the Markov fakes at all (0.63 vs 0.51).

---

## 8. Synthetic reconstructions vs. historical inscriptions (E4)

The proposal's learning outcome: *"classifiers to isolate synthetic reconstructions from historical replicas."*

The `hellosindh/indus-script-synthetic` corpus describes its 5,000 sequences as filtered by a model ensemble (85% acceptance threshold), with a discriminator accuracy of 43.8%, i.e. "indistinguishable" from real seals.

Its `T###` sign IDs match no numbering available to us. Sign `740` does not exist in it, and frequencies of shared numbers are uncorrelated (rank correlation 0.09). We therefore compared it with real text using **statistics that do not depend on numbering** (`synthetic_check.py`; figure `synthetic_vs_real.png`):

| | Real (M3 corpus) | Synthetic | Real, per the synthetic set's own index |
|---|---|---|---|
| Mean length (signs) | 4.12 | **7.53** | 4.06 |
| Inscriptions with ≥ 10 signs | 2% | **24%** | — |
| Distinct signs used | 686 | **170** | 641 |
| Top-10 signs' share | 31.4% | **52.7%** | — |
| Adjacent identical signs | 0.4% of pairs | **25.5%** | — |
| Inscriptions containing a doubled sign | 1.1% | **68.0%** | — |
| Unigram entropy (bits) | 7.05 | 5.37 | — |
| Conditional entropy (bits) | 3.61 | 2.47 | — |

**Within its own numbering**, some signs are heavily over-used. For example, T283 makes up 6.8% of synthetic usage but only 1.45% of real usage according to the corpus's own index. Runs such as `638 789 123 123 123 123 123 609` (3,615 runs of three or more identical signs) are the typical failure of a small generator stuck in a loop.

The corpus's claim of indistinguishability fails a check simple enough to state in one sentence: real inscriptions almost never repeat a sign twice in a row, and 68% of the synthetic ones do.

**Caveat:** if that numbering writes numerals as repeated single strokes, part of the repetition could be an encoding choice. The length gap, the small vocabulary and the per-sign over-use do not depend on this.

---

## 9. Evaluating a proposed reading (E5)

The corpus CSV includes a published Sanskrit reading for 1,763 inscriptions (columns `sanskrit`, `translation`). It appears to come from Yajnadevam's 2024 cryptanalytic decipherment (published at indusscript.net). The CSV itself does not say, so confirm this before submission. The field does not accept this reading. We treat it purely as a **claim to be tested** and make no statement about its content.

**What could be tested without the decipherment's own sign-value key** (`f5_reading_test.py`):

| Test | Result |
|---|---|
| **T1 Consistency:** do identical sign sequences get identical readings? | Yes. 4 conflicts among 1,749 distinct texts. |
| **T2 Phonetic anchoring** (multinomial Naive Bayes, 5-fold CV, n = 1,666): predict the reading's first sound from … | |
|   … the **first sign read** (reading-initial) | **70.8%** |
|   … the last sign read (wrong end; structural control) | 21.8% |
|   … a random sign of the inscription (structural control) | 22.7% |
|   … the first sign read, with readings shuffled across inscriptions (null, 200 permutations) | 18.0% (95th percentile 19.2%); **p < 0.005** |
|   Majority-class baseline | 20.3% |
| **T3 Freedom per sign:** distinct first sounds per sign (16 signs with ≥ 20 occurrences) | Median 2. Most extra values are short/long or aspirated variants (a/ā, d/dh). True alternative values appear for `032` (s / ś / h) and `803` (ś / s / h). Three distinct signs (`817`, `820`, `861`) all read as *r*. |

**Interpretation, stated precisely:**

- The reading is **not arbitrary in the sense of being inconsistent**. It is a systematic sign-to-sound key, applied in right-to-left reading order. That agrees with the direction evidence in Section 2.3.
- **Consistency is required of any cipher key and says nothing about whether the key is correct.** A random but fixed assignment of sounds to signs would pass T1 and T2 just as well.
- The key has measurable freedom: some signs carry several sounds, and several signs share one sound. Every such degree of freedom makes it easier to fit an arbitrary target language.
- One observation for further study: the three signs read as *r* (`817`, `820`, `861`) are also the ones Milestone 3's model confuses most often with each other. The key gives the same value to signs that occur in the same contexts.

**What was not done, and why.** The decisive falsification test is to apply the key to real inscriptions and to inscriptions whose sign values have been shuffled, then score both against a Sanskrit lexicon. If shuffled keys produce "valid Sanskrit" equally often, the method cannot tell text from noise. This test needs the key in machine-readable form and a lexicon. The key is published only as a table in the paper, with its own sign numbering, so we could not run the test in time.

An independent, pre-registered test bench (Savirinc, `github.com/Savirinc/indus-script`) reports running exactly this test. It places this key's readings at the 0.29 percentile of shuffles of its own values. **We have not reproduced that result**, and we cite it as external work only.

**Conclusion for E5:**

- Our data **do not disprove** the reading.
- They show it is internally systematic, quantify its freedom, and identify the one test that would decide the question.
- Claiming more would itself be a semantic hallucination.

---

## 10. How digitization errors propagate (E6)

The proposal's learning outcome: *"analysing how image digitization errors propagate down into downstream sequence analysis layers."*

**Method.** We took the 308 clean held-out inscriptions and corrupted them with an **M1-shaped error model** at strength *s* (`f6_error_propagation.py`; figure `f6_error_propagation.png`). At *s* = 1 the rates equal Milestone 1's measured performance:

- substitution: 83% (M1 top-1 accuracy is 17%)
- deletion: 25% (4.9 detected × 86% precision = 4.2 true hits out of 5.6 true signs)
- insertion: 12% per true sign (4.9 × 14% false boxes = 0.69 extra per text)

*Caveat on these rates.* They mix two hand reviews. Both reviews judged the same detector, and its precision came out as 77% in one sample and 86% in the other (pooled 81%), so that difference is sampling noise. The 17% top-1 comes from the baseline recognizer, but its weights were overwritten; the saved recognizer used in E7 and the engine scores 12% top-1. We keep M2 Step 7's derivation (86%, 17%) so that E6 and M2 stay comparable. With the saved model's numbers, *s* = 1 would correspond to slightly heavier corruption, so the s = 1 row slightly *understates* the damage.

These are the same rates Milestone 2 derived in its Step 7. M4 extends that study in two ways: it measures the effect on M3 and on the guardrail, and it draws misreads from M1's **actual** output distribution instead of random signs.

Substituted and inserted signs are drawn from **what M1 actually outputs** on the 77 real strips, converted through the crosswalk, so M1's bias toward particular templates is kept. Each strength was run 3 times.

| Error strength *s* | Direction detected (M2) | KL from clean transitions | M3 top-1 on untouched signs | Guardrail flags |
|---|---|---|---|---|
| 0 (clean) | 0.895 | 0.000 | 0.342 | 6.2% |
| 0.2 | 0.833 | 0.211 | 0.285 | 17.1% |
| 0.4 | 0.753 | 0.369 | 0.227 | 31.8% |
| 0.6 | 0.683 | 0.529 | 0.172 | 44.1% |
| 0.8 | 0.595 | 0.688 | 0.132 | 61.1% |
| **1.0 (M1 today)** | **0.538** | 0.814 | **0.079** | **69.9%** |

"Direction detected" is the share of inscriptions whose true order is more probable than its reverse under the clean bigram; 0.5 means the direction is undetectable.

**What this shows:**

- At M1's current error level, the fixed-direction signal Milestone 2 relies on is almost gone (54% vs. 50% chance). This matches M2's Step 7, which detected direction in only about 10% of simulated runs at 17% accuracy.
- The language model loses about three-quarters of its accuracy (34% → 8%) even on signs that were read correctly, because their context is corrupted.
- Any statistical claim made from automatically digitized seals today would be largely a claim about M1's errors.
- **Consistency check:** the error model predicts a 70% flag rate at *s* = 1, and Section 11 measures 74% on M1's real output. The simulation is close to reality, and if anything slightly optimistic.

---

## 11. The guardrail applied to our own pipeline (E7)

This experiment connects M1, the crosswalk, M3 and M4 (`m1_outputs.py`, `f7_guardrail_on_m1.py`; figure `f7_guardrail_on_m1.png`).

**Setup.** Milestone 1 read all 77 inscription strips, giving 312 signs. Of these, 57 signs (18%) were IM77 signs with no Wells equivalent and became gaps, and 79 mappings were ambiguous. 54 strips kept at least 2 readable signs and were scored.

| M1 output | Median pseudo-perplexity | Flagged by guardrail |
|---|---|---|
| M1 top-1 reading | 1,454 | **74%** |
| M1 candidates re-ranked by M3 context (158 signs changed) | 378 | **35%** |
| *For reference: real held-out text* | 19.6 | 6.2% |
| *For reference: frequency-random noise* | 504 | 44.5% |

**What this shows:**

- **The guardrail correctly rejects M1's output.** To the language model, M1's readings look like noise, somewhere between frequency-random and uniform-random. This is consistent with M1's measured 17% top-1 accuracy.
- **Re-ranking by M3 is a semantic hallucination produced inside our own pipeline.** We let M3 choose among M1's top-5 visual candidates using a combined score: log visual score + log M3 context probability. The output then looks far more like real text (flag rate 74% → 35%). But M1's top-5 contains the true sign only about 41% of the time, and we have no per-strip ground truth. **Nothing shows the re-ranked text is more correct. It is only more plausible.** The language model has filled in a convincing reading of signs the vision system could not read.

This is the general hazard the proposal warns about, measured inside our own system: fluency and plausibility are not evidence of a correct reading.

---

## 12. Generating Indus-like text (E8)

Your professor asked for "an algorithm that can generate some form of the Indus script" (`f8_generate.py`). We built two generators, both trained on the train split:

- **Markov:** samples from the Witten–Bell bigram, from start to end.
- **Gibbs:** starts from a Markov sample, then runs 10 sweeps, each time hiding one sign and resampling it from the M3 ensemble.

Each produced 1,000 sequences, judged exactly like real text:

| | Real (held-out) | Markov | Gibbs (M3) |
|---|---|---|---|
| Median pseudo-perplexity | 19.6 | 38.3 | **18.2** |
| Guardrail flag rate | 6.2% | 5.7% | 2.6% |
| Mean length | 4.59 | 4.41 | 4.41 |
| Distinct signs | 255 | 355 | 361 |
| Adjacent repeats | 1.5% | 0.7% | 1.4% |
| **Exact copies of a training inscription** | 0.0%* | **16.1%** | **23.0%** |
| Naive Bayes AUC vs. real (5-fold) | — | 0.558 | 0.663 |

\* Real test inscriptions that equal a training inscription. Before the Milestone 3 deduplication fix this was 4.2%; it is now 0%, confirming the leak is closed.

**What this shows:**

- Both generators **pass the guardrail**. The Gibbs output even scores as *more* real than real text (18.2 vs. 19.6), because M3 is judging its own samples.
- **A model judging its own samples is not an independent judge.**
- Naive Bayes still separates the Gibbs output weakly (AUC 0.66).
- Almost a quarter of the Gibbs outputs are verbatim training inscriptions. A generator's "novel" output must always be checked for memorization.

**Honest framing:** we can generate sequences that match the corpus's statistics to a measured degree. That says nothing about meaning. A generator that passes a statistical guardrail has learned the statistics the guardrail checks, not the language.

### 12.1 Novelty filter and image round trip (E8b)

`src/engine/generate_and_read.py` takes the same 2,000 generated sequences further:

1. **Novelty.** Each sequence is compared with every training inscription. A *copy* is identical; a *near-copy* is one sign substituted, inserted or deleted; anything else is *novel*.
2. **Acceptance.** A sequence is accepted if it is novel and passes the guardrail.
3. **Round trip.** 60 accepted sequences (30 per generator) are drawn as synthetic carved strips with the Milestone 1 renderer and read back by the full engine.

| | Markov | Gibbs (M3) |
|---|---|---|
| Exact copies of a training inscription | 16.1% | 23.0% |
| Near-copies (one edit away) | 32.2% | 31.2% |
| **Novel** | 51.7% | 45.8% |
| Novel *and* passes the guardrail (accepted) | **49.9%** | **45.1%** |
| Accepted texts found in the validation/test splits | 0% | 0% |
| Median pseudo-perplexity of accepted texts (real: 19.6) | 45.9 | 27.2 |

Round trip, 60 images:

- The sign count is right on 90% of the images.
- On those, 91% of signs and 67% of whole sequences are read back exactly (at the Mahadevan level).
- Only 37% match at the Wells level. Several Wells signs are variants drawn with one Mahadevan template (e.g. 817/861), so an image cannot tell them apart.
- 98% of the read-backs pass the guardrail, **including 96% of the misread ones**.

**What this shows:**

- **About half of what the generators produce is genuinely new.** Another 30% is only one edit away from a training text, so "novel" needs a definition, and exact-match checking alone understates memorisation.
- **The Gibbs generator's accepted texts are closer to real text than the Markov ones** (27 vs 46 against 19.6), even after removing copies.
- **The guardrail does not catch single-sign misreadings.** A plausible text with one wrong sign stays plausible. The guardrail detects noise (E7), not small errors. This is the same lesson as E7's re-ranking result, seen from the other side.
- **The round-trip accuracy is an upper bound.** The images come from the renderer family M1 was trained on; on real photographs M1 reads 12–17% of signs correctly.

---

## 13. Synthesis: confidence is not understanding

The same pattern appears in every part of the project:

| System | Confident number | What the controls showed |
|---|---|---|
| M1 recognizer | 87.9% top-1 on synthetic data | 17% on real strips; its output looks like noise (E7) |
| M3 confidence ratings | ECE 0.035, "High" = 75% correct | "High" on 8% of pure noise, 0% correct (E2) |
| Bigram judge | Scores Markov fakes the same as real text | Needs an outside judge (E1) |
| Bigram perplexity as direction evidence (M2) | Looks like a model score | Identical in both directions on training data; only a 3% gap on held-out data (Section 2.4) |
| Synthetic corpus | 85–98% ensemble scores, "indistinguishable" | 68% contain a doubled sign vs. 1.1% of real ones (E4) |
| M3-guided re-ranking | Flag rate falls from 74% to 35% | Plausible, not shown to be correct (E7) |
| Gibbs generator | More "real" than real text by M3's own score | 23% memorized; detectable by Naive Bayes (E8) |
| Sanskrit reading | Covers 1,763 inscriptions | Systematic, but the decisive test is pending (E5) |

On a short, low-resource, undeciphered corpus, a claim is worth exactly as much as its margin over structural controls. Every number above that lacks such a margin is a candidate semantic hallucination.

---

## 14. Limitations (stated before anyone else finds them)

1. **Milestone 3 split.** The duplicate leak has been fixed (v2: deduplication on the sign sequence, so no sequence appears in two splits) and every number in this report was regenerated on the new split. M4's experiments use M3's single train/validation/test split. Milestone 3 itself is now cross-validated (5 folds): the 3-model ensemble beats the bigram by a small, consistent margin (top-1 33.3% vs 32.5%, perplexity 28.8 vs 31.8), while a single Transformer does not. The M4 experiments themselves have not been repeated across folds.
2. **Uncertainty is reported** as bootstrap 95% intervals (perplexity medians, guardrail AUCs), Wilson intervals (flag rates) and cross-validation standard deviations (Naive Bayes, ≤ 0.02). The synthetic-corpus statistics (E4) and the generator table (E8) are single runs without intervals.
3. **One Witten–Bell implementation** (`src/common/witten_bell.py`, Milestone 2's definition) is now used by M2's replication, M3's bigram baseline and M4's Markov controls.
4. **The crosswalk is many-to-many for about 22% of corpus occurrences**, and 18% of M1's predicted signs have no Wells equivalent.
5. **E5's decisive test was not run** (Section 9).
6. **E7 has no per-strip ground truth.** "More plausible" is measured; "more correct" is not.
7. **The guardrail is built from M3**, so it cannot independently judge text generated by M3 (E8).
8. **The synthetic corpus uses an unknown numbering**, so only numbering-free comparisons were possible (E4).
9. **Sample sizes:** 308 test inscriptions per set (E1, E2, E6), 3,047 per class (E3), 54 scored strips (E7), 1,666 readings (E5), 16 signs (E5-T3).

---

## 15. Reproducing every number

Run from the project root, after Milestone 3's `artifacts/` exist:

```bash
python src/common/signs.py                         # crosswalk + 91.8% validation
python src/m2_markov/replicate_wells.py            # M2 reproduced on CISI + replicated on the Wells corpus
python src/m4_guardrails/controls.py               # structural controls
python src/m4_guardrails/perplexity_bench.py       # E1, E2
python src/m4_guardrails/nb_guardrail.py           # E3
python src/m4_guardrails/synthetic_check.py        # E4 (needs data/raw/synthetic/)
python src/m4_guardrails/f5_reading_test.py        # E5
python src/m4_guardrails/m1_outputs.py             # M1 on all 77 strips (needs ultralytics + M1 weights)
python src/m4_guardrails/f6_error_propagation.py   # E6
python src/m4_guardrails/f7_guardrail_on_m1.py     # E7 (+ guardrail calibration table)
python src/m4_guardrails/f8_generate.py            # E8
python src/engine/generate_and_read.py             # E8b: novelty filter + render + read back
```

| File | Role |
|---|---|
| `src/common/signs.py` | IM77 ↔ Parpola ↔ Wells crosswalk and its validation |
| `src/m2_markov/` | Milestone 2 (notebook, outputs) + `replicate_wells.py` |
| `src/m4_guardrails/guardrail.py` | Shared scorer: pseudo-perplexity, threshold, verdict, M1 → Wells conversion |
| `controls.py` | Structural controls |
| `perplexity_bench.py` | E1 / E2 |
| `nb_guardrail.py` | E3 |
| `synthetic_check.py` | E4 |
| `f5_reading_test.py` | E5 |
| `m1_outputs.py` | M1 readings with top-5 candidates |
| `f6_error_propagation.py` | E6 |
| `f7_guardrail_on_m1.py` | E7 |
| `f8_generate.py` | E8 |
| `src/engine/generate_and_read.py` | E8b |

---

## Sources

- Proposal: *Indus Epigraphy AI Project Proposal*, Milestone 4 section.
- Milestone 2 report: `reports/m2/summary/Milestone_2_Detailed_Summary.pdf` (Steps 2, 5 and 7 are used here).
- Corpus of Indus Seals and Inscriptions digitization (Parpola et al.), including the grapheme crosswalk in `features/`.
- `hellosindh/indus-script-synthetic` (CC-BY-4.0), Hugging Face.
- Yajnadevam (2024), *A cryptanalytic decipherment of the Indus script*: https://rarebooksocietyofindia.org/book_archive/A_cryptanalytic_decipherment_of_the_Indu.pdf
- Savirinc, *indus-script* pre-registered test bench (external, not reproduced here): https://github.com/Savirinc/indus-script
