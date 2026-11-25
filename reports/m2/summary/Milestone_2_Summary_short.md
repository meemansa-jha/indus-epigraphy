# Milestone 2 — Positional Bias & Markov Chain Modelling
### Indus Epigraphy Project · summary of all work, results and findings

> **For the deck:** the story is **"the script has a fixed direction and real positional
> syntax, and we proved it with tests that cannot be fooled by chance or by small
> samples."** Every claim is checked against a random baseline. One key lesson
> is a trap we found and avoided: **the textbook direction test (forward vs backward
> model perplexity) is mathematically blind to direction.** That lesson, together with
> Step 7's error-propagation result, sets up Milestone 4 directly.

---

## 1. What Milestone 2 asked for

From the project proposal:

- **Objective:** establish the direction of writing and calculate sign-pairing constraints.
- **Technical task:** map sequential transition frequencies across the corpus; construct a Markov chain matrix; evaluate directional asymmetries by comparing text beginners with text enders; apply **Witten-Bell smoothing** to handle unobserved pairings.
- **Deliverable:** *"A directional statistical matrix verifying right-to-left syntax rules and tracking sign collocation indices."*
- **Cross-cutting requirement:** *"analyzing how image digitization errors propagate down into downstream sequence analysis layers."*

Throughout, a sign is an **opaque label**. We never claim what any sign means.

---

## 2. Data and the decisions that shaped it

**Corpus:** the digital CISI corpus (`mayig/indus-valley-script-corpus`, Parpola numbering): **179 inscriptions, 984 signs, 181 distinct signs**, all from Mohenjo-daro (M-1 to M-199). It is small, so every result is reported against a random baseline.

The full IM77 concordance (indusscript.in) was checked, but no public export of its texts could be found, so CISI, which the proposal itself lists, is the corpus.

Three decisions were made once, in Step 0, and applied everywhere:

| Decision | Why |
|---|---|
| **Reverse every text into right-to-left reading order** | The corpus stores signs left-to-right as they appear on the impression. |
| **`P000` (lost material, 19 cases in 18 texts) is a break, not a sign** | Deleting it would create pairs that were never seen. Counting it would make "damage" look like the 4th most common ender. Edges next to damage are not used as genuine beginnings or endings. |
| **Line breaks (7 two-line texts) are treated like breaks** | We cannot be certain how two lines join. |

**Usable evidence:** 796 sign pairs, 166 genuine beginnings, 161 genuine endings, 154 complete single-line texts.

### ⚠ Correction to the Milestone 1 summary
M1 §11 says `P324` *opens* 77 inscriptions and `P385` *closes* 29. Those counts were taken on the corpus's stored (left-to-right) order. **In reading order it is the reverse: `P324` is the dominant *ender* and `P385` the dominant *opener*.** Please update the M1 slides so that the two milestones agree.

---

## 3. Method — the eight steps

| Step | What it does | Key output |
|---|---|---|
| 0 | Load, reverse to reading order, cut at damage/line breaks, verify | `segments_rtl.json` |
| 1 | Sanity check against M1 and Mahadevan | reconciliation table |
| 2 | Beginners vs enders + **flip test** (10,000 random-direction versions) | `m2_03`, `m2_04` |
| 3 | Transition counts → Markov matrix with START/END markers | `markov_counts.csv`, `markov_mle.csv` |
| 4 | **Witten-Bell smoothing**, validated by 10-fold cross-validation | `markov_witten_bell.csv` |
| 5 | Direction: the perplexity trap, positional entropy test, **directional rules** | `directional_rules.csv` |
| 6 | **Collocation index** (PMI + frequency floor + Fisher test) and pair-order asymmetry | `collocations.csv` |
| 7 | **Error propagation**: M1's measured errors injected into clean sequences | `m2_09` |

All code is in one notebook (`m2_markov.ipynb`). It runs end to end with built-in assertions and is deterministic.

---

## 4. RESULTS — the key tables for the deck

### 4a. Sanity check — the cleaned data matches the known corpus

| Measure | M1 report | Cleaned (M2) | Note |
|---|---|---|---|
| Mean / median length | 5.60 / 5 | **5.57 / 5** | M2 uses complete texts only |
| Distinct signs | 182 | **181** | `P000` is no longer counted as a sign |
| Top-67 coverage | 83.4% | **83.4%** | Mahadevan: "over 80 per cent" |
| Signs covering 50% / 80% | — | **17 / 58** | |
| Signs seen exactly once | — | **77 (43%)** | This is why smoothing is needed |

### 4b. Beginners vs enders — the direction fingerprint

| | Beginnings | Endings |
|---|---|---|
| Distinct signs used | **86** | **30** |
| Top sign | `P385`, 17.5% | `P324`, **45.3%** (95% CI 38–53%) |
| Entropy | 5.63 bits | **3.22 bits** |

**Flip test:** beginner–ender divergence (JSD) = **0.89 bits**. Across 10,000 random-direction versions, the maximum was **0.42**, so **p < 0.0001**.

> Texts open in many ways but close in a few fixed ones. The statistics prove the direction is **fixed**. *Which* end is the start comes from physical evidence (e.g. cramped signs at the left edge); every result here is consistent with right-to-left reading.

### 4c. The Markov matrix and Witten-Bell smoothing

| Matrix property | Value |
|---|---|
| Cells (182 × 182) | 33,124 |
| Pairs ever observed | **645 (1.95%)** |
| Rows built from one observation | **73** |
| Rows where unsmoothed MLE is "100% certain" | **84** |

**Validation on held-out inscriptions (10-fold CV):**

| Method | Held-out pairs given P = 0 | Perplexity |
|---|---|---|
| MLE (no smoothing) | **516 of 1,123 (46%)** | ∞ (fails) |
| Add-one (Laplace) | 0 | 77.3 |
| **Witten-Bell** | **0** | **40.8** |

Witten-Bell reserves $T/(C+T)$ of each row for unseen successors: 50% for signs seen once, 12–13% for well-attested, consistent signs (`P324`, `P385`), and 32% for the highly varied `P122`.

### 4d. The perplexity trap (an honest negative result)

| | Right-to-left | Left-to-right |
|---|---|---|
| Training data (bits/transition) | 2.5861 | **2.5861, identical** |
| Held-out perplexity | 40.43 | 40.39 (0.1% difference = noise) |

A bigram model's likelihood depends only on pair counts and sign counts, and reversing a text preserves both. **The overall model score is mathematically blind to direction.** Any claim of direction based on it is empty. This goes on a slide and links straight to M4.

### 4e. The direction test that works — positional entropy

| Position from edge | From START | From END | Random-direction range |
|---|---|---|---|
| 1 | 5.31 | **2.94** | 4.36–5.04 |
| 2 | **4.80** | 5.60 | 5.25–5.68 |

The last sign is far more predictable than chance allows. The opening's structure sits one position in (`P385 → P122`). Position-1 gap = 2.37 bits against a random maximum of 1.29, **p < 0.0001**.

### 4f. Directional rules (significant after Benjamini–Hochberg correction, q < 0.05)

| Role | Signs | Example |
|---|---|---|
| **Ender** | `P324`, `P217`, `P086`, `P013` | `P324` ends 79% of its appearances (chance level = 18%) |
| **Opener** | `P385`, `P378`, `P256` | `P385` opens 86% of its appearances |
| **Medial** | `P122`, `P050`, `P120`, `P062`, `P060` | `P122`: **0 of 65** at an edge (~23 expected) |

### 4g. Collocations

**529** distinct pairs → **48** seen ≥ 3 times → **41 significant collocations**.

| Pair (reading order) | Seen | Expected | ≈ × chance |
|---|---|---|---|
| `P316 → P147` | 10 | 0.28 | 36× |
| `P324 → P013` | 9 | 0.24 | 38× (an alternative two-sign ending) |
| `P385 → P122` | 28 | 3.03 | 9× (most significant, q ≈ 10⁻²⁵) |
| `P060 → P062` | 9 | 0.53 | 17× |
| `P332 → P324` | 10 | 1.22 | 8× (10 of 10) |

- **Pair order:** **17 of 22** frequent pairs never occur reversed (`P385 → P122`: 28 vs 0). This is further evidence of a fixed direction.
- **Why the safeguards matter:** the 421 one-off pairs have a *higher* median PMI (3.66) than the real collocations (3.27). Raw PMI rewards rarity.

### 4h. Error propagation — what M1's errors would do

Derived from M1's measurements: **25% missed signs, 12% extra boxes**, and recognition accuracy swept from 100% down to M1's **17%** (50 simulations per level).

| Finding | Clean | At M1's 17% accuracy |
|---|---|---|
| `P324` share of endings | 44% | **6%** |
| Fixed direction detected | 100% of runs | **10%** |
| Start–end gap | 2.5 bits | **0.1 bits** |
| Collocations recovered | 100% | **0%** |

| Finding | First lost once accuracy drops to |
|---|---|
| Fixed direction detected | 45% (most robust) |
| `P324` as dominant ender / gap ≥ 1 bit | 75% |
| Half of the collocations | already at 100% (most fragile) |

> **Surprise worth a slide:** even with *perfect* recognition, M1's detection errors alone cut collocation recovery to **44%** and halve the start–end gap. Detection that is "solved" for finding signs is **not** good enough for sequence statistics, because every missed or extra box breaks the pairs around it.

---

## 5. Honest negative results and limitations

- **The perplexity trap.** Forward and backward models score identically; direction cannot be inferred from overall perplexity.
- **Raw PMI is misleading** on small corpora. It needs a frequency floor and a significance test.
- **Small corpus.** 179 texts from one site. Results carry confidence intervals and random baselines, and 16 of 28 frequent signs show no clear positional preference simply for lack of data.
- **Bigram model only.** Longer-range structure (e.g. the `P385 → P122` opening formula as a unit) is left to Milestone 3's sequence models.
- **No external corpus.** The IM77 data could not be exported. A larger corpus (`yajnadevam/indus-website`, ~2,500 texts, its own numbering) exists and could serve as an independent replication.
- **Error model assumptions (Step 7).** Misreads are modelled as random signs and reading order as correct. M1's rates carry a ±13 pp margin.

---

## 6. Links to Milestones 3 and 4

- **M3 (masked token prediction):** the Witten-Bell matrix is a ready baseline predictor. The **19 `P000` gaps** are natural test cases for "predict the missing sign". The perplexity of 40.8 is the bar a neural model must beat.
- **M4 (benchmarking hallucinations):** two ready-made exhibits:
  1. the perplexity trap, a confident-looking metric that cannot answer the question asked;
  2. error propagation, where a pipeline reporting results at 17% accuracy would be "finding" structure that is mostly noise.

---

## 7. Figures for the deck

All figures are in `outputs/figures/`, numbered in suggested order.

### Essential — these carry the story

| File | Slide | What it shows |
|---|---|---|
| `m2_03_start_vs_end.png` | Signs have fixed positions | Frequent signs keep to one end: `P324` ends texts, `P385` opens them, `P122` never touches an edge. |
| `m2_04_flip_test.png` | Direction is not chance | The real divergence (0.89) against 10,000 random-direction versions (max 0.42). |
| `m2_07_positional_entropy.png` | Start and end are structured differently | The end is highly predictable; the start's structure sits one position in. |
| `m2_06_witten_bell_reserve.png` | Why and how we smooth | Thin or varied evidence → more probability held back. Pair it with the CV table (4c). |
| `m2_09_error_propagation.png` | How M1 errors propagate | All four findings collapse at 17% accuracy, and detection errors alone already hurt. |

### Strongly recommended

| File | Slide | What it shows |
|---|---|---|
| `m2_05_markov_matrix_mle.png` | The Markov matrix | Mostly empty (98%), a few strong rules, and an END column that lights up for enders. |
| `m2_08_collocations_pmi.png` | Real collocations vs accidents | Raw PMI is highest for one-off pairs; real collocations need frequency + significance. |

### Optional

| File | Slide | What it shows |
|---|---|---|
| `m2_01_inscription_length.png` | The corpus | Mean 5.6 signs; most texts have 3–7. |
| `m2_02_sign_coverage.png` | A small core dominates | 58 signs = 80% of all occurrences. |

### If you only have room for three
**`m2_04`** (direction is real) + **`m2_03`** (which signs and where) + **`m2_09`** (why we used expert transcriptions).

Also add the **perplexity-trap table (4d)** as a text slide. It's the most transferable lesson in the milestone.
