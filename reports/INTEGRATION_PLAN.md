> **Status (updated):** this plan was written before Milestone 2 arrived and before the fixes below. Done since:
> M2 integrated and replicated; crosswalk built and validated; M1 §11 direction corrected; M3 deduplication leak fixed,
> real model search, learning curves for all seeds, 5-fold cross-validation; `predict.py` numbering/order conversion;
> M4 experiments E1–E8 run and the report written; the parsing engine built; README, DATA_SOURCES.md, run_all.sh,
> requirements and .gitignore rewritten; M4 error bars; a single Witten–Bell implementation; a shared corpus loader;
> four milestone notebooks; the M1 README rewritten; `run_all.sh` tested end to end (all results reproduce; M2's
> tie-ordering made deterministic). Still open: Milestone 1 items, and the E5 sign-value key. Items below are kept as the original audit.

# Indus Epigraphy AI — Audit and Integration Plan (Milestones 1–4)

This document has two jobs:

1. An honest audit of the current project. Everything listed was checked against the files in `~/indus-epigraphy` and the original proposal, not from memory.
2. A step-by-step plan that turns four separate milestones into one integrated project, with Milestone 4 as the place where everything meets.

Every step ends with a **"Done when"** check, so you know exactly when it is finished.

Severity labels used throughout:

- 🔴 **MUST FIX**: an examiner who finds this can mark you down.
- 🟠 **SHOULD FIX**: weakens the project, but you can survive it if you disclose it.
- 🟡 **DISCLOSE**: not fixable in time; say it in the report before someone else does.

---

## Part 0 — The blunt summary (read this first)

These are the problems that matter most, in order:

1. 🔴 **Milestone 2 is not in the project folder.** There is no `src/m2_markov/`, no Markov code, and no notebook anywhere in `~/indus-epigraphy`. You said all milestones are complete, so it must be on a teammate's machine. I cannot audit what I cannot see. Integration cannot start until it is copied into `src/m2_markov/`. Part 2.2 lists exactly what it must contain.

2. 🔴 **Three different sign-numbering systems are in use, and the integration silently breaks without a fix.**
   - M1 outputs IM77 numbers (1–417, Mahadevan).
   - The CISI corpus uses Parpola numbers (`P###`).
   - The M3/M4 corpus CSV uses **Wells numbers**. I verified this: its jar sign `740` is Wells W740 = Parpola P324 = IM77 342.
   - Example of the silent failure: `predict.py` would accept M1's output `342` (the jar) and treat it as Wells sign 342, which is a different sign. **No error is raised. The answer is just wrong.**
   - **Good news:** a crosswalk already exists in your data folder. `data/raw/cisi/features/P000–P396.json` lists the Wells and Mahadevan equivalents of every Parpola sign. I validated it by aligning the 148 inscriptions that appear in both CISI and the M3 CSV: **91.8% of 785 sign slots agree**. I also confirmed visually that IM77 templates 342, 267, 59, 87, 176 and 328 are exactly the signs the crosswalk says they are.
   - This also **unblocks** the M1 experiment that the M1 summary calls "blocked on data".

3. 🔴 **Reading direction and storage order are confused in at least two documents.**
   - Both the CISI JSON and the M3 CSV store signs **left-to-right as they appear on the impression**. The script reads **right-to-left**, so the **first stored sign is the last one read**.
   - Proof: the jar sign, a well-known text-*ender*, is stored first in 71% of R/L inscriptions and stored last in only 3%.
   - The CSV has already normalised the 215 L/R inscriptions to the same orientation (jar stored first 60%, last 8%).
   - Consequences:
     - (a) `MILESTONE1_SUMMARY.md` §11 says "P324 **opens** 77 of 179 inscriptions; P385 **closes** 29". This is backwards. P324 is the jar and *ends* them. That was my error; fix it.
     - (b) The M3 README says "first position 44%, last 17%". In reading terms that is reading-**final** 44% and reading-**initial** 17%.
     - (c) M1 outputs signs in reading order (right-to-left), which is the reverse of what M3 expects. Feeding M1 output into M3 without reversing it is wrong.
     - (d) **If M2 treated stored-first as "text beginner", M2's directional conclusions are inverted.** Check this the moment M2 arrives.

4. 🔴 **Milestone 3 has four methodological flaws an examiner can find:**
   - Duplicates were removed including edge markers, so 37 test inscriptions (9.9%) have identical signs to a training inscription.
   - The hyper-parameter "search" in `run_all.py` is hard-coded numbers, not code that runs.
   - Learning curves exist for one seed only.
   - **Running `run_all.py` now would overwrite `figures.png` with an empty learning-curve panel**, because it loads checkpoints and keeps no training history.
   - Fix all four and retrain. Then **every M4 number must be regenerated**, because M4 is built on M3's split.

5. 🔴 **Required deliverables from the proposal do not exist yet:**
   - the unified "Epigraphic Parsing Engine" (one script, image to n-gram report)
   - documented training notebooks (the `notebooks/` folder is empty)
   - cross-validation results
   - the error-propagation study
   - the M4 research report

   Your professor's request for "an algorithm that generates the script" also has no proper home yet.

6. 🟠 **Milestone 1 documentation is stale and has one unhandled assumption.**
   - `src/m1_vision/README.md` still describes the September design (template matching only, no CNN, no strips, no denoiser).
   - No code handles mirror images. A photo of an actual seal, as opposed to its impression, shows the text mirrored and in reverse order. I tested the 77 strips: mirroring each of 262 detected crops made template matches better 89 times and worse 94 times. **So there is no evidence the strips are mirrored, but the assumption must be documented**, and the seal-photo path needs a `--mirror` option.
   - `data/raw/real_seals/` still contains the 3 non-seal images: the Constitution page, the Brahmi-comparison image, and the 2025 "falsifiability sheet".

7. 🟠 **Data provenance gaps.**
   - The M3 CSV (5,679 rows, Wells numbering, with `sanskrit`/`translation` columns from an unaccepted decipherment) is **not one of the proposal's listed datasets**, and nowhere in the project says where it came from. Cite it or you will be asked.
   - Two proposal resources were never used: the RMRL IM77 web platform and the AI-EPIGRAPHY toolkit. One sentence each in the report must say why.

8. 🟠 **Biggest schedule risk: M4 experiment E5** (testing the proposed Sanskrit reading). It needs the decipherment's sign-to-sound table and a Sanskrit word list, and you have neither. Start sourcing them on day 1. If you cannot get them, Part 5 gives a fallback.

---

## Part 1 — What the proposal requires vs. what exists

| Proposal requirement | Where it lives / will live | Status |
|---|---|---|
| **M1** image → ordered list of standardized sign IDs | `src/m1_vision/read_strip.py`, `read_seal.py` | ✅ exists; 17% top-1 / 41% top-5 on real strips (honest, documented) |
| **M2** Markov transition matrix | `src/m2_markov/` | ❓ not in folder |
| **M2** directional asymmetry (beginners vs. enders) | `src/m2_markov/` | ❓ not in folder; see order warning (Part 0 #3) |
| **M2** Witten–Bell smoothing | `src/m2_markov/` (also re-implemented in `src/m3_masking/baselines.py`) | ❓ / duplicate implementations |
| **M2** collocation indices | `src/m2_markov/` | ❓ |
| **M2** "verifying right-to-left syntax rules" | `src/m2_markov/` | ❓; also needs careful wording (Part 6) |
| **M3** masked-token encoder (TinyBERT-style) | `src/m3_masking/` | ✅ with 4 flaws to fix (Part 2.3) |
| **M3** context-aware completion + confidence ratings | `src/m3_masking/predict.py` | ✅ but rejects nothing on wrong numbering (Part 0 #2) |
| **M3** entropy / perplexity | `results.json` (perplexity); bits not reported | 🟠 add bits/sign |
| **M4** multinomial Naive Bayes or controlled verification harness | `src/m4_guardrails/nb_guardrail.py` | ✅ done (4 controls) |
| **M4** plot generative perplexities vs. random-noise baselines | `src/m4_guardrails/perplexity_bench.py` | ✅ done; must rerun after M3 retrain |
| **M4** evaluate proposed readings against structural controls | not started | 🔴 E5 |
| **M4** confidence-scoring metrics vs. controls | `perplexity_bench.py` (confidence check) | ✅ done |
| **M4** empirical research report | not started | 🔴 |
| Learning outcome: "classifiers to isolate synthetic reconstructions from historical replicas" | `src/m4_guardrails/synthetic_check.py` | ✅ numbering-free audit done |
| Learning outcome: "how image digitization errors propagate downstream" | not started | 🔴 E6 |
| **Unified "Epigraphic Parsing Engine"** (image → visualization panels → positional n-gram probability report) | not started | 🔴 Part 4, Phase G |
| **Notebooks** with cross-validation, learning curves, token-prediction loss | `notebooks/` is empty | 🔴 Phase H |
| Professor: "an algorithm that generates some form of the script" | not started | 🟠 E8 |

---

## Part 2 — Audit, milestone by milestone

### 2.1 Milestone 1 (vision)

**What is solid:**

- Detection on strips: 99% coverage, 86% precision.
- The synthetic pipeline.
- The four-attempt recognition ablation.
- The collapse check.
- The honest 17% / 41% headline.

**Problems:**

1. 🔴 **`MILESTONE1_SUMMARY.md` §11 inverts direction** ("P324 opens …"). Rewrite it as: *"P324 (the jar) ends 77 of 179 inscriptions in reading order; P385 begins 29."* Also check every deck slide that quotes this.
2. 🔴 **`MILESTONE1_SUMMARY.md` §10 and §8 say the frequent-sign experiment is "blocked on data".** It is no longer blocked (Part 0 #2). Either run it (Phase E, step E3) or change the text to "unblocked in M4; result in M4 report". Leaving "blocked" while M4 contains the crosswalk tells the examiner you didn't connect your own work.
3. 🟠 **`src/m1_vision/README.md` is stale.** Rewrite it from the actual pipeline: `read_strip.py` and `read_seal.py`, YOLO plus CNN recognizer, template matcher as fallback, denoiser as a documented failed experiment, results table.
4. 🟠 **No mirror handling.** Add `--mirror` to `read_seal.py` and `read_strip.py`; it flips the image horizontally before detection. Document that M1 assumes impression-orientation input, and record the 89-vs-94 mirror test as evidence that the strips already satisfy it.
5. 🟠 **Non-seal images in `data/raw/real_seals/`.** Move these three into `data/raw/real_seals/_excluded/` and say so in `attribution.csv`:
   - `Constitution_Page1_Rammanohar.jpg`
   - `Brahmi_and_Indus_seal_proposed_connection.jpg`
   - `Indus_Script_Falsifiability_Sheet_v3_(Michael_L._Grasa,_2025).png`
6. 🟠 **M1 writes only the top-1 sign per box** (`strip_eval.csv`). The engine and the M3 re-ranking experiment need the **top-5 with scores**. Add a JSON dump (Phase E).
7. 🟠 **No cross-validation in M1.** YOLO and the CNN use a single train/val split. Add 3-fold cross-validation for the CNN on synthetic crops, or state plainly that the real-world evaluation is the 70-sample hand review and explain why that matters more than synthetic cross-validation. The proposal explicitly asks for "cross-validation accuracies".
8. 🟡 **The 70-sample reviews have about ±12 pp margins.** Every M1 recognition number must carry that caveat.
9. 🟡 **`requirements.txt` lacks `ultralytics`**, which M1 needs. Add it (Phase A).

### 2.2 Milestone 2 (Markov), not present, so here is the checklist it must pass

Copy the teammate's work into `src/m2_markov/`, then check every item below. Each "no" is a fix.

1. **Corpus.** Which corpus was used?
   - If it is **CISI only** (179 Mohenjo-daro inscriptions, Parpola numbering), it is too small and in the wrong numbering for integration.
   - 🔴 It must be **re-run on the same corpus as M3**: the M3 CSV, Wells numbering, deduplicated on signs (Phase B). CISI then becomes a **replication check**: run the same statistics on CISI after mapping it to Wells, and report whether the conclusions agree.
2. **Order.** 🔴 Does it know that stored order is reversed reading order? Look for any line that takes `seq[0]` as the "beginning". Every result table must say "reading-initial" or "reading-final", never just "first" or "last".
3. **Transition matrix.** Is it conditional, with rows summing to 1? Does it include START and END states? Are damaged signs (`000`) and `/` alternatives handled the same way as M3 (`000` breaks the chain)?
4. **Witten–Bell.** Is it implemented correctly? Test it on a 3-sign toy example by hand: P(b|a) = (C(a,b) + T(a)·P_uni(b)) / (N(a) + T(a)). Rows with no observations should fall back to the unigram.
   - 🟠 There must be **one** Witten–Bell implementation in the project. Right now M3 has its own. After integration, M3's baseline and M4's controls must import M2's.
5. **Directional asymmetry.** Is it *quantified*, not just shown in a table? Minimum requirements:
   - KL divergence between the reading-initial and reading-final sign distributions
   - conditional entropy left-to-right vs. right-to-left
   - a permutation test that shuffles which end is which and gives a p-value
6. **Collocations.** PMI (or another index) with a **minimum count threshold**, because PMI explodes on rare pairs. Report the threshold.
7. **Deduplication policy.** Are duplicate inscriptions counted once or every time? Either is defensible. 🔴 **Not stating which is a flaw.** Recommended: counts on unique inscriptions as the main result, with all-rows counts as a sensitivity check.
8. **Outputs saved to disk.** At minimum `reports/m2/transition_matrix.csv`, `positional_stats.csv`, `collocations.csv` and `direction_tests.json`, so M4 and the engine can read them without re-running anything.

### 2.3 Milestone 3 (masked encoder)

**Verified on your machine:** test results reproduce exactly (32.1% top-1, perplexity 30.0, ECE 0.022).

**Problems:**

1. 🔴 **Deduplication leak.** `load_unique_sequences()` keys on the full token tuple including edge markers. So `+740-220-590+` and `[740-220-590+` both survive and can land in different splits. 37/373 test inscriptions are affected. The effect is small (59/1,310 predictions; bigram 30.2% → 30.4% without them), but it is a textbook leak. **Fix: deduplicate on the sign sequence only**, keeping the most frequent edge variant.
2. 🔴 **Fake hyper-parameter search.** `run_all.py` defines `grid` and then never uses it. The `search` list is typed-in numbers. Write the actual loop (one seed per configuration, validation NLL), save its output, and choose `best_cfg` from it in code.
3. 🔴 **`run_all.py` destroys the learning curves if re-run.** When checkpoints exist it loads them, keeps `hists = []`, and redraws `figures.png` with an empty left panel. Fix: save each seed's history to `artifacts/history_seed{s}.json` when training, and load it when resuming.
4. 🔴 **Learning curves only for seed 2.** Fixed automatically by retraining after fixes 1–3.
5. 🟠 **"Transformer beats bigram" has no significance test.** The gap is 32.1% vs 30.2% on 1,310 predictions. Add a paired test (McNemar on per-prediction correctness, or a paired bootstrap of the difference) and **5-fold cross-validation** of both models. This also produces the "cross-validation accuracies" the proposal requires.
6. 🟠 **Entropy is never stated, though the milestone is called "Sequential Entropy".** Report cross-entropy in bits per sign: unigram ≈ 6.8, bigram ≈ 5.1, ensemble ≈ 4.9 (= log₂ perplexity). These must be recomputed after the retrain.
7. 🟠 **README wording.**
   - "first/last position" must become "reading-final/reading-initial".
   - "a crosswalk is needed" must become "crosswalk built in `src/common/signs.py`".
   - "re-run in ~20 minutes" must be true after fix 2.
8. 🟠 **`predict.py` accepts IDs in any numbering and any order without complaint.** Add a `numbering=` argument ("wells" default, "im77", "parpola") that converts via the crosswalk, and an `order=` argument ("stored" default, "reading") that reverses when needed.
9. 🟠 **Duplicate data file.** `src/m3_masking/data_inscriptions.csv` duplicates `data/raw/m3_csv/inscriptions.csv`. Keep one. Point `run_all.py` at `data/raw/m3_csv/inscriptions.csv`.
10. 🟡 The validation set is used for early stopping **and** temperature fitting. This is acceptable, but say so.
11. 🟡 The `dir.` column has messy values (`'R/L '`, `'R/l'`). Normalise them in the shared loader. M3 doesn't use the column, but M2 might.

### 2.4 Milestone 4 so far (my own work, held to the same standard)

1. 🔴 Everything in `reports/m4/` is built on M3's **current** split. After the M3 retrain (Phase D), **rerun all M4 scripts**. Do not mix old and new numbers anywhere.
2. 🟠 `perplexity_bench.py` and `controls.py` use M3's copy of the Witten–Bell bigram. Switch them to M2's once M2 is in (Part 2.2, item 4).
3. 🟠 The Naive Bayes result "real vs Markov-generated = 0.50 AUC" is **correct and expected**, not a bug. Naive Bayes sees only sign and pair counts, and a bigram generator reproduces exactly those. The report must explain this, otherwise it looks like a failure.
4. 🟠 Add **confidence intervals**: bootstrap 95% CIs on the median perplexities and on the AUCs. Without them, "real < Markov" is an unqualified claim.
5. 🟡 The synthetic-corpus audit compares numbering-free statistics, because its `T###` IDs match no available scheme. Caveat: if their scheme writes numerals as repeated strokes, part of the 25.5% repeat rate could be encoding. The length gap (7.53 vs ~4), the 170-sign vocabulary, and the per-sign over-use measured in **their own** numbering (T283 at 6.8% vs 1.45%) do not depend on this.
6. 🔴 E5 (reading test), E6 (error propagation), E7 (guardrail on M1 output) and E8 (generator) are not done. See Part 5.

### 2.5 Project-level

1. 🔴 **Not a git repository.** You have no history and no way to prove what changed when. Run `git init` today (Phase A). The `.gitignore` already excludes data and weights.
2. 🔴 **The root `README.md` is from September.** It references `src/m2_markov/`, which doesn't exist, doesn't mention the M3 CSV or the synthetic corpus, and still says the corpus is CISI.
3. 🔴 **No single command reproduces the project.** Add `run_all.sh` (Phase I).
4. 🟠 **Clutter to remove before submission** (Part 7).
5. 🟠 **`requirements.txt` covers M1 only.** Add `ultralytics`, and note `torch>=2.0` (CPU is fine for M2–M4).

---

## Part 3 — The integration design (decide this once, then never again)

Integration fails when each milestone makes its own silent assumptions. Make these decisions once, put them in code, and have every milestone import that code.

**Decision 1 — Canonical sign space = Wells numbering** (the M3 CSV's own numbers).

Why: M2, M3 and M4 all run on this corpus. Only M1 is different, and M1 converts at its boundary.

**Decision 2 — Canonical internal order = stored order** (left-to-right as transcribed = reverse of reading order).

Why: M3's trained models expect it. Convert to reading order **only for display and for direction claims**.

**Decision 3 — Canonical corpus = the M3 CSV, deduplicated on sign sequence.**

- CISI is used for two things only: (a) building and validating the crosswalk, and (b) a replication check of M2.
- Damaged signs `000` are kept as gaps and are never prediction targets. `/` alternatives take the first readable reading. This is M3's existing convention, and M2 must match it.

**Decision 4 — One split for everything.** M3's train/validation/test split (after the dedup fix) is the only split.

- M2 statistics used *as models* (controls, baselines, the engine) are fit on **train** only.
- M2's *descriptive* statistics (the report tables) may use the whole corpus. Say which is which.

**Data flow after integration:**

```
seal / strip image
  └─ M1  read_strip / read_seal  ──► IM77 IDs, reading order, top-5 + scores per sign
        └─ common.signs.im77_to_wells   (crosswalk; ambiguous → most frequent Wells sign, flagged)
        └─ common.order.reading_to_stored   (reverse)
              └─ M2  positional n-gram report: P(start), P(end), P(next|prev) per slot,
              │      collocations present, directional fit
              └─ M3  per-sign surprise; re-rank M1's top-5 with context; fill gaps
              └─ M4  guardrail: pseudo-perplexity percentile vs real & control distributions,
                     Naive Bayes real-vs-fake probability, verdict + reasons
  ► engine report (HTML + PNG panels)
```

---

## Part 4 — Step-by-step instructions

Run everything from the project root (`cd ~/indus-epigraphy`) inside your virtual environment. Each phase depends on the ones before it unless stated otherwise.

### Phase A — Housekeeping (½ day, do first)

**A1. Start version control:**

```bash
cd ~/indus-epigraphy
git init
git add .gitignore README.md requirements.txt src reports/*.md notebooks
git commit -m "Snapshot before integration"
```

**A2. Fix `requirements.txt`.** Add these lines:

```
ultralytics>=8.0
scikit-learn>=1.3
```

**A3. Create the new folders:**

```bash
mkdir -p src/common src/m2_markov src/engine reports/m2 reports/m3 reports/m4 data/raw/real_seals/_excluded
```

**A4. Move the 3 non-seal images into `data/raw/real_seals/_excluded/`** and add a column `excluded_reason` to `attribution.csv`.

**A5. Remove duplicates and clutter.** Moving them to `data/processed/_stale/` is fine; see Part 7 for the full list.

**Done when:** `git log` shows one commit, `requirements.txt` has `ultralytics`, and the three non-seal images are gone from the main folder.

### Phase B — Shared foundations (1 day; everything else depends on this)

**B1. `src/common/signs.py`, the crosswalk.**

- Read every `data/raw/cisi/features/P*.json` file.
- Build a table with one row per (Parpola, Wells, IM77) combination and columns `parpola, wells, im77, description`. Strip the `P`, `W` and `M` prefixes and zero-pad Wells to 3 digits (`W740` becomes `"740"`, `M342` becomes `342`).
- Save it to `data/processed/crosswalk.csv`.
- Provide these functions:
  - `im77_to_wells(id) -> list[str]`
  - `wells_to_im77(id) -> list[int]`
  - `parpola_to_wells(id) -> list[str]`
  - `im77_to_wells_one(id, counts)`, which returns the most frequent candidate in the M3 corpus plus an `ambiguous` flag
- Write a `validate()` function that aligns the CISI corpus against the M3 CSV via the CSV's `cisi` column and prints the slot agreement. **It must reproduce about 91.8% agreement on 785 slots over 148 inscriptions.** If it doesn't, the crosswalk is wrong, so stop.
- Coverage facts to quote in the report (I measured these):
  - 334 of 417 IM77 signs have a Wells mapping, 227 of them one-to-one.
  - 94.2% of M3-corpus sign occurrences map to some IM77 sign; 78.0% map to exactly one.
- 🟡 Disclose that the mapping is many-to-many for about 22% of occurrences. Example: Wells `002` corresponds to IM77 99 *and* 100.

**B2. `src/common/order.py`, one place for direction.**

```python
STORED = "stored"     # left-to-right as transcribed (CISI, M3 CSV) = reverse reading order
READING = "reading"   # right-to-left reading order (M1 output)
def to_reading(seq): return list(reversed(seq))
def to_stored(seq):  return list(reversed(seq))
```

Every function that takes a sequence must document which order it expects. Add the evidence to the module docstring: "jar sign 740 is stored first in 71% of R/L texts and last in 3%".

**B3. `src/common/corpus.py`, the single corpus loader.**

- Move M3's `parse_text` and a **fixed** `load_unique_sequences` here. Deduplicate on the sign sequence only, record the count of duplicates, and keep the most frequent edge variant.
- Normalise the `dir.` values: strip whitespace and uppercase.
- M2, M3 and M4 must all import from here. **Delete the duplicate loaders afterwards.**

**Done when:** `python -m src.common.signs` prints about 91.8% agreement, and `python -c "from src.common.corpus import load; print(len(load()))"` prints the number of unique inscriptions.

### Phase C — Bring Milestone 2 in (1–2 days, depends on B)

**C1.** Copy the teammate's M2 into `src/m2_markov/`.

**C2.** Run the checklist in Part 2.2 item by item. Write the answers into `src/m2_markov/README.md`.

**C3.** Refactor M2 to use `src/common/corpus.py` (Wells, stored order) and `src/common/order.py`. All positional outputs must be labelled reading-initial or reading-final.

**C4.** Make M2 save these files to `reports/m2/`:

- `transition_matrix.csv` (Witten–Bell smoothed, fit on train; plus an all-data version for description)
- `positional_stats.csv`: per sign, count, P(reading-initial), P(reading-final), P(medial)
- `collocations.csv`: PMI with the minimum count stated in the header
- `direction_tests.json`: KL(initial‖final), conditional entropy in both directions, permutation-test p-value
- `cisi_replication.json`: the same tests on CISI mapped to Wells

**C5.** Make M2's Witten–Bell the **only** one. Change `src/m3_masking/baselines.py` to import it, and confirm the M3 bigram numbers are unchanged before you change anything else.

**C6. Direction claim wording** (Part 6). The statistics prove direction is **fixed and asymmetric**. That it is **right-to-left** comes from the physical evidence encoded by epigraphers: the `dir.` column (4,263 R/L vs 215 L/R) and the transcription convention. Say both.

**Done when:** `reports/m2/` contains all five files, every positional table says "reading-initial/final", and M3's bigram score is identical whether it uses its own implementation or M2's.

### Phase D — Fix and retrain Milestone 3 (1 day, depends on B; can run alongside C)

**D1.** Switch M3 to `src/common/corpus.py`. This applies the dedup fix.

**D2.** Replace the typed-in `search` with a real loop over `grid`. Save `artifacts/search.json`.

**D3.** Save the history per seed (`artifacts/history_seed{s}.json`) and load it on resume. The figure must show all 3 seeds.

**D4.** Move the old `artifacts/` to `data/processed/_stale/m3_artifacts_v1/` (keep it for the record) and run `python src/m3_masking/run_all.py`.

**D5.** Add `src/m3_masking/cv.py`:

- 5-fold cross-validation over unique inscriptions.
- In each fold, train 1 Transformer seed and the bigram, and record leave-one-out top-1/top-5/perplexity.
- Report mean ± sd, and a **paired test** (McNemar or paired bootstrap) of Transformer vs. bigram top-1.
- Save to `reports/m3/cv.json`.

**D6.** Report bits per sign (log₂ perplexity) for unigram, bigram and ensemble.

**D7.** Add `numbering=` and `order=` arguments to `predict.py` (Part 2.3, item 8). Then test with `sc.complete(["342","?","..."], numbering="im77", order="reading")` and confirm the jar is converted to `740` and the sequence is reversed.

**D8.** Update `src/m3_masking/README.md`: new numbers, reading-order wording, crosswalk, honest cross-validation and significance result.

**Done when:**

- `figures.png` shows 3 learning curves.
- `search.json` exists and `best_cfg` comes from it.
- `cv.json` exists.
- Test top-1 is reported with its cross-validation mean ± sd and p-value.
- Old and new numbers are never mixed.

### Phase E — Fix Milestone 1's boundary and run the unblocked experiments (1–2 days, depends on B; parallel with C and D)

**E1.** Add `--dump-topk` to `read_strip.py` and `read_seal.py`. It writes `reports/m1/strip_topk.json`: for each image, an ordered list (reading order) of `{box, det_conf, top5:[[im77_id, score], ...]}`.

**E2.** Add `--mirror` (horizontal flip before detection) and document the impression-orientation assumption plus the 262-crop mirror test.

**E3. The frequent-sign restriction** (previously "blocked").

- Map the M3 corpus frequencies to IM77 through the crosswalk.
- Restrict recognition candidates to signs attested in the corpus. Under one-to-one mapping, the top 100 IM77 signs cover 94.1% of occurrences and the top 150 cover 98.0%.
- Re-run recognition on the **same** 70 reviewed detections, so it is a fair comparison, and re-judge them.
- Report top-1/top-5 against the 17%/41% baseline.
- Whatever the result, this closes the "blocked" item honestly.

**E4. Context re-ranking (M3 helps M1).**

- For each detection, combine the visual score with M3's context probability: score = log P_vis + λ · log P_M3(sign | the other predicted signs), with signs converted to Wells and stored order.
- Choose λ on half of the hand-verified detections and test on the other half.
- Only detections whose true sign is known count. That is about 46 across the two reviews, so **report it as a pilot with its n.** Do not dress it up.

**E5.** Update `MILESTONE1_SUMMARY.md` §8, §10 and §11 (direction fix, "blocked" removed, results of E3/E4) and rewrite `src/m1_vision/README.md`.

**Done when:** `strip_topk.json` exists, E3 and E4 have numbers with n stated, and the M1 summary no longer says "blocked" or "opens".

### Phase F — Milestone 4 experiments (3–4 days, depends on C and D)

These experiments are the actual content of M4. Each one names its script, inputs, outputs and what to conclude.

**F1 — Perplexity vs. noise (E1). Exists; rerun.**

- Script: `src/m4_guardrails/perplexity_bench.py`.
- Change it to use M2's bigram, and add bootstrap 95% CIs for each set's median.
- Inputs: M3 artifacts (new), `controls_test.json` (regenerate with `controls.py` first).
- Output: `perplexity_vs_noise.png`, `perplexity_summary.json`.
- Conclusion to test: real < Markov < shuffled < frequency-random < uniform for the Transformer. The bigram rates Markov fakes as more natural than real text, which shows a pair-only model can be fooled.

**F2 — Confidence vs. structural controls (E2). Exists; rerun.**

- Already inside `perplexity_bench.py`.
- Conclusion to test: on noise, M3 still says "High" about 10% of the time and is almost never right. Its calibration holds only on in-distribution input.

**F3 — Naive Bayes guardrail (E3). Exists; rerun, then extend.**

- Add CIs and add **two new fake types**: M3-generated sequences from F8 and M1-output sequences from F7.
- The explanation of the 0.50 Markov AUC goes in the report (Part 2.4, item 3).

**F4 — Synthetic corpus audit (E4). Done.**

- Script: `synthetic_check.py`. Rerun after the M3 dedup change, because the real-corpus statistics shift slightly.

**F5 — The proposed Sanskrit reading (E5). Not started; the hardest one.**

- Find where the CSV came from and the decipherment's published sign-to-sound table. Cite both.
- Get a Sanskrit word list, for example Monier-Williams headwords, as a text file.
- Write `src/m4_guardrails/reading_test.py`:
  - Apply the published sign values to (a) real inscriptions and (b) each control set from `controls.py`.
  - Segment into words with the same rules the decipherer allows. Write those rules down: this is where "arbitrary" lives.
  - Score: the share of output covered by dictionary words.
  - Permutation test: 1,000 shuffles per inscription; the p-value is the share of shuffles that score at least as well as the real order.
- The conclusion is whatever the data says. If shuffled inscriptions produce "valid Sanskrit" about as often as real ones, the method cannot distinguish text from noise. Word this as a statement about **this method**, never about all possible decipherments.
- **Fallback if the table cannot be found by the end of day 2:** test internal consistency only. Use multinomial Naive Bayes to predict each sign's presence from the Sanskrit reading's character n-grams, and compare against a control where readings are shuffled across inscriptions. This shows *whether* the readings are systematic, not whether they are *true*. Say so explicitly. Note that the 1,763 readings are consistent (only 4/1,749 texts have conflicting readings), so consistency alone proves nothing.

**F6 — Error propagation (E6). Not started; required by the proposal.**

- Script: `src/m4_guardrails/error_propagation.py`.
- Take the clean test inscriptions (Wells, stored order). Corrupt them with an **M1-shaped error model** at a strength *s* from 0 to 1, where *s* = 1 equals M1's measured rates:
  - substitution 83% (top-1 wrong)
  - deletion about 12.5% (4.9 detected vs. 5.6 true signs)
  - insertion about 14% (86% precision)
- Draw substitutes from **M1's actual output distribution** (the IDs M1 tends to predict, from `strip_eval.csv`, mapped to Wells). This reproduces M1's collapse bias instead of assuming uniform errors.
- At each *s*, measure:
  - M2's direction statistic and whether its p-value stays below 0.05
  - KL(clean matrix ‖ corrupted matrix)
  - M3 top-1 on the uncorrupted positions
  - the share of inscriptions the M4 guardrail flags as non-real
- Output: one figure with four panels against *s*, plus `error_propagation.json`.
- Key number for the report: **the error level at which the direction finding disappears**, and where M1's real error level sits relative to it.

**F7 — The guardrail applied to M1's real output (E7). New; this is the integration showcase.**

- Run the engine (Phase G) over all 77 strips.
- For each strip, compute the M4 verdict on M1's predicted sequence: pseudo-perplexity percentile against the real test distribution, and Naive Bayes P(real).
- Expected result, to be confirmed: most M1 outputs look like noise to M3. **That is the guardrail correctly detecting digitization errors.** It connects M1's failure, M3's model and M4's guardrail in one result.
- Also compare against strips where E4's re-ranking changed the reading: does re-ranking make outputs look more real? Report n.

**F8 — The generator (E8). Your professor's request.**

- Script: `src/m4_guardrails/generate.py` with two generators:
  - **Markov**: sample from M2's train-fit matrix from START until END.
  - **M3 Gibbs**: start from a Markov sample, then repeatedly mask one position and resample it from M3, for 20 sweeps.
- Generate 2,000 sequences of each.
- Evaluate them with F1 (perplexity) and F3 (Naive Bayes), and compare with real text and with the hellosindh synthetic set (numbering-free statistics, as in F4).
- The honest framing: "we can generate sequences that match the corpus's statistics to a measured degree. That says nothing about meaning." This turns the professor's request into an M4 result instead of a separate toy.

**Done when:** `reports/m4/` has one JSON and one figure per experiment F1–F8, and every number has n and a confidence interval or p-value.

### Phase G — The unified "Epigraphic Parsing Engine" (1–2 days, depends on B–F)

The proposal's wording: *"A unified Python script processing an input seal image all the way through visualization panels to output positional n-gram probability reports."*

**G1.** Create `src/engine/parse.py`:

```
python src/engine/parse.py data/raw/isd/1/1text.jpg --out reports/engine/1text
python src/engine/parse.py --all-strips --out reports/engine/      # batch mode for F7
```

**G2. Stages,** each as a function, each saving its intermediate output:

1. M1: detect, recognise top-5, order → `m1.json`
2. Convert: IM77 → Wells (with ambiguity flags) and reading → stored order → `converted.json`
3. M2: for each slot, P(start) if first in reading order, P(end) if last, P(next|prev) for each adjacent pair, collocations present, and the sequence's total log-probability → `m2_report.csv` (**this is the "positional n-gram probability report"**)
4. M3: surprise per sign, context re-ranked top-5 (E4), suggested fills for low-confidence signs → `m3.json`
5. M4: pseudo-perplexity and its percentile against real/control distributions, Naive Bayes P(real), verdict (`plausible` / `suspicious` / `noise-like`) with reasons → `m4.json`

**G3. Visualization panels** in one PNG plus an `index.html`:

- (a) the image with numbered boxes
- (b) each crop beside its top-3 IM77 templates
- (c) a heat-row of M2 transition probabilities along the sequence
- (d) a bar chart of M3 per-sign surprise
- (e) where this inscription's perplexity falls on the F1 distributions

**G4.** The engine must **refuse loudly** instead of guessing:

- if a sign has no crosswalk entry, show `UNMAPPED` and exclude it from M2/M3
- if fewer than 2 signs survive, skip the M4 verdict with a reason

**Done when:** one command on one image produces the 5 JSON/CSV files, the PNG and the HTML, and batch mode over 77 strips feeds F7.

### Phase H — Notebooks (1 day, can start after D)

The proposal requires "documented training notebooks showcasing cross-validation accuracies, learning curves, and token prediction loss charts".

**Notebooks load saved results. They do not retrain.** Each starts with the exact command that regenerates its inputs.

| Notebook | Contents |
|---|---|
| `notebooks/01_m1_vision.ipynb` | dataset audit (one-shot discovery), synthetic examples, YOLO training curves (`runs/sign_detector/results.png`), CNN recognizer curves, the 4-attempt ablation, the 70-sample reviews with margins, E3/E4 results |
| `notebooks/02_m2_markov.ipynb` | transition matrix heatmap (top 30 signs), positional tables, direction tests with p-values, collocations, CISI replication |
| `notebooks/03_m3_masking.ipynb` | **learning curves (3 seeds) = token prediction loss**, **5-fold cross-validation table**, significance test, calibration diagram, breakdowns |
| `notebooks/04_m4_guardrails.ipynb` | F1–F8 figures and tables |

**Done when:** "Restart & Run All" works on every notebook without errors and without training.

### Phase I — Reproducibility and the report (2 days)

**I1. `run_all.sh`** at the root, running every step in order with comments. Mark the slow ones (YOLO training, M3 training) as optional with flags.

**I2. Rewrite the root `README.md`:**

- the four-milestone pipeline with **real folder names**
- the three numbering systems and the crosswalk
- the order convention
- a data-sources table (including the M3 CSV's provenance and the synthetic corpus)
- how to run the engine

**I3. `DATA_SOURCES.md`.** Every dataset, its URL, its license, and what it's used for, including the two proposal resources you didn't use and why.

**I4. The M4 research report** (`reports/M4_REPORT.md`, then exported to PDF). Suggested structure:

1. Question and scope (testing claims, not deciphering)
2. Data and the three numbering systems; the crosswalk and its 91.8% validation
3. The structural controls
4. Perplexity vs. noise (F1) and confidence vs. controls (F2)
5. Naive Bayes guardrail (F3)
6. The synthetic corpus audit (F4)
7. The proposed reading test (F5)
8. Error propagation from M1 (F6) and the guardrail on M1's real output (F7)
9. Generation (F8)
10. What the whole project shows: high confidence on benchmarks ≠ understanding (M1's 87.9% vs 17%, M3's "High" ratings on noise, the synthetic corpus's 85–98% self-scores)
11. Limitations

**I5. Final deck.** Update the Milestone 1 slides (direction fix, "blocked" removed), then add M2–M4 slides.

---

## Part 5 — Order of work and division of labour

| Day | Person A | Person B | Person C |
|---|---|---|---|
| 1 | Phase A, then B1 crosswalk | Get M2 into the folder; C2 checklist | **Source the F5 materials** (CSV provenance, sign-value table, word list) |
| 2 | B2–B3 | C3–C5 | F5 sourcing / fallback decision |
| 3 | D1–D4 (M3 retrain) | C4 outputs, C6 wording | E1–E2 (M1 top-5 dump, mirror flag) |
| 4 | D5–D8 (cross-validation, README) | F6 error propagation | E3–E4 (unblocked M1 experiments) |
| 5 | F1–F3 rerun with CIs | F8 generator | F5 reading test |
| 6 | Phase G engine | F4 rerun, F7 via engine | Notebooks 01–02 |
| 7 | Notebooks 03–04 | Report §1–6 | Report §7–11 |
| 8 | `run_all.sh`, README, DATA_SOURCES | Deck | Full read-through against Part 1's table |

The critical path runs B → D → F → G → report. Anything that slips there delays everything after it.

---

## Part 6 — What you may and may not claim

Read this before writing any slide.

| ✅ Say | ❌ Never say |
|---|---|
| "Sign order carries information: shuffling the same signs raises perplexity about 8×." | "The script encodes a language." (not shown) |
| "The positional asymmetry shows a fixed direction; that the direction is right-to-left follows from the epigraphers' physical evidence (`dir.` column) and the transcription convention." | "Our statistics prove right-to-left." |
| "This specific proposed reading does / does not beat structural controls." | "We disproved all decipherments." |
| "M1 recognises 17% top-1 / 41% top-5 (n = 70 per review, ±12 pp)." | "M1 reads seals." |
| "M3's confidence is calibrated on real inscriptions (ECE 0.02) but not on out-of-distribution input." | "M3 is 76% accurate." (only in the High band) |
| "The synthetic corpus differs from real text on length, vocabulary and repetition." | "The synthetic corpus is fake / worthless." |
| Sign IDs as opaque labels (`W740`, `IM77-342`) | "the jar sign means …" |

Always state the numbering when quoting a sign: `W740` (= IM77 342 = P324), never a bare `740`.

---

## Part 7 — What to remove or move before submission

Move these to `data/processed/_stale/` or delete them; none of them should be in the final project:

- `data/raw/Milestone_3_AI.zip` (already extracted)
- `src/m3_masking/data_inscriptions.csv` (duplicate of `data/raw/m3_csv/inscriptions.csv`)
- `reports/m1_requested_code.zip` and `reports/deck_figures.zip` (regenerable)
- `Claude outputs/` (move the `.pptx` into `reports/presentations/`; delete the duplicate README)
- all `__pycache__/` and `.DS_Store` files
- `data/processed/_stale/` (keep locally, exclude from submission)
- `yolov8n.pt` in the root: these are downloaded base weights, not yours. Leave it out of the submission and note it in the README; ultralytics re-downloads it automatically.

**Keep, but label clearly as documented baselines or failed experiments:**

- `detect_signs.py`, `crop_and_match.py` (classical CV baselines)
- `denoiser.py`, `train_denoiser.py`, `synth_pairs.py` (the collapsed converter)
- `batch_eval.py`, `make_review.py` (the museum-photo phase)

Negative results are part of your story. Unlabelled dead code is a flaw.

---

## Part 8 — Final checklist (tick all before submitting)

- [ ] `src/m2_markov/` exists and passes every item in Part 2.2
- [ ] `src/common/` crosswalk validates at about 91.8%; every milestone imports the shared corpus, order and sign modules
- [ ] No document says "first/last" without "reading-" or "stored-"; M1 §11 is fixed
- [ ] M3 retrained with the dedup fix, real search, 3 learning curves, 5-fold cross-validation and a significance test
- [ ] All M4 numbers regenerated after the M3 retrain; none mixed
- [ ] F1–F8 each have a script, a JSON, a figure, an n, and a CI or p-value
- [ ] The engine runs on one image and in batch; it refuses to guess on unmapped signs
- [ ] Four notebooks pass "Restart & Run All" without training
- [ ] `run_all.sh`, root README, `DATA_SOURCES.md` written; the M3 CSV's source cited
- [ ] M1 README rewritten; "blocked" removed; mirror assumption documented; non-seal images excluded
- [ ] Clutter removed (Part 7); `git log` shows the history
- [ ] Every claim in the report and the deck passes Part 6
