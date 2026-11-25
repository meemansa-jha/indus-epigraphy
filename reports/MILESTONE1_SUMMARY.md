# Milestone 1 — Automated Visual Grapheme Extraction
### Indus Epigraphy Project · summary of all work, results and findings

> **For the deck:** the story is *not* "we built a sign reader and it works." It is
> **"we built it, measured it honestly, and discovered that synthetic training data
> produces near-perfect benchmark scores that do not survive contact with real
> artifacts."** That finding is reproducible, measured three independent times, and
> is the strongest result we have. It also sets up Milestone 4 directly.

---

## 1. What Milestone 1 asked for

From the project proposal:

- **Objective:** parse raw artifact images to extract, crop and categorise individual script symbols.
- **Technical task:** a deep-learning vision backbone (MobileNet / CNN in PyTorch) detecting bounding boxes around individual signs and mapping them to standardized numerical IDs from the concordance.
- **Deliverable:** *"An image pipeline script that inputs a photo of a seal inscription and outputs an ordered list of standardized sign IDs."*

**Background constraints:** the Indus script has **417 distinct signs** (Mahadevan's 1977 "IM77" standard). Inscriptions average about **5 signs** and read **right-to-left**. There is no bilingual text, so no reading can be verified. Throughout, a sign is treated as an **opaque integer label** — we never claim what any sign *means*.

---

## 2. The discovery that shaped the whole approach

The dataset the proposal pointed to (Kaggle `storesource/im-417-150`) is **not a training set — it is a reference chart**:

- **417 PNG images, exactly one per sign**, named `1.png` … `417.png`
- Each **150 × 150 px, grayscale** (white ink on black tile; we invert on load)
- A companion CSV that is just the same images flattened to pixels — no extra information

So the name decodes as **417 signs at 150×150 resolution**, one image each.

**Why this matters:** you cannot train a normal classifier with one example per class — there is no second image to validate against and no variation to learn from. This is a **one-shot recognition problem**. Every later decision follows from this. Had we skipped the audit, we'd have wasted days.

A second consequence: these are **clean line-drawings**, while real signs are shallow grooves carved in stone, lit from an angle and worn by 4,000 years. That **domain gap** became the central technical problem.

---

## 3. Architecture

```
seal / inscription image
   → isolate the inscription region
   → YOLO detector  : WHERE is each sign?     (single class: "sign")
   → CNN recognizer : WHICH of the 417 is it?
   → order boxes right-to-left = reading order
```

Two deliberate design decisions:

1. **Detection is single-class.** "A sign-shaped mark in a row" transfers from synthetic to real far better than 417 individual glyph appearances.
2. **Detection and recognition are decoupled**, so each can be improved independently. This proved essential — they failed for completely different reasons, and fixing one never fixed the other.

**The core technique, used twice: *synthesize the domain*.** With one image per sign and no annotated real seals, there was no training data — so we generated it, rendering the 417 real templates under thousands of simulated conditions (carved relief, random lighting, wear, perspective, blur, noise). Because *we* place each glyph, every bounding box and identity is known exactly: perfect labels, unlimited quantity.

---

## 4. Datasets used

| Dataset | What it is | Role |
|---|---|---|
| **Kaggle IM-417-150** | 417 sign templates, 150×150 | The sign inventory / reference |
| **Indus-Seal-Dataset** (GitHub, akksshhaay) | 137 processed seals; **77 isolated `text` crops** | **The real test set** — tight inscription strips |
| **Wikimedia Commons** | 38 museum seal photographs | Early real-world test (largely superseded) |
| **CISI corpus** (GitHub, Parpola-based) | 179 inscriptions as sign sequences | Ground-truth statistics; feeds Milestone 2 |
| **Mahadevan 1977** (Internet Archive) | 3 scanned volumes + OCR | Sign list, frequency facts, inscription plates |
| **Indus-Alphabates-Dataset** | 1,840 clean glyph forms | **Unusable** — no labels (see §8) |

---

## 5. What we built — 17 Python modules in `src/m1_vision/`

| Module | Purpose |
|---|---|
| `audit.py` | Dataset audit; detected the one-shot problem |
| `preprocess.py` | Otsu preprocessing + augmentation helpers |
| `synth.py` | **Synthetic inscription generator** — 4 styles: `clean`, `realistic`, `photo`, `strip` |
| `synth_signs.py` | 41,700 realistic single-sign crops (dissolves the one-shot problem) |
| `synth_pairs.py` | 24,000 paired (carved → clean) images for the converter |
| `prepare_yolo.py` | Train/val split + `data.yaml` |
| `train_yolo.py` | Trains the YOLOv8n sign detector |
| `train_recognizer.py` | Trains the 417-way MobileNetV3 recognizer |
| `train_denoiser.py` | Trains the relief→clean U-Net |
| `template_match.py` | One-shot matcher (no training) |
| `recognizer.py`, `denoiser.py` | Inference wrappers |
| `read_seal.py` | Full pipeline on a museum photo |
| `read_strip.py` | **Main deliverable** — pipeline on inscription strips |
| `batch_eval.py`, `make_review.py` | Evaluation + hand-verification tooling |
| `detect_signs.py`, `crop_and_match.py` | Classical baselines (document failure modes) |
| `fetch_real_seals.py` | Wikimedia downloader with licence recording |

---

## 6. The experimental arc

Each step failed in a specific, diagnosable way that motivated the next.

1. **Classical CV failed — for a structural reason.** Thresholding + projection profiles produced one blob covering 83% of the image plus fragments; given a hand-cropped band it over-segmented 6 signs into 9, then 12. **Indus signs are multi-stroke**, so a projection profile cannot distinguish a gap *inside* a sign from a gap *between* signs. No threshold fixes this — it requires knowing what signs look like. This is why the literature uses trained detectors.

2. **Template matching worked immediately** on clean crops: **72.1% top-1 / 86.9% top-5**, zero training (chance = 0.24%).

3. **First YOLO detector:** mAP50 **0.995** on synthetic; on a real seal, 5 boxes mostly on *parts* of signs. We hypothesised a scale mismatch, measured it (synthetic 0.19 of frame height vs real 0.16 — near-identical) and **discarded the hypothesis**.

4. **Museum photographs, 38 seals:** 76% "coverage" looked fine, but inspection showed a worn seal producing 13 boxes on cracks, and a clear illustration producing 1 box on a bull's ear. Hand-labelling 11 seals gave **93 visible signs vs 54 detections**, with **4 seals returning zero despite clearly visible signs**.

5. **Switching to isolated inscription strips** removed the two worst failure modes (seal isolation, motif clutter). One geometry mismatch had to be fixed: strips are tight crops where signs fill ~70–90% of frame height, while our synthetic data had signs at ~19%. Adding a matching `strip` style and retraining **fixed detection**.

6. **Three attempts to fix recognition, all measured, none successful** — see §7.

---

## 7. RESULTS — the key tables for the deck

### 7a. Every model: near-perfect on synthetic, fails on real

| Model | Synthetic score | Real-world result |
|---|---|---|
| YOLO detector | **mAP50 0.995** | Works — but only after geometry was matched |
| CNN recognizer | **87.9% top-1 / 97.7% top-5** | **12–17% top-1** |
| Relief→clean U-Net | **IoU 0.816** | Collapsed to **17 of 417** signs |

> **This is the headline slide.** Three independent models, three different tasks, three near-perfect synthetic scores — and no synthetic metric predicted the real-world outcome in any case.

### 7b. Detection — solved

| Measure | Museum photos | Inscription strips (final) |
|---|---|---|
| Coverage | 76% (9 seals with nothing) | **99% (76 of 77)** |
| Signs per image | mean 4.5, max 13 (noise) | **mean 4.9, median 5, max 11** |
| Detection precision (hand-verified) | — | **86%** |
| Detector confidence | 0.41 | **0.49** |

Real inscriptions average **5.60 signs** (measured from the CISI corpus), so a detected mean of **4.9** is close to correct. Detector `mAP50-95` also improved 0.825 → **0.875** (final epoch of the saved run, `runs/sign_detector/results.csv`; an earlier draft said 0.855) after the geometry fix, meaning tighter boxes.

### 7c. Recognition — four attempts, no usable accuracy

All hand-verified on random samples of 70 detections (±~13pp margin at 95% confidence):

| Attempt | Detection precision | Top-1 | Top-5 |
|---|---|---|---|
| **Baseline** (CNN recognizer) | 77% | **17%** | **41%** |
| + strip-domain training crops | 86% | 12% | 40% |
| + relief→clean converter | — | — | *collapsed, not judged* |
| + binarization & aspect-preserving features | — | — | *clearly worse* |

**Best result: 17% top-1 / 41% top-5.** Against 417 classes, chance is 0.24% — so the model learned something real (≈70× chance) but nowhere near usable accuracy.

### 7d. The collapse check — a metric that works without ground truth

Because no labels exist for real seals, we added a statistical check: if a few sign IDs dominate predictions across the whole corpus, recognition has collapsed.

| Run | Distinct IDs (of 417) | Top-5 IDs' share | Verdict |
|---|---|---|---|
| Baseline | 194 | 13% | Healthy |
| Relief→clean converter | **17** | **96%** | **COLLAPSED** |

This caught the converter's failure *before* any hand-judging time was spent — worth a slide on its own as good experimental practice.

---

## 8. Honest negative results (include these — they're part of the science)

- **Scale-mismatch hypothesis:** proposed, measured, **disproved** (0.19 vs 0.16).
- **Strip-domain training crops:** a measured **null result** — 41% → 40% top-5 is within noise. This told us the problem is not crop *framing* but the **appearance of carved stone**.
- **Relief→clean converter:** IoU **0.816** on synthetic, total collapse on real — a *second, independent* sim-to-real gap, this time inside the converter.
- **The 1,840-glyph font dataset is unusable:** filenames are font-grid coordinates, not sign IDs, and the accompanying CSVs hold unsupervised contour features. Attempting to recover labels by shape-matching to our templates gave **only 12 of 1,840 glyphs (0.7%) above a 0.80 threshold**.
- **Data contamination:** 3 of the 38 Wikimedia images are **not seal photographs** (a painted Constitution illustration, a modern "falsifiability sheet", a Brahmi comparison chart). Any corpus scraped from a public category needs manual curation.
- **Mahadevan's frequency table could not be reliably extracted** from OCR: 377 of 417 rows passed a checksum, but the table is printed two-up so row→sign-ID alignment is unreliable (our total 6,491 vs published 13,372).

---

## 9. Literature and citable facts

**Dixit, V., Hussain, N., Basak, S., Atturu, D., Mitra, D., & Bhattacharya, U. (2025).** "Deep Learning in Archiving Indus Script and Motif Information." *Journal of Computer Applications in Archaeology*, 8(1), 156–169. https://doi.org/10.5334/jcaa.175

Architecturally the **same approach as ours** (YOLOv3 + MobileNet). Crucially: they hand-annotated **232 training + 13 validation images**, explicitly because annotating hundreds of seals was too labour-intensive, and their classifier covered **40 grapheme classes**.

> **Key context for the deck:** the published state of the art, *with hundreds of hand-annotated real seals*, attacked a **40-class** problem. We attempted **417 classes with zero annotated real images**. Our task was substantially harder than the literature's.

**From Mahadevan (1977), *The Indus Script: Texts, Concordance and Tables*, ASI:**
- **417 signs**; 179 have variants, totalling **641 recorded forms**
- **13,372 total sign occurrences** in the corpus
- *"67 signs account for over 80 per cent of the total occurrences"*

---

## 10. Conclusions

1. **Detection is solved** for isolated inscription strips: 99% coverage, 86% precision, correct sign counts, visibly correct boxes.
2. **Recognition does not reach usable accuracy** on real artifacts: best 17% top-1 / 41% top-5 over 417 classes, after four documented attempts.
3. **The cause is a reproducible sim-to-real gap.** Three independent models scored near-perfectly on synthetic data and failed on real artifacts. **Synthetic validation scores had no predictive value for real-world performance.**
4. **Remaining principled option (future work):** restrict recognition to the ~67 frequent signs, as both Mahadevan and Dixit et al. support. This was blocked on data (frequencies were in *Parpola* numbering, templates in *Mahadevan* numbering). **Update:** a crosswalk now exists (`src/common/signs.py`, built from the CISI `features/` files, validated at 91.8% agreement), so the experiment is unblocked; it has not been run yet.

### Why this matters for Milestone 4
Milestone 4 asks us to demonstrate the limits of AI claims about this script. We did not have to argue that abstractly — **we produced it experimentally, three times, with our own system.** A model reporting 99.5% accuracy that cannot read a real seal is the cleanest possible illustration of why confident benchmark numbers must not be mistaken for understanding.

---

## 11. Note for Milestone 2

**Milestone 2 does not depend on Milestone 1's output.** The sign sequences already exist, transcribed by epigraphers, in the CISI corpus (already downloaded and parsed). M1 demonstrates that image→sequence *can* be automated; M2–M4 run on the authoritative transcriptions.

Already measured from the corpus (179 inscriptions, 1,003 sign occurrences, 182 distinct signs):
- **Sequence length: mean 5.60, median 5, range 1–13** — independently reproduces the famous "~5 signs" fact
- **Top 67 signs cover 83.4%** of occurrences — independently replicating Mahadevan's published "over 80 per cent"
- **Strong positional asymmetry:** in **reading order** (right-to-left), sign `P324` **ends** 77 of 179 inscriptions and `P385` **begins** 29. *(Corrected: an earlier version said "opens/closes" because it counted on the corpus's stored left-to-right order, which is the reverse of reading order. Milestone 2 confirms the corrected reading.)* This is the statistical fingerprint of a **fixed reading direction** — the evidence Milestone 2 asks for.

M1's measured error rate can also be injected into the clean sequences to satisfy the proposal's "how do digitization errors propagate downstream" requirement.

---

## 12. Figures for the deck

All images are in **`reports/deck_figures/`**, numbered in suggested presentation
order. A standalone copy of this guide is in `deck_figures/README.md`.

### Essential — these five carry the story

| File | Slide | What it shows & why it matters |
|---|---|---|
| `01_synthetic_training_data.png` | Synthesizing the training data | Four generated inscriptions with bounding boxes. We had only one image per sign, so we *manufactured* training data — real templates rendered as carved relief with random lighting, wear and noise. Because the program places each glyph, every box and identity is exact: perfect labels, unlimited quantity. **The central technique of the milestone.** |
| `02_detection_works.png` | Detection: solved | A real inscription with nine signs boxed and numbered right-to-left. No multi-stroke splitting, no background boxes. Across 77 inscriptions: 99% coverage, 86% precision, mean 4.9 signs vs a true 5.6. **Our most convincing image** — classical CV could not do this at all. |
| `03_inputs_are_legible.png` | The inputs are clear — so why does recognition fail? | Four inscriptions; top = original, middle = standard binarization, bottom = an alternative we tested. Binarization yields fully legible line drawings. **Rules out "the images were too poor."** |
| `04_recognition_failure.png` | Recognition: where it breaks | Real sign crops (top) above the templates they matched to (bottom). A wheel, a jar and a grid all collapse onto similar simple templates. **Shows the failure concretely** — anyone can see the matches are wrong. |
| `05_synthetic_training_curves.png` | The sim-to-real gap | Training curves reaching 87.9% top-1 / 97.7% top-5 on synthetic data, against 12–17% on real artifacts. **Put this on the same slide as `04`** — a textbook-perfect learning curve beside visibly wrong predictions is the project's strongest argument, and it sets up Milestone 4. |

### Strongly recommended

| File | Slide | What it shows & why it matters |
|---|---|---|
| `06_false_positives_on_damage.png` | Why aggregate metrics mislead | A damaged seal with 13 detections, almost all on cracks, against ~3 surviving signs. The summary "4.5 signs per seal" concealed this. **Found only by inspecting outputs, not summaries.** |
| `07_line_art_blindness.png` | Two findings in one image | A painted illustration with ~7 crisp signs and a single detection — on the bull's ear. Trained only on carved relief, the detector is blind to flat line art. It is also **not a seal photograph at all**: a modern Constitution illustration that contaminated the scraped dataset (3 of 38). |

### Optional

| File | Slide | What it shows |
|---|---|---|
| `08_domain_randomization.png` | Making synthetic data look real | Synthetic seals as stone panels on cloth with mounting clips and perspective — including unlabelled clutter so the model learns to ignore it. |
| `09_one_shot_to_many.png` | Dissolving the one-shot problem | One clean template (left) beside eight synthetic carved variants. 100 variants × 417 signs = a 41,700-image training set from one image per class. |
| `10_unlabelled_font_dataset.png` | A dataset we could not use | 1,840 clean glyph forms whose filenames encode font-grid positions, not sign IDs. Shape-matching recovered labels for only 12 of 1,840 (0.7%). |
| `11_dataset_audit.png` | Audit first | Images-per-class across the inventory — a flat line at 1. Auditing before modelling revealed the one-shot problem that shaped every later decision. |

### If you only have room for three

**`02`** (detection works) + **`04`** (recognition fails) + **`05`** (87.9% synthetic vs 17% real).
Those three tell the complete, honest arc on their own.
