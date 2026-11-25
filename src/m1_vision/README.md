# Milestone 1 - Visual grapheme extraction

**Goal (proposal):** a script that takes a photo of a seal inscription and outputs an ordered list of standardized sign IDs.

**Result:**
- **Detection works.** On real inscription strips, 76 of 77 strips get at least one detection, and 86% of detections are real signs (hand-judged).
- **Recognition does not.** Our best result is 17% top-1 / 41% top-5 over 417 signs, against 0.24% chance. Every model scored near-perfectly on synthetic data and failed on real data, and this sim-to-real gap is the main finding.

The full story is in `reports/MILESTONE1_SUMMARY.md`. All figures and tables load from saved results in `notebooks/01_m1_vision.ipynb`.

## Why the design looks like this
The Kaggle IM-417-150 set has **one template image per sign** (417 images, 150x150), which makes recognition a one-shot problem. There are also no labelled real seals. So **all training data is synthetic**: we rendered the real templates as carved relief under random lighting, wear, blur and perspective.

## Pipeline (`read_strip.py`, the deliverable)
```
inscription strip image
  -> YOLOv8n detector, single class "sign"      (runs/sign_detector/weights/best.pt)
  -> MobileNetV3-Small recognizer, 417 classes  (runs/recognizer/best.pt)
  -> boxes sorted right-to-left = reading order -> IM77 (Mahadevan) sign IDs
```
Output is in IM77 numbering. Converting to the Wells numbering used by M2-M4 is done by `src/common/signs.py`: 334 of 417 IM77 signs have a Wells equivalent; the other 83 become unknown slots. The integrated engine `src/engine/parse.py` runs this conversion and then M2-M4.

## Files
| file | role |
|---|---|
| `config.py` | paths, seed |
| `audit.py` | dataset audit (found the one-image-per-class problem) |
| `preprocess.py` | Otsu preprocessing, augmentation helpers |
| `synth.py` | synthetic inscriptions; styles `clean`, `realistic`, `photo`, `strip` |
| `prepare_yolo.py` | YOLO train/val split + `data.yaml` |
| `train_yolo.py` | trains the detector |
| `synth_signs.py` | synthetic single-sign crops for the recognizer (`data/processed/sign_crops.npz`) |
| `train_recognizer.py` | trains the 417-way recognizer, writes `reports/figures/recognizer_curves.png` |
| `recognizer.py` | recognizer inference wrapper |
| `template_match.py` | one-shot template matcher (no training); alternative to the CNN (`--match template`) |
| `synth_pairs.py`, `train_denoiser.py`, `denoiser.py` | relief-to-line-art U-Net (failed experiment: collapsed to 17 sign IDs) |
| `read_strip.py` | **deliverable**: strip image -> sign IDs; `--all --review` builds the hand-review page |
| `read_seal.py`, `batch_eval.py`, `make_review.py` | earlier pipeline on whole museum photos, plus its evaluation tools |
| `detect_signs.py`, `crop_and_match.py` | classical-CV baselines (they document why classical segmentation fails) |
| `fetch_real_seals.py`, `make_pairs_figure.py` | Wikimedia downloader with licence recording; figure helper |

## Reproduce
Training is not part of `run_all.sh`. It needs a GPU or Apple `mps` and takes hours.
```bash
# detector (the saved run used: strip style, 2,500 images, 40 epochs, imgsz 640, batch 16, device mps)
python src/m1_vision/synth.py --n 2500 --style strip --single-class
python src/m1_vision/prepare_yolo.py --style strip
python src/m1_vision/train_yolo.py --epochs 40 --device mps
# recognizer (41,700 crops = 100 per sign)
python src/m1_vision/synth_signs.py --per-class 100
python src/m1_vision/train_recognizer.py --device mps
# read real strips (Indus-Seal-Dataset, 77 text crops, stored in data/raw/isd/)
python src/m1_vision/read_strip.py <strip.jpg>
python src/m1_vision/read_strip.py --all --review
```

## Results
| measure | value | source |
|---|---|---|
| detector, synthetic validation | mAP50 0.995, mAP50-95 0.875 (final epoch) | `runs/sign_detector/results.csv` |
| recognizer, synthetic validation | 87.9% top-1 / 97.7% top-5 | `reports/figures/recognizer_curves.png` |
| real strips with >=1 detection | 76 / 77 (99%); mean 4.9 signs per strip (corpus mean 5.6) | `reports/strip_eval.csv` |
| hand review, baseline recognizer (70 detections) | precision 77%, top-1 17% [9-29%], top-5 41% [29-54%] | `reports/strip_review_baseline.csv` |
| hand review, strip-domain recognizer (70 detections) | precision 86%, top-1 12% [6-22%], top-5 40% [29-53%] | `reports/strip_review_recog.csv` |
| same reader on real inscriptions drawn synthetically | 87.9% top-1, 70% exact sequences | `reports/engine/synthetic_eval.json` |

Brackets are 95% Wilson intervals. Chance top-1 is 0.24%.

## Known issues
- **The 17% baseline recognizer weights no longer exist.** `runs/recognizer/best.pt` (saved 15:27) was written after the baseline review (15:21) and before the second review (15:37). It is therefore the strip-domain model, with 12% top-1 and 40% top-5, and that is the model used by the engine and by M4 E7. The two reviews' top-5 results (41% vs 40%) are within noise of each other.
- `reports/MILESTONE1_SUMMARY.md` quoted detector mAP50-95 as 0.855. The saved run's `results.csv` gives 0.875.
- The training commands above were reconstructed from the file names in `data/processed/yolo/` and from `runs/sign_detector/args.yaml`. The exact `synth_signs.py --strip-frac` value used for the saved recognizer was not recorded.
- `runs/sign_detector/args.yaml` contains an absolute local path.
- There are no ground-truth transcriptions for real strips, so real-data accuracy comes from hand review only.
