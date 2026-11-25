"""Builds the four milestone notebooks. They LOAD saved results (no training), so 'Restart & Run All'
takes seconds. Each notebook states the command that regenerates its inputs.

  python notebooks/build_notebooks.py
  jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
SETUP = '''import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
for p in ("src", "src/m3_masking", "src/m4_guardrails", "src/m2_markov", "src/m1_vision"):
    sys.path.insert(0, str(ROOT / p))
pd.set_option("display.width", 160); pd.set_option("display.max_colwidth", 60)
def wilson(k, n, z=1.96):
    p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return c - h, c + h'''


def nb(cells, path):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, HERE / path)


# ------------------------------------------------------------------ M1
nb([
("md", """# Milestone 1 - Visual grapheme extraction

**Goal (proposal):** an image pipeline that takes a photo of a seal inscription and outputs an ordered list of standardized sign IDs.

**Summary:** detection works on real inscription strips; recognition of *which* of the 417 signs does not (17% top-1). This notebook loads saved results only.

*Regenerate:* training scripts in `src/m1_vision/` (see `src/m1_vision/README.md`); real-strip evaluation `python src/m1_vision/read_strip.py --all --review`."""),
("code", SETUP),
("md", "## 1. Dataset audit: one template per sign\nThe Kaggle IM-417-150 set has **exactly one image per sign**, a one-shot problem. It cannot be used as a training set in the usual sense."),
("code", '''tpl = sorted((ROOT / "data/raw/IM_417_150_PNG").glob("*.png"), key=lambda p: int(p.stem))
print("template images:", len(tpl), "| images per class: 1")
fig, ax = plt.subplots(1, 10, figsize=(12, 1.6))
for a, p in zip(ax, [tpl[i] for i in (0, 58, 86, 98, 175, 266, 327, 341, 390, 410)]):
    a.imshow(plt.imread(p), cmap="gray"); a.set_title(p.stem, fontsize=8); a.axis("off")
plt.suptitle("Mahadevan (IM77) sign templates", fontsize=10); plt.show()'''),
("md", "## 2. Synthetic training data\nWe generated labelled training data by composing templates into carved-relief inscriptions. Below: real inscriptions from the M3 test split drawn by the `strip` renderer (used by the engine evaluation)."),
("code", '''imgs = sorted((ROOT / "data/processed/engine_synthetic/images").glob("*.png"))[:4]
fig, ax = plt.subplots(1, len(imgs), figsize=(14, 2.2))
for a, p in zip(ax, imgs): a.imshow(plt.imread(p), cmap="gray"); a.axis("off"); a.set_title(p.name, fontsize=8)
plt.show()'''),
("md", "## 3. Detector training curves (YOLOv8n, single class `sign`)\nLearning curves and loss charts from `runs/sign_detector/results.csv`."),
("code", '''r = pd.read_csv(ROOT / "runs/sign_detector/results.csv"); r.columns = [c.strip() for c in r.columns]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for c in ("train/box_loss", "val/box_loss"): ax[0].plot(r.epoch, r[c], label=c)
ax[0].set(title="box loss", xlabel="epoch"); ax[0].legend()
for c in ("train/cls_loss", "val/cls_loss"): ax[1].plot(r.epoch, r[c], label=c)
ax[1].set(title="classification loss", xlabel="epoch"); ax[1].legend()
for c in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"): ax[2].plot(r.epoch, r[c], label=c)
ax[2].set(title="validation mAP (synthetic)", xlabel="epoch"); ax[2].legend(); plt.tight_layout(); plt.show()
print(f"final mAP50 {r['metrics/mAP50(B)'].iloc[-1]:.3f}, mAP50-95 {r['metrics/mAP50-95(B)'].iloc[-1]:.3f} (synthetic validation)")'''),
("md", "## 4. Recognizer training curves (MobileNetV3-Small, 417 classes, synthetic crops)"),
("code", 'display(Image(filename=str(ROOT / "reports/figures/recognizer_curves.png"), width=900))'),
("md", "**Validation is synthetic only.** No cross-validation on real data is possible, because there is no labelled real data. The real-data evaluation below is a hand review."),
("md", "## 5. Real inscription strips (77 strips, Indus-Seal-Dataset)"),
("code", '''s = pd.read_csv(ROOT / "reports/strip_eval.csv")
print(f"strips: {len(s)} | with >=1 detection: {(s.n_signs>0).mean():.0%} | signs per strip: mean {s.n_signs.mean():.1f}, median {s.n_signs.median():.0f}")
rows = []
for name, f in (("baseline recognizer", "strip_review_baseline.csv"), ("+ strip-domain crops", "strip_review_recog.csv")):
    v = pd.read_csv(ROOT / "reports" / f).verdict.value_counts(); n = len(pd.read_csv(ROOT / "reports" / f))
    signs = n - v.get("not_a_sign", 0); t1 = v.get("top1_correct", 0); t5 = t1 + v.get("top5_correct", 0)
    rows.append(dict(attempt=name, judged=n, detection_precision=f"{signs/n:.0%}",
                     top1=f"{t1/signs:.0%} [{wilson(t1,signs)[0]:.0%}-{wilson(t1,signs)[1]:.0%}]",
                     top5=f"{t5/signs:.0%} [{wilson(t5,signs)[0]:.0%}-{wilson(t5,signs)[1]:.0%}]"))
display(pd.DataFrame(rows))'''),
("md", """Hand-judged random samples of 70 detections, with 95% Wilson intervals. Chance top-1 over 417 signs is 0.24%.

Two further attempts are not in the table. A relief-to-line-art U-Net collapsed to 17 sign IDs, and binarization was clearly worse; both are documented in `reports/MILESTONE1_SUMMARY.md`."""),
("md", "## 6. The sim-to-real gap, measured end to end\nThe same reader applied to real inscriptions drawn synthetically (truth known) vs real photographs (`src/engine/eval_synthetic.py`)."),
("code", '''e = json.load(open(ROOT / "reports/engine/synthetic_eval.json")); rr = e["recognition_on_count_matched"]
display(pd.DataFrame([dict(input="synthetic strips (truth known)", signs=rr["signs"], top1=f"{rr['top1']:.1%}", top5=f"{rr['top5']:.1%}", exact_sequence=f"{rr['exact_sequence']:.0%}"),
                      dict(input="real strips, saved (strip-domain) recognizer, hand review", signs=60, top1="12%", top5="40%", exact_sequence="-"),
                      dict(input="real strips, baseline recognizer (weights overwritten), hand review", signs=54, top1="17%", top5="41%", exact_sequence="-")]))
print("The synthetic row and the first real row use the same saved weights, so they are the like-for-like comparison.")'''),
("md", """## Limitations
- There are no ground-truth transcriptions for the real strips, so recognition is measured by hand review of samples only.
- All training data is synthetic, and the gap between synthetic (88%) and real (17%) accuracy is the main finding.
- The pipeline assumes impression-orientation input (a mirror test on 262 crops found no evidence the strips are mirrored).
- The crosswalk maps 334 of the 417 IM77 signs to Wells numbers; the other 83 have no equivalent."""),
], "01_m1_vision.ipynb")

# ------------------------------------------------------------------ M2
nb([
("md", """# Milestone 2 - Positional bias and Markov chains

The full Milestone 2 analysis on the CISI corpus (Parpola numbering) is the executed notebook `src/m2_markov/m2_markov.ipynb`. This notebook gathers its outputs and the integration work:
1. the replication on the larger Wells corpus;
2. the external check against Mahadevan (1977);
3. the embedding comparison (word2vec / LLM2Vec / M3 encoder).

*Regenerate:* `python src/m2_markov/replicate_wells.py`, `python src/m2_markov/mahadevan_table1.py`, `python src/m2_markov/llm2vec.py --table`."""),
("code", SETUP),
("md", "## 1. Headline results of the M2 notebook (CISI)"),
("code", '''m2 = json.load(open(ROOT / "src/m2_markov/outputs/results_summary.json"))
print(json.dumps({k: m2[k] for k in ("step2_positions", "step4_smoothing", "step5_direction")}, indent=1)[:2500])'''),
("code", '''for f in ("m2_04_flip_test.png", "m2_06_witten_bell_reserve.png", "m2_07_positional_entropy.png"):
    display(Image(filename=str(ROOT / "src/m2_markov/outputs/figures" / f), width=760))'''),
("md", "## 2. Cross-validation of the smoothing (10-fold, by inscription): from the M2 notebook\nPlain MLE gives probability 0 to 46% of held-out pairs; Witten–Bell gives the lowest held-out perplexity."),
("code", 'display(pd.DataFrame(m2["step4_smoothing"]).T)'),
("md", "## 3. Replication on the Wells corpus (M3/M4 data, all sites)\nThe same definitions applied to both corpora. The CISI column must reproduce M2's numbers, which checks that the re-implementation is faithful."),
("code", '''r = json.load(open(ROOT / "reports/m2/replication_wells.json"))
keys = ["complete_texts", "top_ender_label", "top_ender_share", "top_beginner_label", "flip_test_jsd", "flip_test_null_max", "pos1_gap", "pos1_gap_null_max",
        "heldout_ppl_forward", "heldout_ppl_reversed", "folds_reading_order_better", "significant_collocations"]
display(pd.DataFrame({c: [r[c][k] for k in keys] for c in r}, index=keys))'''),
("md", "## 4. External check: Mahadevan (1977), Table I"),
("code", '''t = json.load(open(ROOT / "reports/m2/mahadevan_table1_check.json"))
print(f"rows recovered from OCR: {t['rows_recovered']}/417, tokens {t['tokens_recovered']}/{t['tokens_published']}")
display(pd.DataFrame({"Mahadevan": t["jar_sign_IM77_342"]["mahadevan_shares"], "our corpus": t["jar_sign_IM77_342"]["our_shares"]}).T)  # jar sign (IM77 342), share of positions, reading order
display(pd.DataFrame({"Mahadevan": t["mahadevan_aggregate_position_shares"], "our corpus": t["our_corpus_aggregate_position_shares"]}).T)'''),
("md", "## 5. Sign embeddings: word2vec vs LLM2Vec vs the M3 encoder (3 runs each)"),
("code", 'display(Markdown((ROOT / "reports/m2/embedding_comparison.md").read_text().split("**How to read")[0]))'),
], "02_m2_markov.ipynb")

# ------------------------------------------------------------------ M3
nb([
("md", """# Milestone 3 - Masked-token encoder

Predict hidden or damaged signs from context, with calibrated confidence. This notebook shows the learning curves (token-prediction loss), the model search, the held-out results, **5-fold cross-validation** and calibration.

*Regenerate:* `cd src/m3_masking && python run_all.py` (search + 3 seeds) and `python src/m3_masking/cv.py`."""),
("code", SETUP),
("md", "## 1. Learning curves: token-prediction loss, 3 seeds\nTraining loss is the masked-sign cross-entropy; validation is leave-one-out NLL on the validation split."),
("code", '''fig, ax = plt.subplots(figsize=(8, 4))
for s in range(3):
    h = json.load(open(ROOT / f"src/m3_masking/artifacts/history_seed{s}.json"))
    ax.plot([x[1] for x in h], alpha=.6, label=f"train seed {s}"); ax.plot([x[2] for x in h], "--", label=f"val seed {s}")
ax.set(xlabel="epoch", ylabel="NLL (nats)", title="Masked-sign loss"); ax.legend(fontsize=8); plt.show()'''),
("md", "## 2. Model search (one seed per configuration, validation NLL)"),
("code", 'display(pd.DataFrame([{**r["cfg"], "val_nll": r["val_nll"], "epochs": r["epochs"]} for r in json.load(open(ROOT / "src/m3_masking/artifacts/search.json"))]).sort_values("val_nll"))'),
("md", "## 3. Held-out test split (350 inscriptions, 1,304 hidden signs)"),
("code", '''res = json.load(open(ROOT / "src/m3_masking/artifacts/results.json"))
L = res["protocols"]["loo"]; f = lambda v: v[0] if isinstance(v, list) else v
display(pd.DataFrame({k: {m: round(f(v[m]), 4) for m in ("top1", "top3", "top5", "ppl", "ece")} for k, v in L.items()}).T)'''),
("md", "## 4. Five-fold cross-validation (all 3,491 unique sequences, 12,525 predictions)"),
("code", '''cv = json.load(open(ROOT / "reports/m3/cv.json")); m = cv["metrics"]; e = cv["ensemble_cv"]
rows = [dict(model=k, top1=f"{100*m[k]['top1']['mean']:.1f}% ± {100*m[k]['top1']['sd']:.1f}", top5=f"{100*m[k]['top5']['mean']:.1f}% ± {100*m[k]['top5']['sd']:.1f}",
             perplexity=f"{m[k]['ppl']['mean']:.1f} ± {m[k]['ppl']['sd']:.1f}") for k in ("unigram", "bigram", "transformer")]
ens5 = [f_["ensemble"]["top5"] for f_ in e["per_fold"]]
rows.append(dict(model="ensemble of 3 (deliverable)", top1=f"{100*e['ensemble_top1']['mean']:.1f}% ± {100*e['ensemble_top1']['sd']:.1f}",
                 top5=f"{100*np.mean(ens5):.1f}% ± {100*np.std(ens5, ddof=1):.1f}", perplexity=f"{e['ensemble_ppl']['mean']:.1f} ± {e['ensemble_ppl']['sd']:.1f}"))
display(pd.DataFrame(rows)); p = e["pooled"]
print(f"ensemble vs bigram, pooled: {p['only_ensemble_right']} vs {p['only_bigram_right']} disagreements, McNemar p = {p['mcnemar_p']:.3f}; "
      f"perplexity gap {p['ppl_gap_bigram_minus_ensemble']:.2f} (95% CI {p['ppl_gap_ci95'][0]:.2f}-{p['ppl_gap_ci95'][1]:.2f}); folds won: top-1 {e['folds_ensemble_better_top1']}/5, perplexity {e['folds_ensemble_better_ppl']}/5")'''),
("md", "**Reading:** a single Transformer is no better than the Witten–Bell bigram. The ensemble of three beats it by a small, consistent margin."),
("md", "## 5. Calibration and confidence bands"),
("code", '''t = res["ece_table_calibrated"]
fig, ax = plt.subplots(figsize=(4.5, 4)); ax.plot([r[3] for r in t], [r[4] for r in t], "o-"); ax.plot([0, 1], [0, 1], "k:")
ax.set(xlabel="stated confidence", ylabel="accuracy", title="Reliability (test)"); plt.show()
display(pd.DataFrame(res["confidence_bands_test"]).T)'''),
("md", "## 6. Using the model (`predict.py`), including Milestone 1-style input"),
("code", '''from predict import SignCompleter
sc = SignCompleter()
for args in (("+740-000-590+", {}), (["P385", "?", "P324"], dict(numbering="parpola", order="reading"))):
    r = sc.complete(args[0], top_k=3, **args[1])
    print(r["completed_in_input_numbering_and_order"], "|", [(c["sign_in_input_numbering"], round(c["prob"], 2)) for c in r["gaps"][0]["candidates"]], r["gaps"][0]["rating"])'''),
], "03_m3_masking.ipynb")

# ------------------------------------------------------------------ M4
nb([
("md", """# Milestone 4 - Benchmarking semantic hallucinations

Every claim is tested against **structural controls** (shuffled, frequency-random, uniform-random and Markov-generated inscriptions). The full write-up is `reports/M4_REPORT.md`; this notebook loads the saved results.

*Regenerate:* the M4 block of `run_all.sh`."""),
("code", SETUP),
("md", "## E1. Perplexity vs noise baselines (medians with bootstrap 95% CIs)"),
("code", '''s = json.load(open(ROOT / "reports/m4/perplexity_summary.json"))
display(pd.DataFrame({k: {"Transformer": f"{v['median_ppl_transformer']:.1f} [{v['median_ppl_transformer_ci95'][0]:.1f}-{v['median_ppl_transformer_ci95'][1]:.1f}]",
                          "bigram": f"{v['median_ppl_bigram']:.1f} [{v['median_ppl_bigram_ci95'][0]:.1f}-{v['median_ppl_bigram_ci95'][1]:.1f}]"} for k, v in s.items()}).T)
display(Image(filename=str(ROOT / "reports/m4/perplexity_vs_noise.png"), width=900))'''),
("md", "## E2. Confidence scoring against controls"),
("code", 'display(pd.DataFrame({k: v["confidence"] for k, v in s.items()}).T)'),
("md", "## E3. Naive Bayes guardrail (5-fold CV) and the perplexity guardrail"),
("code", '''nb_ = json.load(open(ROOT / "reports/m4/nb_guardrail.json"))
display(pd.DataFrame({k: {"accuracy": round(v["accuracy"], 3), "AUC": round(v["auc"], 3), "AUC sd": round(v["auc_sd"], 3)} for k, v in nb_.items()}).T)
f7 = json.load(open(ROOT / "reports/m4/f7_guardrail_on_m1.json"))
display(pd.DataFrame({k: {"flag rate": f"{v['flag_rate']:.1%}", "flag 95% CI": [round(x, 3) for x in v["flag_rate_ci95"]], "AUC": v["auc_vs_real"], "AUC 95% CI": v["auc_ci95"]} for k, v in f7["calibration_on_test"].items()}).T)'''),
("md", "## E4. Synthetic corpus vs real inscriptions (numbering-free)"),
("code", '''sc_ = json.load(open(ROOT / "reports/m4/synthetic_check.json"))
keys = ["mean_length", "distinct_signs", "top10_share", "adjacent_repeat_rate", "unigram_entropy_bits", "conditional_entropy_bits"]
display(pd.DataFrame({"real (M3 corpus)": [sc_["real_m3_corpus"][k] for k in keys], "synthetic": [sc_["synthetic"][k] for k in keys]}, index=keys))
display(Image(filename=str(ROOT / "reports/m4/synthetic_vs_real.png"), width=900))'''),
("md", "## E5. A proposed reading (the Sanskrit columns) against structural controls"),
("code", 'print(json.dumps({k: v for k, v in json.load(open(ROOT / "reports/m4/f5_reading_test.json")).items() if k != "T3_freedom_per_sign"}, indent=1))'),
("md", "## E6. Error propagation from Milestone 1"),
("code", '''f6 = json.load(open(ROOT / "reports/m4/f6_error_propagation.json"))
display(pd.DataFrame(f6["rows"])[["strength", "direction", "kl", "m3_top1", "flag"]].round(3))
display(Image(filename=str(ROOT / "reports/m4/f6_error_propagation.png"), width=950))'''),
("md", "## E7. The guardrail on Milestone 1's real output"),
("code", '''print({k: f7[k] for k in ("m1", "m1_top1", "m1_reranked_by_m3")})
display(Image(filename=str(ROOT / "reports/m4/f7_guardrail_on_m1.png"), width=800))'''),
("md", "## E8. Generation (the professor's request)"),
("code", 'display(pd.DataFrame(json.load(open(ROOT / "reports/m4/f8_generation.json"))["results"]).T)'),
("md", "## The integrated engine: synthetic vs real input"),
("code", '''a = json.load(open(ROOT / "reports/engine/synthetic_eval.json")); b = json.load(open(ROOT / "reports/engine/real_eval.json"))
print("synthetic strips:", {k: a[k] for k in ("detection", "recognition_on_count_matched", "m4_flag_rate")})
print("real strips     :", {k: b[k] for k in ("strips", "scored", "verdicts", "flag_rate")})
display(Image(filename=str(ROOT / "reports/engine/demo_typed/panels.png"), width=950))'''),
("md", "## E8b. New inscriptions: novelty filter, rendered and read back (`src/engine/generate_and_read.py`)"),
("code", '''g = json.load(open(ROOT / "reports/engine/generated_roundtrip.json"))
display(pd.DataFrame(g["generation"]).T[["copy", "near_copy", "novel", "accepted_novel_and_passes", "median_pppl_accepted"]].round(3))
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in g["round_trip"].items() if k != "note_wells"})
display(Image(filename=str(ROOT / "reports/engine/generated_gallery.png"), width=800))'''),
], "04_m4_guardrails.ipynb")
print("notebooks written")
