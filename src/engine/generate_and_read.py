"""Generate -> filter for novelty -> render -> read back -> re-check.  (Extends M4 / E8.)

Step 1  GENERATE  the 1,000 Markov and 1,000 Gibbs (M3) sequences from `f8_generate.py`
                  (data/processed/m4/generated.json; trained on the M3 train split only).
Step 2  NOVELTY   compare each generated text (signs only, edges ignored) with every TRAINING inscription:
                    copy        identical to a training inscription
                    near-copy   one edit away (one sign substituted, inserted or deleted)
                    novel       two or more edits from every training inscription
Step 3  GUARDRAIL the M4 perplexity guardrail (same threshold as everywhere else).
                  ACCEPTED = novel AND not flagged.
Step 4  RENDER    accepted texts are drawn as synthetic carved strips (Milestone 1 `strip` renderer, the same
                  one used by make_synthetic_inputs.py). Only texts whose every sign has an IM77 template.
Step 5  READ BACK each image goes through the full engine (M1 -> crosswalk -> M2 -> M3 -> M4) and the read-back
                  is compared with what was generated.

What this does and does not show:
- Novelty is measured against the TRAINING split only (what the generators saw). A "novel" text may still
  occur in the validation/test split or elsewhere; that is reported separately.
- Passing the guardrail means "statistically like real inscriptions", NOT "meaningful". Nobody can check meaning.
- The images are synthetic and drawn with the renderer family M1 was trained on, so the read-back accuracy is
  an UPPER BOUND: on real photographs M1 reads 12-17% of signs correctly.

  python src/engine/generate_and_read.py            (needs: f8_generate.py run first, M1 + M3 weights)
Outputs: reports/engine/generated_roundtrip.json, reports/engine/generated_gallery.png,
         reports/engine/generated_examples/<k>/ (engine reports), data/processed/engine_generated/
"""
import sys, json, collections
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/engine", "src/m1_vision", "src/m3_masking", "src"):
    sys.path.insert(0, str(ROOT / p))
from parse import Engine  # noqa: E402
from synth import load_templates, build_strip  # noqa: E402
from common.signs import wells_to_im77  # noqa: E402

GEN = ROOT / "data/processed/m4/generated.json"
IMG = ROOT / "data/processed/engine_generated"
OUT = ROOT / "reports/engine"
N_RENDER, SEED = 60, 11


class Nearest:
    """Exact / one-edit lookup against a set of sequences, without pairwise edit distances."""
    def __init__(self, seqs):
        self.exact = set(seqs)
        self.sub = {s[:i] + ("*",) + s[i + 1:] for s in seqs for i in range(len(s))}
        self.dele = {s[:i] + s[i + 1:] for s in seqs for i in range(len(s))}

    def kind(self, s):
        if s in self.exact: return "copy"
        if any(s[:i] + ("*",) + s[i + 1:] in self.sub for i in range(len(s))): return "near-copy"   # substitution
        if s in self.dele: return "near-copy"                                                      # one sign missing
        if any(s[:i] + s[i + 1:] in self.exact for i in range(len(s))): return "near-copy"         # one sign extra
        return "novel"


def main():
    eng = Engine(with_vision=True); S = eng.S
    gen = json.load(open(GEN))
    train = Nearest([tuple(s[1:-1]) for s in S.split["train"]])
    heldout = set(tuple(s[1:-1]) for k in ("val", "test") for s in S.split[k])
    tpl = load_templates(); rng = np.random.default_rng(SEED)

    summary, pool = {}, []
    for name, key in (("markov", "mk"), ("gibbs (M3)", "gibbs")):
        seqs = gen[key]
        kinds = [train.kind(tuple(s[1:-1])) for s in seqs]
        p, flag = S.verdict(seqs)
        passed = [f == 0 for f in flag]                     # NaN (too few known signs) counts as not passed
        acc = [k == "novel" and ok for k, ok in zip(kinds, passed)]
        c = collections.Counter(kinds)
        summary[name] = dict(n=len(seqs), copy=c["copy"] / len(seqs), near_copy=c["near-copy"] / len(seqs),
                             novel=c["novel"] / len(seqs),
                             passes_guardrail=float(np.mean(passed)),
                             passes_guardrail_among_novel=float(np.mean([ok for k, ok in zip(kinds, passed) if k == "novel"])),
                             accepted_novel_and_passes=float(np.mean(acc)),
                             accepted_distinct=len({tuple(s[1:-1]) for s, a in zip(seqs, acc) if a}),
                             accepted_found_in_val_or_test=float(np.mean([tuple(s[1:-1]) in heldout for s, a in zip(seqs, acc) if a])),
                             median_pppl_accepted=float(np.nanmedian([x for x, a in zip(p, acc) if a])))
        for s, a, pp in zip(seqs, acc, p):
            if not a: continue
            body = s[1:-1]; wr = body[::-1]                 # reading order
            im = [wells_to_im77(w) for w in wr]
            if all(x and x[0] in tpl for x in im) and 2 <= len(body) <= 10:
                pool.append(dict(generator=name, stored=s, wells_reading=wr, im77_candidates=im, pppl_generated=float(pp)))
        print(f"{name:11s} copies {summary[name]['copy']:.1%}  near-copies {summary[name]['near_copy']:.1%}  novel {summary[name]['novel']:.1%}"
              f"  | passes guardrail {summary[name]['passes_guardrail']:.1%}  | ACCEPTED {summary[name]['accepted_novel_and_passes']:.1%}")

    # ---- render a balanced sample of accepted texts and read them back ----
    seen, uniq = set(), []
    for g in pool:
        if tuple(g["wells_reading"]) not in seen:
            seen.add(tuple(g["wells_reading"])); uniq.append(g)
    picks = []
    for name in ("gibbs (M3)", "markov"):
        cand = [g for g in uniq if g["generator"] == name]
        idx = rng.permutation(len(cand))[:N_RENDER // 2]
        picks += [cand[i] for i in idx]
    (IMG / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for k, g in enumerate(picks):
        img, _ = build_strip([x[0] for x in g["im77_candidates"]], tpl, rng)
        name = f"gen_{k:03d}.png"; cv2.imwrite(str(IMG / "images" / name), img)
        ex = OUT / "generated_examples" / f"gen_{k:03d}" if k < 4 else None
        r = eng.run(image=str(IMG / "images" / name), truth_wells=g["wells_reading"], out=ex,
                    source=f"GENERATED by {g['generator']} (not a real inscription), drawn synthetically, read by M1")
        n_true, n_pred = len(g["wells_reading"]), len(r["m1"])
        row = dict(image=name, generator=g["generator"], generated=g["wells_reading"], read_back=r["reading_order_wells"],
                   n_true=n_true, n_pred=n_pred, verdict_read_back=r["m4"]["verdict"])
        if n_true == n_pred:
            top1 = [s["top5"][0][0] in c for s, c in zip(r["m1"], g["im77_candidates"])]
            row.update(sign_correct=top1, exact=all(top1), wells_identical=r["reading_order_wells"] == g["wells_reading"])
        rows.append(row)
    same = [r for r in rows if "exact" in r]
    consistent = lambda rs: float(np.mean([r["verdict_read_back"].startswith("consistent") for rs_ in [rs] for r in rs_
                                           if not r["verdict_read_back"].startswith("not enough")])) if rs else None
    rt = dict(images=len(rows),
              count_exact=float(np.mean([r["n_true"] == r["n_pred"] for r in rows])),
              sign_top1_on_count_matched=float(np.mean([x for r in same for x in r["sign_correct"]])),
              exact_sequence_on_count_matched=float(np.mean([r["exact"] for r in same])),
              wells_identical_on_count_matched=float(np.mean([r["wells_identical"] for r in same])),
              note_wells=("lower than exact_sequence because several Wells signs share one IM77 template (e.g. 817/861); "
                          "M1 reads IM77, so the crosswalk cannot recover which Wells variant was drawn"),
              read_back_passes_guardrail=consistent(rows),
              read_back_passes_guardrail_when_exact=consistent([r for r in same if r["exact"]]),
              read_back_passes_guardrail_when_wrong=consistent([r for r in rows if not r.get("exact")]))
    res = dict(settings=dict(source=str(GEN.relative_to(ROOT)), novelty_reference="M3 train split",
                             near_copy="one substitution, insertion or deletion", rendered=len(rows), seed=SEED),
               generation=summary, round_trip=rt, rows=rows)
    json.dump(res, open(OUT / "generated_roundtrip.json", "w"), indent=1, default=str)
    print(json.dumps(rt, indent=1))

    # ---- gallery for the report / video: 4 Gibbs + 4 Markov ----
    show = [r for r in rows if r["generator"] == "gibbs (M3)"][:4] + [r for r in rows if r["generator"] == "markov"][:4]
    fig, ax = plt.subplots(len(show), 1, figsize=(8, 1.75 * len(show)))
    for a, r in zip(ax, show):
        a.imshow(cv2.imread(str(IMG / "images" / r["image"]), 0), cmap="gray"); a.axis("off")
        ok = ("read back exactly" if r.get("exact") else "misread") if "exact" in r else "wrong sign count"
        a.set_title(f"{r['generator']} | generated (Wells, reading order): {' '.join(r['generated'])}\n"
                    f"engine read back: {' '.join(r['read_back'])}  ->  {ok} (IM77 level); M4: {r['verdict_read_back'].split(':')[0]}",
                    fontsize=7, loc="left")
    plt.tight_layout(); fig.savefig(OUT / "generated_gallery.png", dpi=130); print("->", OUT / "generated_gallery.png")


if __name__ == "__main__":
    main()
