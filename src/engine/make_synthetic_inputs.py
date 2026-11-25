"""Synthetic test inputs for the engine: REAL inscriptions rendered as synthetic strip images.

Each image is drawn from a real held-out inscription (Milestone 3 v2 TEST split, never used for
training M2/M3), so the downstream milestones see a genuine sign sequence and the true answer
is known for every sign. The pictures themselves are synthetic (Milestone 1's `strip` renderer:
carved relief from the 417 Mahadevan templates), so this measures the pipeline UNDER THE
CONDITIONS WHERE MILESTONE 1 WORKS. It says nothing about real photographs.

Selection (stated, not tuned): test inscriptions with 2-10 signs, no illegible sign, every sign
in the M3 vocabulary and mappable to at least one IM77 template. When a Wells sign maps to
several IM77 signs, the first (lowest-numbered) template is drawn and every candidate counts as
correct when scoring.

  python src/engine/make_synthetic_inputs.py --n 100
Output: data/processed/engine_synthetic/{images/*.png, truth.json}
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/m1_vision", "src/m3_masking", "src"):
    sys.path.insert(0, str(ROOT / p))
from synth import load_templates, build_strip  # noqa: E402
from data import load_splits  # noqa: E402
from common.signs import wells_to_im77  # noqa: E402

OUT = ROOT / "data" / "processed" / "engine_synthetic"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    vocab, split, _ = load_splits(ROOT / "src/m3_masking/artifacts")
    tpl = load_templates(); rng = np.random.default_rng(a.seed)
    cands = []
    for s in split["test"]:
        body = s[1:-1]
        if not (2 <= len(body) <= 10) or any(t == "000" or t not in vocab.stoi for t in body): continue
        im = [wells_to_im77(t) for t in body]
        if any(not x or x[0] not in tpl for x in im): continue
        cands.append((s, body[::-1], [x for x in im][::-1]))       # reading order
    rng.shuffle(cands)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    truth = []
    for k, (stored, wells_r, im_r) in enumerate(cands[:a.n]):
        img, recs = build_strip([x[0] for x in im_r], tpl, rng)
        name = f"syn_{k:03d}.png"
        cv2.imwrite(str(OUT / "images" / name), img)
        truth.append(dict(image=name, stored_tokens=stored, wells_reading=wells_r,
                          im77_reading_drawn=[x[0] for x in im_r], im77_reading_candidates=im_r))
    json.dump(dict(source="M3 v2 test split", selected=len(truth), eligible=len(cands), seed=a.seed, items=truth),
              open(OUT / "truth.json", "w"), indent=1)
    print(f"eligible test inscriptions: {len(cands)}; rendered {len(truth)} -> {OUT}")


if __name__ == "__main__":
    main()
