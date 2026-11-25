"""Screen-recordable demo of the whole project (about 2-3 minutes). Loads all models once.

  python demo/demo.py              runs every step, pausing for Enter between steps
  python demo/demo.py --no-pause   runs straight through
  python demo/demo.py --open       also opens each step's HTML report in the browser

Every step writes the engine's full report to reports/demo/<step>/ (index.html, panels.png, report.json).
"""
import sys, os, json, argparse, webbrowser, time, warnings
warnings.filterwarnings("ignore")          # hide library warnings so the recording stays clean
if os.name == "nt": os.system("")           # enable colours in the Windows terminal
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[1]
for p in ("src/engine", "src/m1_vision", "src/m3_masking", "src/m4_guardrails", "src"):
    sys.path.insert(0, str(ROOT / p))
OUT = ROOT / "reports/demo"

B, D, G, R, Y, C, X = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"
INSCRIPTION = ["817", "002", "013", "840", "176", "100", "740"]   # a real held-out inscription (Wells, reading order)
SYN_IMAGE = ROOT / "data/processed/engine_synthetic/images/syn_016.png"   # the same inscription, drawn synthetically
REAL_PHOTO = ROOT / "data/raw/isd/1/1text.jpg"                            # a real inscription strip (Indus-Seal-Dataset)


def banner(n, title, what):
    print(f"\n{B}{C}{'=' * 78}\n STEP {n}  {title}\n{'=' * 78}{X}\n{D}{what}{X}\n")


def verdict_colour(v):
    return (G if v.startswith("consistent") else R if v.startswith("flagged") else Y) + v + X


def show(rep, truth=None):
    print(f"  input source : {rep['source']}")
    print(f"  reading order: {B}{' '.join(rep['reading_order_wells'])}{X}  (Wells numbering, right-to-left)")
    if truth is not None:
        from common.signs import wells_to_im77
        print(f"  ground truth : {' '.join(truth)}")
        if rep.get("m1") is not None and len(rep["m1"]) == len(truth):
            ok = [m["top5"][0][0] in wells_to_im77(t) for m, t in zip(rep["m1"], truth)]
            marks = " ".join((G + "ok" + X) if o else (R + "XX" + X) for o in ok)
            print(f"  M1 per sign  : {marks}   -> {G + 'every sign read correctly' + X if all(ok) else R + f'{len(ok) - sum(ok)} sign(s) misread' + X}")
            if all(ok) and rep["reading_order_wells"] != truth:
                print(f"  {D}(Wells numbers can differ: some Wells signs are variants drawn with the same Mahadevan template,{X}\n"
                      f"  {D} e.g. 817/861, so an image cannot tell them apart.){X}")
        elif rep.get("m1") is not None:
            print(f"  M1           : {R}found {len(rep['m1'])} signs, truth has {len(truth)}{X}")
    print(f"  M2  sign roles: {', '.join(f'{w}={r}' for w, r in zip(rep['reading_order_wells'], rep['m2_roles']))}")
    print(f"  M2  surprise   : {rep['m2_total_bits']:.1f} bits")
    p = rep["m4"]["pseudo_perplexity"]
    print(f"  M4  verdict    : {verdict_colour(rep['m4']['verdict'])}"
          + (f"  (pseudo-perplexity {p:.0f}; threshold {rep['m4']['threshold']:.0f}; real texts median {rep['m4']['median_real']:.0f})" if p else ""))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-pause", action="store_true"); ap.add_argument("--open", action="store_true")
    ap.add_argument("--seed", type=int, default=5, help="seed for the shuffle, random signs and generation")
    a = ap.parse_args()

    def step_end(path=None):
        if path and a.open: webbrowser.open((path / "index.html").as_uri())
        if path: print(f"\n  {D}full report -> {path.relative_to(ROOT)}/index.html{X}")
        if not a.no_pause: input(f"\n{D}[Enter] next step{X}")

    print(f"{B}Indus Epigraphy AI - live demo{X}\n{D}loading M1 (YOLO + recognizer), M2 (Markov), M3 (Transformer ensemble), M4 (guardrail)...{X}")
    t0 = time.time()
    from parse import Engine
    eng = Engine(with_vision=True)
    print(f"{D}ready in {time.time() - t0:.0f}s{X}")
    rng = np.random.default_rng(a.seed)

    banner(1, "A real inscription, typed in", "M2 labels each sign's position role; M4 checks whether the text looks like real Indus text.")
    rep = eng.run(seq=INSCRIPTION, out=OUT / "1_typed"); show(rep); step_end(OUT / "1_typed")

    banner(2, "Damage each sign in turn: can M3 recover it?",
           "M3 (masked-sign Transformer) hides one sign at a time and predicts it from the others, with calibrated confidence.")
    hits = 0
    for i, true in enumerate(INSCRIPTION):
        dmg = list(INSCRIPTION); dmg[i] = "?"
        g = eng.sc.complete(dmg, top_k=3, numbering="wells", order="reading")["gaps"][0]
        c = [x["sign_in_input_numbering"] for x in g["candidates"]]; hits += c[0] == true
        res = (G + "correct" + X) if c[0] == true else (Y + "in top 3" + X) if true in c else (R + "missed" + X)
        print(f"  {' '.join(dmg):32s} best guess {c[0]} (p {g['candidates'][0]['prob']:.2f}, {g['rating']:6s})  true {true}  {res}")
    print(f"\n  recovered {hits} of {len(INSCRIPTION)}.  {D}Over all held-out data: 33% first guess, 56% in top 5; 'High' guesses are right 75% of the time.{X}")
    step_end()

    banner(3, "The same signs, shuffled", "Same signs, wrong order. A real structure test must notice.")
    sh = list(INSCRIPTION); rng.shuffle(sh)
    rep = eng.run(seq=sh, out=OUT / "3_shuffled", source="CONTROL: shuffled real inscription"); show(rep)
    print(f"  {Y}The score is much worse than the original's (step 1), but a 7-sign shuffle can still fall under the threshold:{X}\n"
          f"  {Y}the guardrail flags only ~22% of shuffled texts. It is built to catch noise, not every disorder.{X}")
    step_end(OUT / "3_shuffled")

    banner(4, "Random signs", "Signs drawn at random from the sign list: noise that looks superficially like an inscription.")
    signs = [t for t in eng.vocab.itos if t.isdigit()]
    rnd = list(rng.choice(signs, len(INSCRIPTION)))
    rep = eng.run(seq=rnd, out=OUT / "4_random", source="CONTROL: random signs"); show(rep); step_end(OUT / "4_random")

    banner(5, "From an image: the inscription drawn as a synthetic carved strip",
           "M1 detects and identifies every sign, the crosswalk converts numberings, then M2 -> M3 -> M4.")
    rep = eng.run(image=str(SYN_IMAGE), truth_wells=INSCRIPTION, out=OUT / "5_image_synthetic"); show(rep, INSCRIPTION)
    step_end(OUT / "5_image_synthetic")

    banner(6, "A REAL photograph", "Detection works on real strips; recognition does not (12-17% of signs right). The guardrail should catch it.")
    rep = eng.run(image=str(REAL_PHOTO), out=OUT / "6_image_real"); show(rep)
    print(f"  {Y}No ground truth exists for this photo; the guardrail's job is to say the reading is not trustworthy.{X}")
    step_end(OUT / "6_image_real")

    banner(7, "Generate a NEW inscription, draw it, and read it back",
           "M3 writes a sequence (Gibbs sampling); we reject copies of training texts, check it with M4,\n"
           " render it as a carved strip, and let the engine read it back.")
    import f8_generate as f8
    from synth import load_templates, build_strip
    from common.signs import wells_to_im77
    from generate_and_read import Nearest
    f8.N = 40
    S = eng.S; grng = np.random.default_rng(a.seed)
    gen = f8.gibbs(S, f8.markov(S, grng), grng)
    train = Nearest([tuple(s[1:-1]) for s in S.split["train"]])
    tpl = load_templates(); pick = None
    for s in gen:
        wr = s[1:-1][::-1]; im = [wells_to_im77(w) for w in wr]
        if 3 <= len(wr) <= 8 and train.kind(tuple(s[1:-1])) == "novel" and all(x and x[0] in tpl for x in im) \
                and eng.m4_verdict(s)["verdict"].startswith("consistent"):
            pick = (wr, im); break
    if pick is None:
        print("  no accepted sample in this batch; run again"); return
    wr, im = pick
    print(f"  generated    : {B}{' '.join(wr)}{X}  (not in the training corpus, not one edit away from any training text)")
    img, _ = build_strip([x[0] for x in im], tpl, grng)
    (OUT / "7_generated").mkdir(parents=True, exist_ok=True); cv2.imwrite(str(OUT / "7_generated" / "generated.png"), img)
    rep = eng.run(image=str(OUT / "7_generated" / "generated.png"), truth_wells=wr, out=OUT / "7_generated",
                  source="GENERATED by M3 (not a real inscription), drawn synthetically, read by M1")
    show(rep, wr)
    print(f"  {Y}Passing M4 means 'statistically like real inscriptions', not 'meaningful'.{X}")
    step_end(OUT / "7_generated")
    print(f"\n{B}done.{X}  All reports: reports/demo/")


if __name__ == "__main__":
    main()
