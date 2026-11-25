"""Read isolated inscription strips (Indus-Seal-Dataset '*text*' crops).

These images ARE the inscription: one horizontal row of signs, no animal motif,
no museum background, no mounting clips. So the pipeline reduces to:

    strip image -> detect sign boxes -> classify each against the 417 -> order R-to-L

Usage:
  python src/m1_vision/read_strip.py --all                  # every strip + CSV
  python src/m1_vision/read_strip.py data/raw/isd/1/1text.jpg
  python src/m1_vision/read_strip.py --all --review         # + hand-verification sheet
"""
from __future__ import annotations
import argparse, base64, csv, collections, glob
from pathlib import Path
import numpy as np
import cv2

from config import RAW_DIR, FIG_DIR, ROOT
from template_match import load_bank, classify as tclassify
import recognizer as _rec
import denoiser as _den

STRIPS = sorted(glob.glob(str(RAW_DIR / "isd" / "*" / "*text*")))
TPL_DIR = RAW_DIR / "IM_417_150_PNG"
OUT_DIR = FIG_DIR / "strips"
WEIGHTS = ROOT / "runs" / "sign_detector" / "weights" / "best.pt"
MINW = 420          # upscale small strips so the detector sees enough pixels


def load_strip(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    if img.shape[1] < MINW:                      # many crops are tiny
        f = MINW / img.shape[1]
        img = cv2.resize(img, (MINW, int(img.shape[0] * f)), interpolation=cv2.INTER_CUBIC)
    return img


def read_one(path, model, ids, bank, rec, conf, den=None, binarize=False):
    img = load_strip(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if binarize:
        # Otsu over the FULL strip: it sees the whole ink/stone range and picks a
        # good threshold. Per-crop Otsu on a 40x60 fragment has no range to work
        # with and fails. Result is dark-ink-on-white, same convention as templates.
        gg = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        _, bw = cv2.threshold(gg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (bw < 128).mean() > 0.5:            # signs must be the dark minority
            bw = 255 - bw
        gray = bw
    res = model(img, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return {"image": Path(path).name, "n": 0, "seq": [], "panels": [], "overlay": img}
    boxes = [tuple(map(int, b)) for b in res.boxes.xyxy.cpu().numpy()]
    confs = res.boxes.conf.cpu().numpy().tolist()
    order = np.argsort([-(b[0] + b[2]) / 2 for b in boxes])     # right-to-left

    overlay, seq, panels = img.copy(), [], []
    for rank, i in enumerate(order):
        x1, y1, x2, y2 = boxes[i]
        crop = gray[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        shown = crop
        if den is not None:
            shown = den.clean(crop)
            top = tclassify(shown, ids, bank, k=5)       # cleaned -> template domain
        else:
            top = (rec.classify(crop, k=5) if rec else tclassify(crop, ids, bank, k=5))
        pid, sc = top[0]
        seq.append({"rank": rank + 1, "sign": pid, "rec_conf": round(float(sc), 2),
                    "det_conf": round(float(confs[i]), 2)})
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(overlay, f"{rank+1}:{pid}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
        panels.append((shown, top, float(confs[i])))
    return {"image": Path(path).name, "n": len(seq), "seq": seq,
            "panels": panels, "overlay": overlay}


def b64(img, size=96):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode() if ok else ""


def write_review(rows, path, sample=0, seed=42):
    tpl_cache = {}
    def tpl(sid):
        if sid not in tpl_cache:
            raw = cv2.imread(str(TPL_DIR / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
            tpl_cache[sid] = b64(255 - raw) if raw is not None else ""
        return tpl_cache[sid]
    # flatten to one entry per detection, then optionally sample
    flat = [(r["image"], c, t, d) for r in rows for (c, t, d) in r["panels"]]
    total_all = len(flat)
    if sample and sample < total_all:
        import random
        flat = random.Random(seed).sample(flat, sample)
        moe = 100 * 1.96 * (0.25 / max(1, sample)) ** 0.5
        note = (f"Random sample of <b>{sample}</b> of {total_all} detections "
                f"(±{moe:.0f}pp margin of error at 95% confidence).")
    else:
        note = f"All <b>{total_all}</b> detections."
    h = ["""<!doctype html><meta charset=utf-8><title>Strip sign review</title><style>
body{font:14px system-ui;margin:0;background:#f6f5f2;color:#1d2127}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 20px;z-index:9}
h1{font-size:17px;margin:0 0 4px}.muted{color:#666;font-size:12.5px}
button{font:600 13px system-ui;padding:8px 14px;border-radius:7px;border:1px solid #274A82;background:#274A82;color:#fff;cursor:pointer}
.row{display:flex;align-items:center;gap:14px;background:#fff;border:1px solid #e3e0d8;border-radius:10px;padding:10px 14px;margin:10px 20px}
.crop img{width:96px;height:96px;border:2px solid #274A82;border-radius:6px;display:block}
.cands{display:flex;gap:10px;flex-wrap:wrap}.cand{text-align:center;font-size:11px}
.cand img{width:72px;height:72px;border:1px solid #ccc;border-radius:5px;display:block;background:#fff}
.cand.sel img{border:3px solid #2e7d32}
.meta{min-width:170px;font-size:12px;color:#555}
.none{font-size:12px;padding-left:6px;border-left:1px solid #e0ddd5}
</style><header><h1>Strip sign review — pick the matching template</h1>
<div class=muted>NOTE Blue = real crop. Then the top-5 guesses. <span id=done>0</span> of TOTAL judged.
<button onclick=dl()>Download CSV</button></div></header>"""]
    k = 0; meta = []
    for _img, crop, top, dconf in flat:
        if True:
            cands = "".join(
                f"<div class=cand id=c{k}_{j}><label><img src='data:image/png;base64,{tpl(s)}'>"
                f"<input type=radio name=q{k} value='{s}' onclick=pick({k},{j},'{s}')> {s}<br>{round(c,2)}</label></div>"
                for j, (s, c) in enumerate(top))
            h.append(f"<div class=row><div class=meta>#{k+1} · {_img[:26]}<br>det {round(dconf,2)}</div>"
                     f"<div class=crop><img src='data:image/png;base64,{b64(crop)}'></div>"
                     f"<div class=cands>{cands}</div>"
                     f"<div class=none><label><input type=radio name=q{k} value=NONE onclick=pick({k},-1,'NONE')> none of these</label><br>"
                     f"<label><input type=radio name=q{k} value=NOTSIGN onclick=pick({k},-1,'NOTSIGN')> not a sign</label></div></div>")
            meta.append({"image": _img, "cands": [s for s, _ in top]}); k += 1
    h.append("<script>const META=" + repr(meta).replace("'", '"') + ";const A={};")
    h.append("""
function pick(k,j,v){A[k]=v;for(let x=0;x<5;x++){const e=document.getElementById('c'+k+'_'+x);if(e)e.className='cand';}
 if(j>=0){const e=document.getElementById('c'+k+'_'+j);if(e)e.className='cand sel';}
 document.getElementById('done').textContent=Object.keys(A).length;}
function dl(){let s='idx,image,top1,chosen,verdict\\n';
 META.forEach((m,k)=>{const c=A[k]===undefined?'':A[k];let v='';
  if(c==='NOTSIGN')v='not_a_sign';else if(c==='NONE')v='wrong';
  else if(c!==''){v=(String(m.cands[0])===c)?'top1_correct':'top5_correct';}
  s+=k+',"'+m.image+'",'+m.cands[0]+','+c+','+v+'\\n';});
 const b=new Blob([s],{type:'text/csv'});const a=document.createElement('a');
 a.href=URL.createObjectURL(b);a.download='strip_review.csv';a.click();}
</script>""")
    Path(path).write_text("".join(h).replace("TOTAL", str(k)).replace("NOTE", note), encoding="utf-8")
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--sample", type=int, default=0,
                    help="judge only N randomly sampled detections (60-80 is plenty)")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--match", choices=["cnn", "template"], default="cnn")
    ap.add_argument("--binarize", action="store_true",
                    help="Otsu-binarize the WHOLE strip, then cut crops from it")
    ap.add_argument("--clean", action="store_true",
                    help="run crops through the relief->clean converter first")
    ap.add_argument("--weights", default=str(WEIGHTS))
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed. Run: pip install ultralytics")

    paths = STRIPS if args.all else [args.image]
    if not paths or paths == [None]:
        raise SystemExit("give an image path or --all")
    print(f"{len(paths)} strip(s) | conf={args.conf} | match={args.match}")

    model = YOLO(args.weights)
    ids, bank = load_bank()
    rec = _rec.try_load() if args.match == "cnn" else None
    den = _den.try_load() if args.clean else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, counts, allids, rconf, dconf = [], [], [], [], []
    for p in paths:
        r = read_one(p, model, ids, bank, rec, args.conf, den, args.binarize)
        if r is None:
            continue
        cv2.imwrite(str(OUT_DIR / f"{Path(p).stem}.png"), r["overlay"])
        rows.append(r); counts.append(r["n"])
        allids += [s["sign"] for s in r["seq"]]
        rconf += [s["rec_conf"] for s in r["seq"]]
        dconf += [s["det_conf"] for s in r["seq"]]
        seq = "-".join(str(s["sign"]) for s in r["seq"])
        print(f"  {r['image'][:24]:24s} {r['n']:2d} signs   R-to-L: {seq}")

    rep = ROOT / "reports"; rep.mkdir(exist_ok=True)
    with open(rep / "strip_eval.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "n_signs", "sequence_R_to_L", "mean_det_conf", "mean_rec_conf"])
        for r in rows:
            sq = [s["sign"] for s in r["seq"]]
            w.writerow([r["image"], r["n"], "-".join(map(str, sq)),
                        round(np.mean([s["det_conf"] for s in r["seq"]]), 3) if sq else "",
                        round(np.mean([s["rec_conf"] for s in r["seq"]]), 3) if sq else ""])

    c = np.array(counts) if counts else np.array([0])
    print("\n" + "=" * 56)
    print(f"strips processed      : {len(c)}")
    print(f"strips with >=1 sign  : {(c>0).sum()}  ({(c>0).mean():.0%})")
    print(f"signs per strip       : mean {c.mean():.1f}, median {np.median(c):.0f}, max {c.max()}")
    print(f"total signs detected  : {c.sum()}   (real inscriptions average ~5)")
    if dconf: print(f"detector confidence   : mean {np.mean(dconf):.2f}")
    if rconf: print(f"recognizer confidence : mean {np.mean(rconf):.2f}")
    if allids:
        cnt = collections.Counter(allids); top = cnt.most_common(5)
        share = sum(n for _, n in top) / len(allids)
        print(f"distinct sign IDs     : {len(cnt)} of 417")
        print(f"top-5 IDs take        : {share:.0%}  ({'COLLAPSING' if share > .5 else 'healthy spread'})")
    print("=" * 56)
    print(f"CSV -> reports/strip_eval.csv | overlays -> reports/figures/strips/")
    if args.review:
        n = write_review(rows, rep / "strip_review.html", args.sample, args.sample_seed)
        print(f"review sheet -> reports/strip_review.html  ({n} detections to judge)")


if __name__ == "__main__":
    main()
