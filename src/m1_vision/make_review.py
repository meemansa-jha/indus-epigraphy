"""Build a hand-verification sheet for every detection on the real seals.

Produces ONE self-contained HTML file. For each detected sign it shows the real
crop next to the recognizer's TOP-5 candidate templates. You click which
candidate is correct (or "none"/"not a sign"), then export a CSV.

That CSV gives what no automatic metric can: manually verified detection
precision and recognition top-1 / top-5 accuracy on REAL seals.

Usage:
  python src/m1_vision/make_review.py --conf 0.15
  # then open reports/review.html in a browser
"""
from __future__ import annotations
import argparse, base64, io
import numpy as np
import cv2

from config import RAW_DIR, ROOT
from template_match import load_bank, classify as tclassify
from read_seal import find_seal_regions
import recognizer as _rec

REAL_DIR = RAW_DIR / "real_seals"
TPL_DIR = RAW_DIR / "IM_417_150_PNG"
OUT = ROOT / "reports" / "review.html"
MAXW = 1280
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def b64(img, size=96):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode() if ok else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--match", choices=["cnn", "template"], default="cnn")
    ap.add_argument("--weights", default=str(ROOT / "runs" / "sign_detector" / "weights" / "best.pt"))
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    ids, bank = load_bank()
    rec = _rec.try_load() if args.match == "cnn" else None

    tpl_cache = {}
    def tpl_b64(sid):
        if sid not in tpl_cache:
            raw = cv2.imread(str(TPL_DIR / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
            tpl_cache[sid] = b64(255 - raw) if raw is not None else ""
        return tpl_cache[sid]

    rows = []
    files = sorted(p for p in REAL_DIR.iterdir() if p.suffix.lower() in EXTS)
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[1] > MAXW:
            s = MAXW / img.shape[1]
            img = cv2.resize(img, (MAXW, int(img.shape[0] * s)))
        H, W = img.shape[:2]
        regions = find_seal_regions(img) or [(0, 0, W, H)]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for (rx, ry, rw, rh) in regions:
            x0, y0 = max(0, rx - 6), max(0, ry - 6)
            x1r, y1r = min(W, rx + rw + 6), min(H, ry + rh + 6)
            res = model(img[y0:y1r, x0:x1r], conf=args.conf, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            bxs = [tuple(map(int, b)) for b in res.boxes.xyxy.cpu().numpy()]
            cfs = res.boxes.conf.cpu().numpy().tolist()
            bxs = [(a + x0, b + y0, c + x0, d + y0) for a, b, c, d in bxs]
            for i in np.argsort([-(b[0] + b[2]) / 2 for b in bxs]):
                bx1, by1, bx2, by2 = bxs[i]
                crop = gray[max(0, by1):by2, max(0, bx1):bx2]
                if crop.size == 0:
                    continue
                top = (rec.classify(crop, k=5) if rec else tclassify(crop, ids, bank, k=5))
                rows.append({"image": p.name, "det_conf": round(float(cfs[i]), 2),
                             "crop": b64(crop), "cands": [(int(s), round(float(c), 2)) for s, c in top]})
        print(f"  {p.name[:50]:50s} running total {len(rows)}")

    # ---- build the HTML ----
    h = ["""<!doctype html><meta charset=utf-8><title>Sign review</title><style>
body{font:14px system-ui;margin:0;background:#f6f5f2;color:#1d2127}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 20px;z-index:9}
h1{font-size:17px;margin:0 0 4px}.muted{color:#666;font-size:12.5px}
button{font:600 13px system-ui;padding:8px 14px;border-radius:7px;border:1px solid #274A82;
background:#274A82;color:#fff;cursor:pointer}
.row{display:flex;align-items:center;gap:14px;background:#fff;border:1px solid #e3e0d8;
border-radius:10px;padding:10px 14px;margin:10px 20px}
.crop img{width:96px;height:96px;border:2px solid #274A82;border-radius:6px;display:block}
.cands{display:flex;gap:10px;flex-wrap:wrap}
.cand{text-align:center;font-size:11px}
.cand img{width:72px;height:72px;border:1px solid #ccc;border-radius:5px;display:block;background:#fff}
.cand.sel img{border:3px solid #2e7d32}
label{cursor:pointer}
.meta{min-width:190px;font-size:12px;color:#555;word-break:break-all}
.none{font-size:12px;padding-left:6px;border-left:1px solid #e0ddd5}
#done{font-weight:600}
</style><header><h1>Indus sign review — click the correct template for each crop</h1>
<div class=muted>Left (blue border) = real crop from the seal. Then the recognizer's top-5 guesses,
best first. Pick the matching one, or <b>none</b> (wrong ID) / <b>not a sign</b> (bad detection).
<span id=done>0</span> of TOTAL judged. <button onclick=dl()>Download CSV</button></div></header>"""]
    for k, r in enumerate(rows):
        cands = "".join(
            f"<div class=cand id=c{k}_{j}><label><img src='data:image/png;base64,{tpl_b64(sid)}'>"
            f"<input type=radio name=q{k} value='{sid}' onclick=pick({k},{j},'{sid}')> {sid}<br>{cf}</label></div>"
            for j, (sid, cf) in enumerate(r["cands"]))
        h.append(
            f"<div class=row><div class=meta>#{k+1} · {r['image'][:34]}<br>det {r['det_conf']}</div>"
            f"<div class=crop><img src='data:image/png;base64,{r['crop']}'></div>"
            f"<div class=cands>{cands}</div>"
            f"<div class=none><label><input type=radio name=q{k} value=NONE onclick=pick({k},-1,'NONE')> none of these</label><br>"
            f"<label><input type=radio name=q{k} value=NOTSIGN onclick=pick({k},-1,'NOTSIGN')> not a sign</label></div></div>")

    meta = [{"image": r["image"], "det_conf": r["det_conf"],
             "cands": [s for s, _ in r["cands"]]} for r in rows]
    h.append("<script>const META=" + repr(meta).replace("'", '"') + ";const A={};")
    h.append("""
function pick(k,j,v){A[k]=v;
 for(let x=0;x<5;x++){const e=document.getElementById('c'+k+'_'+x); if(e)e.className='cand';}
 if(j>=0){const e=document.getElementById('c'+k+'_'+j); if(e)e.className='cand sel';}
 document.getElementById('done').textContent=Object.keys(A).length;}
function dl(){let s='idx,image,det_conf,top1,chosen,verdict\\n';
 META.forEach((m,k)=>{const c=A[k]===undefined?'':A[k];
  let v='';if(c==='NOTSIGN')v='not_a_sign';else if(c==='NONE')v='wrong';
  else if(c!==''){v=(String(m.cands[0])===c)?'top1_correct':'top5_correct';}
  s+=k+',"'+m.image+'",'+m.det_conf+','+m.cands[0]+','+c+','+v+'\\n';});
 const b=new Blob([s],{type:'text/csv'});const a=document.createElement('a');
 a.href=URL.createObjectURL(b);a.download='review_results.csv';a.click();}
</script>""")
    html = "".join(h).replace("TOTAL", str(len(rows)))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"\n{len(rows)} detections -> {OUT}")
    print("Open reports/review.html in a browser, judge each, then Download CSV.")


if __name__ == "__main__":
    main()
