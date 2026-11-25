"""The Epigraphic Parsing Engine - one command through all four milestones.

Proposal: "A unified Python script processing an input seal image all the way through
visualization panels to output positional n-gram probability reports."

Two kinds of input:
  image     python src/engine/parse.py --image data/raw/isd/1/1text.jpg
            M1 detects and recognises signs (IM77, reading order) -> converted to Wells
  sequence  python src/engine/parse.py --seq "342 ? 59" --numbering im77
            a typed / expert transcription in READING order (right-to-left); '?' = unreadable
            numbering: wells (default) | im77 (Mahadevan) | parpola

Stages and what each reports
  M1  boxes, top-5 Mahadevan candidates per sign                      (image input only)
  ->  crosswalk IM77/Parpola -> Wells; unmapped signs become gaps, never guessed
  M2  positional n-gram report in READING order: P(sign opens a text), P(next | previous),
      P(sign closes a text), role of each sign (opener / ender / medial), known collocations,
      bits per transition. Witten-Bell bigram with START/END, Milestone 2's definitions,
      fitted on the M3 v2 TRAIN split (Wells corpus).
  M3  each sign hidden in turn: probability the 3-model ensemble gives it, and its top-3
      alternatives (calibrated)
  M4  guardrail verdict: pseudo-perplexity vs real held-out inscriptions and structural controls

Every report states where the reading came from (`source`), so an automatic reading is never
mistaken for a verified one.

Outputs (--out DIR): report.json, m2_positional_ngrams.csv, panels.png, index.html
"""
from __future__ import annotations
import sys, json, argparse, collections, html, base64, warnings
from pathlib import Path
import numpy as np, pandas as pd, cv2

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
for p in ("src/m1_vision", "src/m3_masking", "src/m4_guardrails", "src/m2_markov", "src"):
    sys.path.insert(0, str(ROOT / p))
from scipy.stats import fisher_exact  # noqa: E402
from guardrail import scorer  # noqa: E402
from predict import SignCompleter  # noqa: E402
from common.signs import im77_to_wells, wells_to_im77, parpola_to_wells, _rows  # noqa: E402
import replicate_wells as M2  # noqa: E402

TPL = ROOT / "data/raw/IM_417_150_PNG"
PROC = ROOT / "data/processed/m4"


def w2p(w):
    return "/".join(r["parpola"] for r in _rows() if w in r["wells"]) or "-"


class Engine:
    def __init__(self, with_vision=True):
        self.S = scorer()                                  # M3 ensemble + M4 guardrail
        self.sc = SignCompleter()                          # M3 completion with conversion
        self.vocab = self.S.vocab
        self._fit_m2()
        self._fit_m4_reference()
        self.m1 = None
        if with_vision:
            from ultralytics import YOLO
            from read_strip import WEIGHTS
            from template_match import load_bank
            import recognizer
            self.m1 = (YOLO(str(WEIGHTS)), *load_bank(), recognizer.try_load())

    # ---------- M2: Witten-Bell bigram in reading order, Milestone 2 definitions ----------
    def _fit_m2(self):
        from common.order import segments
        segs = [seg for s in self.S.split["train"] for seg in segments(s)]
        self.m2_vocab = sorted({x for s, *_ in segs for x in s})
        self.P, self.ri, self.ci, self.pu = M2.wb_model(segs, self.m2_vocab)
        role = collections.defaultdict(lambda: [0, 0, 0])
        for s, ts, te in segs:
            for j, t in enumerate(s):
                role[t][0 if (j == 0 and ts) else 2 if (j == len(s) - 1 and te) else 1] += 1
        self.role = {}
        for t, (a, m, b) in role.items():
            n = a + m + b
            self.role[t] = "opener" if n >= 10 and a / n > 0.5 else "ender" if n >= 10 and b / n > 0.5 else \
                           "medial" if n >= 10 and (a + b) == 0 else "mixed / rare"
        pairs = collections.Counter((a, b) for s, *_ in segs for a, b in zip(s[:-1], s[1:]))
        N = sum(pairs.values()); left, right = collections.Counter(), collections.Counter()
        for (a, b), c in pairs.items(): left[a] += c; right[b] += c
        cand = [(a, b, c) for (a, b), c in pairs.items() if c >= 3]
        ps = np.array([fisher_exact([[c, left[a] - c], [right[b] - c, N - left[a] - right[b] + c]], alternative="greater")[1] for a, b, c in cand])
        order = np.argsort(ps); q = np.empty(len(ps)); prev = 1.0
        for rank, i in reversed(list(enumerate(order, 1))):
            prev = min(prev, ps[i] * len(ps) / rank); q[i] = prev
        self.colloc = {(a, b) for (a, b, c), qq in zip(cand, q) if qq < 0.05}

    def m2_report(self, wr):
        """wr = Wells signs in reading order (None = gap). Returns rows and total bits."""
        rows, bits = [], 0.0
        seq = ["<S>"] + wr + ["</S>"]
        for i in range(1, len(seq)):
            a, b = seq[i - 1], seq[i]
            if a is None or b is None or (a not in self.ri) or (b not in self.ci):
                p = None
            else:
                p = float(self.P[self.ri[a], self.ci[b]])
                bits += -np.log2(p)
            sa, sb = (a or "?"), (b or "?")
            label = f"START -> {sb}" if a == "<S>" else (f"{sa} -> END" if b == "</S>" else f"{sa} -> {sb}")
            rows.append(dict(transition=label, probability=p, bits=None if p is None else float(-np.log2(p)),
                             known_collocation=(a, b) in self.colloc))
        return rows, bits

    # ---------- M4 reference distributions ----------
    def _fit_m4_reference(self):
        sets = json.load(open(PROC / "controls_test.json"))["sets"]
        self.ref = {k: self.S.pppl(sets[k])["tf"] for k in ("real", "shuffled", "freq_random", "uniform_random")}

    def m4_verdict(self, stored):
        if self.S.n_known(stored) < 2:
            return dict(pseudo_perplexity=None, verdict="not enough readable signs (need >= 2)")
        p = float(self.S.pppl([stored])["tf"][0])
        real = self.ref["real"][np.isfinite(self.ref["real"])]
        pct = float((real <= p).mean())
        fr = float(np.nanmedian(self.ref["freq_random"]))
        v = "consistent with real inscriptions" if p <= self.S.threshold else \
            "flagged: unusual" if p <= fr else "flagged: noise-like"
        return dict(pseudo_perplexity=p, threshold=self.S.threshold, percentile_among_real=pct,
                    median_real=float(np.nanmedian(real)), median_shuffled=float(np.nanmedian(self.ref["shuffled"])),
                    median_frequency_random=fr, verdict=v)

    # ---------- M3 ----------
    def m3_report(self, stored):
        """For each sign (stored order): probability the calibrated ensemble gives it when it is
        hidden (leave-one-out), plus the model's top-3 alternatives for that slot."""
        toks = stored[1:-1]
        known = [j for j, t in enumerate(toks) if t not in ("000", "UNMAPPED") and t in self.vocab.stoi]
        loo = self.S.loo_probs([stored])                 # one row per known sign, in position order
        prob = {j: r[1] for j, r in zip(known, loo)}
        out = []
        for j, t in enumerate(toks):
            if j not in prob:
                out.append(dict(position_stored=j, sign=t, prob=None, alternatives=[])); continue
            g = self.sc.complete(list(toks), top_k=3, mask=[j],
                                 edges="complete" if stored[0] == "<S>" else "broken")["gaps"][0]
            out.append(dict(position_stored=j, sign=t, prob=float(prob[j]), rating=g["rating"],
                            alternatives=[(c["sign"], round(c["prob"], 3)) for c in g["candidates"]]))
        return out

    # ---------- inputs ----------
    def from_image(self, path, conf=0.15):
        from read_strip import read_one
        yolo, ids, bank, rec = self.m1
        r = read_one(path, yolo, ids, bank, rec, conf)
        signs = [dict(top5=[(int(i), float(s)) for i, s in top], det_conf=round(d, 3), crop=c) for (c, top, d) in r["panels"]]
        return signs, r["overlay"]

    def to_wells(self, ids, numbering):
        wr, notes = [], []
        for x in ids:
            if x is None or str(x) in ("?", "000"):
                wr.append(None); continue
            if numbering == "wells":
                w = str(x).zfill(3); wr.append(w if w in self.vocab.stoi else w); continue
            c = im77_to_wells(int(x)) if numbering == "im77" else parpola_to_wells(str(x).upper())
            if not c:
                wr.append(None); notes.append(f"{numbering} {x}: no Wells equivalent -> treated as unreadable")
                continue
            best = max(c, key=lambda w: self.S.vocab.stoi.get(w, 10 ** 6) * -1)
            if len(c) > 1: notes.append(f"{numbering} {x} -> Wells {best} (ambiguous: {'/'.join(c)})")
            wr.append(best)
        return wr, notes

    # ---------- run ----------
    def run(self, image=None, seq=None, numbering="wells", truth_wells=None, out=None, source=None):
        m1, overlay = None, None
        if image is not None:
            signs, overlay = self.from_image(image)
            m1 = signs; ids = [s["top5"][0][0] for s in signs]; numbering = "im77"
            source = source or "M1 automatic reading (not verified)"
        else:
            ids = seq.split() if isinstance(seq, str) else list(seq)
            source = source or "typed / transcribed sequence"
        wr, notes = self.to_wells(ids, numbering)
        stored = ["<S>"] + [("000" if w is None else w) for w in wr][::-1] + ["<E>"]
        m2_rows, m2_bits = self.m2_report(wr)
        m3 = self.m3_report(stored)
        m4 = self.m4_verdict(stored)
        rep = dict(source=source, input=str(image or seq), numbering_in=numbering,
                   reading_order_wells=[w or "?" for w in wr],
                   reading_order_im77=["/".join(map(str, wells_to_im77(w))) if w else "?" for w in wr],
                   reading_order_parpola=[w2p(w) if w else "?" for w in wr],
                   conversion_notes=notes,
                   m1=None if m1 is None else [dict(top5=s["top5"], det_conf=s["det_conf"]) for s in m1],
                   m2_positional_ngrams=m2_rows, m2_total_bits=m2_bits,
                   m2_roles=[self.role.get(w, "unseen") if w else "-" for w in wr],
                   m3_per_sign=m3, m4=m4)
        if truth_wells is not None:
            rep["truth_wells_reading"] = truth_wells
        if out:
            self.save(rep, out, overlay, m1)
        return rep

    # ---------- outputs ----------
    def save(self, rep, out, overlay, m1):
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(out / "report.json", "w"), indent=1, default=str)
        pd.DataFrame(rep["m2_positional_ngrams"]).to_csv(out / "m2_positional_ngrams.csv", index=False)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n_top = 2 if overlay is not None else 0
        fig = plt.figure(figsize=(13, 3.2 * (n_top + 1.2)))
        gs = fig.add_gridspec(n_top + 1, 3)
        if overlay is not None:
            ax = fig.add_subplot(gs[0, :]); ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); ax.axis("off")
            ax.set_title(f"M1: detected signs, numbered in reading order ({rep['source']})", fontsize=10)
            k = len(m1); sub = gs[1, :].subgridspec(2, max(k, 1), hspace=0.45)
            for i, s in enumerate(m1):
                a1 = fig.add_subplot(sub[0, i]); a1.imshow(s["crop"], cmap="gray"); a1.axis("off")
                a1.set_title(f"#{i+1}", fontsize=8)
                tid = s["top5"][0][0]; t = cv2.imread(str(TPL / f"{tid}.png"), 0)
                a2 = fig.add_subplot(sub[1, i]); a2.axis("off")
                if t is not None: a2.imshow(255 - t, cmap="gray")
                a2.set_title(f"IM77 {tid}", fontsize=8)
        row = gs[n_top, :].subgridspec(1, 3, wspace=0.35)
        a = fig.add_subplot(row[0])
        rr = rep["m2_positional_ngrams"]
        a.bar(range(len(rr)), [r["bits"] if r["bits"] is not None else 0 for r in rr],
              color=["#2a9d8f" if r["known_collocation"] else "#8ab17d" if r["bits"] is not None else "#cccccc" for r in rr])
        a.set_xticks(range(len(rr))); a.set_xticklabels([r["transition"] for r in rr], rotation=60, fontsize=7, ha="right")
        a.set_ylabel("bits (lower = more expected)"); a.set_title("M2: positional n-gram surprise\n(dark = known collocation)", fontsize=9)
        b = fig.add_subplot(row[1])
        m3 = rep["m3_per_sign"][::-1]                                   # reading order
        b.bar(range(len(m3)), [-np.log2(x["prob"]) if x["prob"] else 0 for x in m3], color="#264653")
        b.set_xticks(range(len(m3))); b.set_xticklabels([x["sign"] for x in m3], fontsize=8)
        b.set_ylabel("bits"); b.set_title("M3: surprise of each sign given the rest\n(reading order)", fontsize=9)
        c = fig.add_subplot(row[2])
        ref = [v[np.isfinite(v)] for v in self.ref.values()]
        c.boxplot(ref, showfliers=False, widths=.5); c.set_yscale("log")
        c.set_xticks(range(1, 5)); c.set_xticklabels(["real", "shuffled", "freq-rand", "uniform"], fontsize=8)
        p = rep["m4"]["pseudo_perplexity"]
        if p: c.axhline(p, color="#e76f51", lw=2); c.text(4.45, p, " this", color="#e76f51", fontsize=8, va="center")
        c.axhline(self.S.threshold, ls="--", c="k", lw=.8)
        c.set_title(f"M4: {rep['m4']['verdict']}", fontsize=9)
        plt.tight_layout(); plt.savefig(out / "panels.png", dpi=120); plt.close(fig)
        img = base64.b64encode(open(out / "panels.png", "rb").read()).decode()
        def fmt(v, f):
            return "" if v is None else format(v, f)
        trs = "".join(f"<tr><td>{html.escape(r['transition'])}</td><td>{fmt(r['probability'], '.4f')}</td>"
                      f"<td>{fmt(r['bits'], '.2f')}</td><td>{'yes' if r['known_collocation'] else ''}</td></tr>"
                      for r in rep["m2_positional_ngrams"])
        page = f"""<!doctype html><meta charset=utf-8><title>Engine report</title>
<style>body{{font:14px system-ui;margin:24px;color:#1d2127}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:3px 8px}}
.v{{font-size:17px;font-weight:600}}</style>
<h2>Epigraphic Parsing Engine</h2><p><b>Source:</b> {html.escape(rep['source'])} &nbsp; <b>Input:</b> {html.escape(rep['input'])}</p>
<p>Reading order (right-to-left) &mdash; Wells: <b>{' '.join(rep['reading_order_wells'])}</b> &nbsp;|&nbsp; Mahadevan: {' '.join(rep['reading_order_im77'])} &nbsp;|&nbsp; Parpola: {' '.join(rep['reading_order_parpola'])}</p>
<p>{'<br>'.join(html.escape(n) for n in rep['conversion_notes'])}</p>
<p class=v>M4 verdict: {html.escape(rep['m4']['verdict'])}</p>
<img src="data:image/png;base64,{img}" style="max-width:100%">
<h3>M2 positional n-gram probability report</h3><table><tr><th>transition (reading order)</th><th>P (Witten-Bell)</th><th>bits</th><th>known collocation</th></tr>{trs}</table>
<p>Total: {rep['m2_total_bits']:.1f} bits. Sign roles: {', '.join(f'{w}={r}' for w, r in zip(rep['reading_order_wells'], rep['m2_roles']))}</p>"""
        (out / "index.html").write_text(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image"); ap.add_argument("--seq"); ap.add_argument("--numbering", default="wells", choices=["wells", "im77", "parpola"])
    ap.add_argument("--out", default=str(ROOT / "reports/engine/last"))
    a = ap.parse_args()
    if not (a.image or a.seq): raise SystemExit("give --image PATH or --seq 'IDS in reading order'")
    eng = Engine(with_vision=bool(a.image))
    rep = eng.run(image=a.image, seq=a.seq, numbering=a.numbering, out=a.out)
    print(f"source : {rep['source']}")
    print(f"reading order (Wells): {' '.join(rep['reading_order_wells'])}   (IM77: {' '.join(rep['reading_order_im77'])})")
    for n in rep["conversion_notes"]: print("  note:", n)
    print(f"M2 total {rep['m2_total_bits']:.1f} bits; roles {rep['m2_roles']}")
    print(f"M4: {rep['m4']['verdict']} (pseudo-perplexity {rep['m4']['pseudo_perplexity']})")
    print(f"-> {a.out}/index.html")


if __name__ == "__main__":
    main()
