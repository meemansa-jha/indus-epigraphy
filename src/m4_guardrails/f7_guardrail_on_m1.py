"""M4 / F7 - the guardrail applied to Milestone 1's real output (integration of M1 -> M3 -> M4).

1. Calibrate the guardrail: flag rate and AUC on held-out REAL test inscriptions vs
   each structural control (threshold set on the validation split, see guardrail.py).
2. Convert M1's readings of the 77 strips (IM77, reading order) to Wells / stored order
   through the crosswalk and score them.
3. Context re-ranking: let M3 choose, for each sign, among M1's top-5 visual candidates
   (score = log P_visual + log P_M3(candidate | rest of the strip), 3 coordinate sweeps).
   This shows what a language model does to noisy OCR output: it makes it LOOK real.
   There is no per-strip ground truth, so we cannot tell whether it is more CORRECT -
   that gap is the point.

  python src/m4_guardrails/f7_guardrail_on_m1.py
Outputs: reports/m4/f7_guardrail_on_m1.json, f7_guardrail_on_m1.png
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from guardrail import scorer, m1_to_stored  # noqa: E402
from common.signs import im77_to_wells  # noqa: E402
from data import MASK_ID  # noqa: E402
from model import pad_batch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

PROC = ROOT / "data" / "processed" / "m4"
REP = ROOT / "reports" / "m4"


def ctx_logprob(S, seq, pos, cands):
    """log P_M3(candidate at pos | rest), ensemble-averaged, for each candidate Wells id."""
    x = S.vocab.encode(seq); x[pos] = MASK_ID
    b = pad_batch([x])
    with torch.no_grad():
        p = np.mean([F.softmax(m(b), -1)[0, pos].numpy() for m in S.models], 0)
    return [np.log(max(p[S.vocab.stoi[c]], 1e-12)) if c in S.vocab.stoi else np.log(1e-12) for c in cands]


def rerank(S, strip):
    """Choose one Wells sign per position from M1's top-5 using visual score + M3 context."""
    opts = []
    for sg in strip["signs"][::-1]:                         # reading -> stored order
        o = []
        for im, sc in sg["top5"]:
            ws = im77_to_wells(im)
            if ws:
                w = max(ws, key=lambda z: z in S.vocab.stoi)
                o.append((w, np.log(max(sc, 1e-6))))
        opts.append(o or [("000", 0.0)])
    seq = ["<S>"] + [o[0][0] for o in opts] + ["<E>"]
    for _ in range(3):
        for i, o in enumerate(opts):
            if len(o) < 2:
                continue
            lc = ctx_logprob(S, seq, i + 1, [w for w, _ in o])
            seq[i + 1] = o[int(np.argmax([v + c for (_, v), c in zip(o, lc)]))][0]
    return seq


def main():
    S = scorer()
    sets = json.load(open(PROC / "controls_test.json"))["sets"]
    real_p, _ = S.verdict(sets["real"])
    calib = {}
    for k, v in sets.items():
        p, flag = S.verdict(v)
        ok = ~np.isnan(p)
        auc = None if k == "real" else float(roc_auc_score(
            np.r_[np.zeros(np.isfinite(real_p).sum()), np.ones(ok.sum())],
            np.r_[real_p[np.isfinite(real_p)], p[ok]]))
        fr = float(np.nanmean(flag)); nn_ = int(ok.sum()); z = 1.96            # Wilson 95% CI for the flag rate
        cen = (fr + z * z / (2 * nn_)) / (1 + z * z / nn_); half = z * np.sqrt(fr * (1 - fr) / nn_ + z * z / (4 * nn_ * nn_)) / (1 + z * z / nn_)
        auc_ci = None
        if k != "real":
            rng = np.random.default_rng(0); rp = real_p[np.isfinite(real_p)]; fp = p[ok]; bs = []
            for _ in range(1000):
                a_, b_ = rng.choice(rp, len(rp)), rng.choice(fp, len(fp))
                bs.append(roc_auc_score(np.r_[np.zeros(len(a_)), np.ones(len(b_))], np.r_[a_, b_]))
            auc_ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        calib[k] = dict(n=nn_, flag_rate=fr, flag_rate_ci95=[float(cen - half), float(cen + half)],
                        median_pppl=float(np.nanmedian(p)), auc_vs_real=auc, auc_ci95=auc_ci)

    strips = json.load(open(PROC / "m1_outputs.json"))
    top1, rer, unm, amb = [], [], 0, 0
    for st in strips:
        seq, u, a = m1_to_stored([sg["top5"][0][0] for sg in st["signs"]])
        top1.append(seq); unm += u; amb += a
        rer.append(rerank(S, st))
    keep = [i for i, s in enumerate(top1) if S.n_known(s) >= 2]
    t1 = [top1[i] for i in keep]; rr = [rer[i] for i in keep]
    p1, f1 = S.verdict(t1); p2, f2 = S.verdict(rr)
    n_signs = sum(len(st["signs"]) for st in strips)
    changed = sum(a != b for i in keep for a, b in zip(top1[i], rer[i]))
    res = dict(
        threshold_pppl=S.threshold,
        calibration_on_test=calib,
        m1=dict(strips=len(strips), signs=n_signs, unmapped_signs=unm, ambiguous_mappings=amb,
                strips_scored=len(keep)),
        m1_top1=dict(median_pppl=float(np.nanmedian(p1)), flag_rate=float(np.nanmean(f1)), n=int(np.isfinite(p1).sum())),
        m1_reranked_by_m3=dict(median_pppl=float(np.nanmedian(p2)), flag_rate=float(np.nanmean(f2)),
                               signs_changed=int(changed)),
        examples=[dict(image=strips[i]["image"], m1_top1="-".join(top1[i][1:-1]),
                       reranked="-".join(rer[i][1:-1]), pppl_top1=float(a), pppl_reranked=float(b))
                  for i, a, b in list(zip(keep, p1, p2))[:8]])
    REP.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(REP / "f7_guardrail_on_m1.json", "w"), indent=1)

    print(f"guardrail threshold (95th pct of real validation pppl): {S.threshold:.0f}")
    for k, c in calib.items():
        auc_txt = "-" if c["auc_vs_real"] is None else f"{c['auc_vs_real']:.3f} [{c['auc_ci95'][0]:.3f}-{c['auc_ci95'][1]:.3f}]"
        print(f"  {k:15s} flagged {c['flag_rate']:5.1%} [{c['flag_rate_ci95'][0]:.1%}-{c['flag_rate_ci95'][1]:.1%}]  "
              f"median pppl {c['median_pppl']:7.1f}  AUC vs real {auc_txt}")
    print(f"M1: {len(strips)} strips, {n_signs} signs, {unm} unmapped, {amb} ambiguous; {len(keep)} strips scored")
    print(f"  M1 top-1 readings      : median pppl {np.nanmedian(p1):7.1f}  flagged {np.nanmean(f1):.0%}")
    print(f"  after M3 re-ranking    : median pppl {np.nanmedian(p2):7.1f}  flagged {np.nanmean(f2):.0%}  ({changed} signs changed)")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = [real_p[np.isfinite(real_p)], S.verdict(sets["shuffled"])[0], S.verdict(sets["uniform_random"])[0], p1, p2]
    labs = ["Real (held-out)", "Shuffled", "Uniform-random", "M1 output\n(top-1)", "M1 output\nre-ranked by M3"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bp = ax.boxplot([d[np.isfinite(d)] for d in data], patch_artist=True, showfliers=False, widths=.6)
    for b, c in zip(bp["boxes"], ["#2a9d8f", "#bbbbbb", "#bbbbbb", "#e76f51", "#f4a261"]):
        b.set_facecolor(c)
    ax.axhline(S.threshold, ls="--", c="k", lw=1); ax.text(5.42, S.threshold, "guardrail\nthreshold", va="center", fontsize=8)
    ax.set_yscale("log"); ax.set_xticks(range(1, 6)); ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("pseudo-perplexity (M3 ensemble, log scale)")
    ax.set_title("Milestone 1's readings of real strips, judged by the Milestone 4 guardrail", fontsize=11)
    plt.tight_layout(); plt.savefig(REP / "f7_guardrail_on_m1.png", dpi=140)
    print("->", REP / "f7_guardrail_on_m1.png")


if __name__ == "__main__":
    main()
