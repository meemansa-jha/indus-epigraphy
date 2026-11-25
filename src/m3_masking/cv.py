"""Milestone 3 - 5-fold cross-validation (the proposal asks for "cross-validation accuracies").

The single 80/10/10 split in run_all.py gives one estimate. Here every unique sign sequence
(v2 deduplication, so no sequence can appear in two folds) is used as test data exactly once:
  * 5 folds, seed 0;  in each fold: 4/5 for training (of which 10% held out for early stopping
    and temperature), 1/5 for testing
  * vocabulary rebuilt per fold from that fold's training data (signs seen >= 2 times)
  * models per fold: unigram, Witten-Bell bigram, Transformer (selected config from
    artifacts/search.json, ONE seed, temperature-calibrated on the fold's validation part)
  * protocol = M3's leave-one-out: every readable test sign hidden once
Reports top-1 / top-5 / perplexity per fold, mean +- sd over folds, and a paired comparison
Transformer vs bigram (McNemar on pooled predictions, and how many folds each model wins).

  python src/m3_masking/cv.py            # resumable: each finished fold is saved
Output: artifacts/cv_folds.json, ../../reports/m3/cv.json
"""
import os, sys, json, random, time
import numpy as np
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from data import load_unique_sequences, Vocab, encode_split, make_loo, NSPEC  # noqa: E402
from model import train_model, nn_logprobs  # noqa: E402
from baselines import UnigramModel, WittenBellBigram  # noqa: E402
from evaluate import score_examples, summarize, fit_temperature, apply_temperature  # noqa: E402
import collections, torch  # noqa: E402

K, SEED = 5, 0
FOLDS = "artifacts/cv_folds.json"
OUT = os.path.join(HERE, "..", "..", "reports", "m3", "cv.json")
CFG = min(json.load(open("artifacts/search.json")), key=lambda r: r["val_nll"])["cfg"]


def folds():
    _, seqs = load_unique_sequences(os.path.join(HERE, "..", "..", "data", "raw", "m3_csv", "inscriptions.csv"))
    keys = [list(k) for k in seqs]
    random.Random(SEED).shuffle(keys)
    return keys, [keys[i::K] for i in range(K)]


def run_fold(k, parts):
    t0 = time.time()
    test = parts[k]
    rest = [s for j, p in enumerate(parts) if j != k for s in p]
    nv = len(rest) // 10
    val, train = rest[:nv], rest[nv:]
    cnt = collections.Counter(t for s in train for t in s[1:-1] if t != "000")
    vocab = Vocab([t for t, c in cnt.most_common() if c >= 2]); V = len(vocab)
    tr, va, te = (encode_split(vocab, x) for x in (train, val, test))
    vex, tex = make_loo(va), make_loo(te)
    model, _, hist = train_model(CFG, tr, vex, V, seed=k, epochs=60, patience=8, repeats=2)
    probs = lambda ex: [np.exp(l) for l in nn_logprobs(model, ex)]
    T = fit_temperature(probs(vex), vex)
    tf = score_examples(apply_temperature(probs(tex), T), tex)
    bg = score_examples(WittenBellBigram(tr, V).probs(tex), tex)
    un = score_examples(UnigramModel(tr, V).probs(tex), tex)
    return dict(fold=k, n_train=len(train), n_val=len(val), n_test=len(test), vocab=V - NSPEC,
                epochs=len(hist), temperature=T, seconds=round(time.time() - t0),
                unigram=summarize(un), bigram=summarize(bg), transformer=summarize(tf),
                tf_correct=tf.top1.astype(int).tolist(), bg_correct=bg.top1.astype(int).tolist())


ENS = "artifacts/cv_folds_ensemble.json"


def run_ensemble_fold(k, parts):
    """Same fold as run_fold, but the deliverable's set-up: 3 seeds averaged, then calibrated."""
    t0 = time.time()
    test = parts[k]; rest = [s for j, p in enumerate(parts) if j != k for s in p]
    nv = len(rest) // 10; val, train = rest[:nv], rest[nv:]
    cnt = collections.Counter(t for s in train for t in s[1:-1] if t != "000")
    vocab = Vocab([t for t, c in cnt.most_common() if c >= 2]); V = len(vocab)
    tr, va, te = (encode_split(vocab, x) for x in (train, val, test))
    vex, tex = make_loo(va), make_loo(te)
    models = [train_model(CFG, tr, vex, V, seed=sd, epochs=60, patience=8, repeats=2)[0] for sd in (k, 100 + k, 200 + k)]
    probs = lambda ex: [np.mean(z, 0) for z in zip(*[[np.exp(l) for l in nn_logprobs(m, ex)] for m in models])]
    T = fit_temperature(probs(vex), vex)
    ens = score_examples(apply_temperature(probs(tex), T), tex)
    bg = score_examples(WittenBellBigram(tr, V).probs(tex), tex)
    return dict(fold=k, temperature=T, seconds=round(time.time() - t0), ensemble=summarize(ens), bigram=summarize(bg),
                ens_correct=ens.top1.astype(int).tolist(), bg_correct=bg.top1.astype(int).tolist(),
                ens_nll=ens.nll.tolist(), bg_nll=bg.nll.tolist())


def report_ensemble(done):
    rows = [done[str(k)] for k in range(K)]
    e = np.concatenate([r["ens_correct"] for r in rows]); b_ = np.concatenate([r["bg_correct"] for r in rows])
    b, c = int(((e == 1) & (b_ == 0)).sum()), int(((e == 0) & (b_ == 1)).sum())
    en = np.concatenate([r["ens_nll"] for r in rows]); bn = np.concatenate([r["bg_nll"] for r in rows])
    rng = np.random.default_rng(0); idx = np.arange(len(en)); diffs = []
    for _ in range(2000):
        i = rng.choice(idx, len(idx)); diffs.append(np.exp(bn[i].mean()) - np.exp(en[i].mean()))
    res = dict(per_fold=[dict(fold=r["fold"], ensemble=r["ensemble"], bigram=r["bigram"], temperature=r["temperature"]) for r in rows],
               ensemble_top1=dict(mean=float(np.mean([r["ensemble"]["top1"] for r in rows])), sd=float(np.std([r["ensemble"]["top1"] for r in rows], ddof=1))),
               ensemble_ppl=dict(mean=float(np.mean([r["ensemble"]["ppl"] for r in rows])), sd=float(np.std([r["ensemble"]["ppl"] for r in rows], ddof=1))),
               ensemble_ece=dict(mean=float(np.mean([r["ensemble"]["ece"] for r in rows]))),
               pooled=dict(ensemble_top1=float(e.mean()), bigram_top1=float(b_.mean()), only_ensemble_right=b, only_bigram_right=c,
                           mcnemar_p=float(binomtest(b, b + c, 0.5).pvalue),
                           ppl_ensemble=float(np.exp(en.mean())), ppl_bigram=float(np.exp(bn.mean())),
                           ppl_gap_bigram_minus_ensemble=float(np.exp(bn.mean()) - np.exp(en.mean())),
                           ppl_gap_ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]),
               folds_ensemble_better_top1=sum(r["ensemble"]["top1"] > r["bigram"]["top1"] for r in rows),
               folds_ensemble_better_ppl=sum(r["ensemble"]["ppl"] < r["bigram"]["ppl"] for r in rows))
    return res


def report(done):
    rows = [done[str(k)] for k in range(K)]
    agg = {}
    for m in ("unigram", "bigram", "transformer"):
        agg[m] = {x: dict(mean=float(np.mean([r[m][x] for r in rows])), sd=float(np.std([r[m][x] for r in rows], ddof=1)),
                          per_fold=[round(r[m][x], 4) for r in rows]) for x in ("top1", "top5", "ppl", "ece")}
    tf = np.concatenate([r["tf_correct"] for r in rows]); bg = np.concatenate([r["bg_correct"] for r in rows])
    b, c = int(((tf == 1) & (bg == 0)).sum()), int(((tf == 0) & (bg == 1)).sum())
    res = dict(config=CFG, folds=K, n_sequences=sum(r["n_test"] for r in rows), n_predictions=int(len(tf)),
               metrics=agg,
               transformer_vs_bigram=dict(
                   pooled_top1=dict(transformer=float(tf.mean()), bigram=float(bg.mean()),
                                    only_transformer_right=b, only_bigram_right=c,
                                    mcnemar_p=float(binomtest(b, b + c, 0.5).pvalue)),
                   folds_transformer_better_top1=sum(r["transformer"]["top1"] > r["bigram"]["top1"] for r in rows),
                   folds_transformer_better_ppl=sum(r["transformer"]["ppl"] < r["bigram"]["ppl"] for r in rows)),
               per_fold=[{k: v for k, v in r.items() if not k.endswith("_correct")} for r in rows])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"{'':12s}" + "".join(f"  fold{k}" for k in range(K)) + "     mean ± sd")
    for m in ("unigram", "bigram", "transformer"):
        for x in ("top1", "top5", "ppl"):
            a = agg[m][x]; f = (lambda v: f"{v:7.1f}") if x == "ppl" else (lambda v: f"{100*v:6.1f}%")
            print(f"{m[:6]+' '+x:12s}" + "".join(f(v) for v in a["per_fold"]) + f"   {f(a['mean'])} ± {(a['sd'] if x=='ppl' else 100*a['sd']):.1f}")
    t = res["transformer_vs_bigram"]
    print(f"\npooled over {len(tf)} predictions: TF {100*tf.mean():.1f}% vs bigram {100*bg.mean():.1f}%  "
          f"(TF-only {b}, bigram-only {c}, McNemar p = {t['pooled_top1']['mcnemar_p']:.3f})")
    print(f"folds where TF wins: top-1 {t['folds_transformer_better_top1']}/5, perplexity {t['folds_transformer_better_ppl']}/5")


def main():
    torch.set_num_threads(4)
    keys, parts = folds()
    done = json.load(open(FOLDS)) if os.path.exists(FOLDS) else {}
    for k in range(K):
        if str(k) in done: continue
        done[str(k)] = run_fold(k, parts)
        json.dump(done, open(FOLDS, "w"))
        r = done[str(k)]
        print(f"fold {k}: TF {100*r['transformer']['top1']:.1f}% / ppl {r['transformer']['ppl']:.1f} | "
              f"bigram {100*r['bigram']['top1']:.1f}% / {r['bigram']['ppl']:.1f} | {r['seconds']}s", flush=True)
    report(done)
    ens = json.load(open(ENS)) if os.path.exists(ENS) else {}
    for k in range(K):
        if str(k) in ens: continue
        ens[str(k)] = run_ensemble_fold(k, parts); json.dump(ens, open(ENS, "w"))
        r = ens[str(k)]
        print(f"ensemble fold {k}: {100*r['ensemble']['top1']:.1f}% / ppl {r['ensemble']['ppl']:.1f} | bigram {100*r['bigram']['top1']:.1f}% / {r['bigram']['ppl']:.1f} | {r['seconds']}s", flush=True)
    res = json.load(open(OUT)); res["ensemble_cv"] = report_ensemble(ens); json.dump(res, open(OUT, "w"), indent=1)
    x = res["ensemble_cv"]; p = x["pooled"]
    print(f"\nENSEMBLE (deliverable set-up): top-1 {100*x['ensemble_top1']['mean']:.1f}% ± {100*x['ensemble_top1']['sd']:.1f}, "
          f"ppl {x['ensemble_ppl']['mean']:.1f} ± {x['ensemble_ppl']['sd']:.1f}")
    print(f"pooled vs bigram: top-1 {100*p['ensemble_top1']:.1f}% vs {100*p['bigram_top1']:.1f}% (McNemar p = {p['mcnemar_p']:.3f}); "
          f"ppl {p['ppl_ensemble']:.1f} vs {p['ppl_bigram']:.1f}, gap 95% CI {p['ppl_gap_ci95'][0]:.2f} to {p['ppl_gap_ci95'][1]:.2f}")
    print(f"folds where ensemble wins: top-1 {x['folds_ensemble_better_top1']}/5, perplexity {x['folds_ensemble_better_ppl']}/5")


def refresh_bigram():
    """Recompute only the Witten-Bell bigram in every cached fold (after the smoothing was unified);
    the Transformer results are untouched because their training did not involve the bigram."""
    keys, parts = folds()
    for path, key in ((FOLDS, "bg_correct"), (ENS, "bg_correct")):
        if not os.path.exists(path): continue
        done = json.load(open(path))
        for k in range(K):
            if str(k) not in done: continue
            test = parts[k]; rest = [s for j, p in enumerate(parts) if j != k for s in p]
            nv = len(rest) // 10; train = rest[nv:]
            cnt = collections.Counter(t for s in train for t in s[1:-1] if t != "000")
            vocab = Vocab([t for t, c in cnt.most_common() if c >= 2]); V = len(vocab)
            tr, te = encode_split(vocab, train), encode_split(vocab, test); tex = make_loo(te)
            bg = score_examples(WittenBellBigram(tr, V).probs(tex), tex)
            done[str(k)]["bigram"] = summarize(bg); done[str(k)]["bg_correct"] = bg.top1.astype(int).tolist()
            if "bg_nll" in done[str(k)]: done[str(k)]["bg_nll"] = bg.nll.tolist()
        json.dump(done, open(path, "w"))
    print("bigram refreshed in cached folds")


if __name__ == "__main__":
    if "--refresh-bigram" in sys.argv:
        refresh_bigram()
    main()
