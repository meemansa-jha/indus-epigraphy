"""M4 / F8 - generating Indus-like sign sequences (the professor's request), judged by M4.

Two generators, both trained on the M3 TRAIN split only:
  markov  sample from the Witten-Bell bigram (M2-style Markov chain): <S> -> signs -> <E>
  gibbs   start from a Markov sample, then 10 sweeps of: hide one sign, resample it from
          the Milestone 3 Transformer ensemble (calibrated) given all the others

Each generated set (n = 1000) is judged exactly like real text and the controls:
  guardrail pseudo-perplexity and flag rate, Naive Bayes real-vs-generated AUC (5-fold CV),
  numbering-free statistics (length, distinct signs, repeats), and NOVELTY
  (share of outputs that are exact copies of a training inscription = memorisation).

  python src/m4_guardrails/f8_generate.py
Outputs: data/processed/m4/generated.json, reports/m4/f8_generation.json
"""
from __future__ import annotations
import sys, json, collections
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from guardrail import scorer  # noqa: E402
from data import NSPEC, MASK_ID, E_OK  # noqa: E402
from model import pad_batch  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

PROC = ROOT / "data" / "processed" / "m4"
REP = ROOT / "reports" / "m4"
N, MAXLEN, SWEEPS, SEED = 1000, 15, 10, 0


def markov(S, rng):
    P = S.bigram.P.copy()
    allowed = np.zeros(S.V, bool); allowed[NSPEC:] = True
    e = S.vocab.stoi[E_OK]
    out = []
    while len(out) < N:
        seq, prev = [], S.vocab.stoi["<S>"]
        while len(seq) < MAXLEN:
            p = P[prev] * allowed
            p[e] = P[prev][e] if seq else 0          # at least one sign
            p /= p.sum()
            nxt = rng.choice(S.V, p=p)
            if nxt == e:
                break
            seq.append(nxt); prev = nxt
        if nxt == e:
            out.append(["<S>"] + [S.vocab.itos[i] for i in seq] + ["<E>"])
    return out


def gibbs(S, seqs, rng):
    out = []
    for s in seqs:
        ids = S.vocab.encode(s)
        for _ in range(SWEEPS):
            for i in rng.permutation(range(1, len(ids) - 1)):
                x = list(ids); x[i] = MASK_ID
                with torch.no_grad():
                    p = np.mean([F.softmax(m(pad_batch([x])) / S.T, -1)[0, i].numpy() for m in S.models], 0)
                p[:NSPEC] = 0; p /= p.sum()
                ids[i] = rng.choice(S.V, p=p)
        out.append(S.vocab.decode(ids))
    return out


def stats(seqs, train_set):
    bodies = [s[1:-1] for s in seqs]
    L = np.array([len(b) for b in bodies]); u = collections.Counter(t for b in bodies for t in b)
    pairs = [(a, c) for b in bodies for a, c in zip(b[:-1], b[1:])]
    return dict(mean_length=float(L.mean()), distinct_signs=len(u),
                top_sign_share=float(u.most_common(1)[0][1] / sum(u.values())),
                adjacent_repeat_rate=float(np.mean([a == c for a, c in pairs])) if pairs else 0.0,
                copies_of_training_inscriptions=float(np.mean([tuple(b) in train_set for b in bodies])),
                distinct_outputs=len(set(map(tuple, bodies))))


def nb_auc(real, fake):
    # generators only emit complete edges; normalise broken-edge markers so NB cannot
    # separate the sets on edge type alone
    norm = lambda s: " ".join({"<S?>": "<S>", "<E?>": "<E>"}.get(t, t) for t in s)
    X = [norm(s) for s in real] + [norm(s) for s in fake]
    y = np.r_[np.ones(len(real)), np.zeros(len(fake))]
    Xm = CountVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), lowercase=False).fit_transform(X)
    return float(cross_val_score(MultinomialNB(alpha=.5), Xm, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                 scoring="roc_auc").mean())


def main():
    S = scorer(); rng = np.random.default_rng(SEED)
    mk = markov(S, rng)
    gb = gibbs(S, mk, rng)
    train_set = set(tuple(s[1:-1]) for s in S.split["train"])
    real_all = [s for s in json.load(open(PROC / "controls_all.json"))["sets"]["real"]]
    real_test = json.load(open(PROC / "controls_test.json"))["sets"]["real"]
    rr = np.random.default_rng(1)
    res = {}
    for name, seqs in [("real (held-out test)", real_test), ("markov", mk), ("gibbs (M3)", gb)]:
        p, flag = S.verdict([s for s in seqs if S.n_known(s) >= 2])
        r = dict(n=len(seqs), median_pppl=float(np.nanmedian(p)), guardrail_flag_rate=float(np.nanmean(flag)),
                 **stats(seqs, train_set))
        if name != "real (held-out test)":
            sub = [real_all[i] for i in rr.choice(len(real_all), len(seqs), replace=False)]
            r["nb_auc_vs_real"] = nb_auc(sub, seqs)
        res[name] = r
        print(f"{name:22s} pppl {r['median_pppl']:6.1f}  flagged {r['guardrail_flag_rate']:5.1%}  len {r['mean_length']:.2f}"
              f"  signs {r['distinct_signs']:3d}  repeats {r['adjacent_repeat_rate']:.1%}  copies {r['copies_of_training_inscriptions']:.1%}"
              + (f"  NB AUC {r['nb_auc_vs_real']:.3f}" if "nb_auc_vs_real" in r else ""))
    syn = json.load(open(REP / "synthetic_check.json"))["synthetic"]
    res["hellosindh synthetic (numbering-free stats only)"] = {k: syn[k] for k in
        ("mean_length", "distinct_signs", "top_sign_share", "adjacent_repeat_rate")}
    json.dump(dict(mk=mk, gibbs=gb), open(PROC / "generated.json", "w"))
    json.dump(dict(settings=dict(n=N, maxlen=MAXLEN, gibbs_sweeps=SWEEPS, seed=SEED), results=res,
                   examples=dict(markov=["-".join(s[1:-1]) for s in mk[:6]], gibbs=["-".join(s[1:-1]) for s in gb[:6]])),
              open(REP / "f8_generation.json", "w"), indent=1)
    print("examples markov:", ["-".join(s[1:-1]) for s in mk[:3]])
    print("examples gibbs :", ["-".join(s[1:-1]) for s in gb[:3]])


if __name__ == "__main__":
    main()
