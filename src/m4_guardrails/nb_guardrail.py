"""M4 step 5 - Multinomial Naive Bayes guardrail: can a simple classifier tell
real inscriptions from each kind of fake?

Features: counts of single signs and adjacent sign pairs, with the edge markers
(<S>, <E>, broken-edge <S?>, <E?>) kept as tokens, so "sign X opens the text" is a feature.
Fakes share each real inscription's length and edges, so length alone cannot give them away.

One balanced binary task per fake type, 5-fold stratified cross-validation.
Chance = 0.5 accuracy / 0.5 AUC.

  python src/m4_guardrails/nb_guardrail.py
Outputs: reports/m4/nb_guardrail.json, nb_guardrail.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed" / "m4"
REP = ROOT / "reports" / "m4"
FAKES = ["synthetic", "shuffled", "markov", "freq_random", "uniform_random"]
LABEL = {"synthetic": "Synthetic corpus", "shuffled": "Shuffled", "markov": "Markov-generated",
         "freq_random": "Frequency-random", "uniform_random": "Uniform-random"}
SEED = 0


def doc(seq):
    return " ".join(seq)


def run(real, fake):
    X = [doc(s) for s in real] + [doc(s) for s in fake]
    y = np.array([1] * len(real) + [0] * len(fake))
    vec = CountVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), lowercase=False)
    Xm = vec.fit_transform(X)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    r = cross_validate(MultinomialNB(alpha=0.5), Xm, y, cv=cv, scoring=["accuracy", "roc_auc"])
    nb = MultinomialNB(alpha=0.5).fit(Xm, y)               # refit on all data, for interpretation only
    lr = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]  # >0 = points to REAL
    names = vec.get_feature_names_out()
    order = np.argsort(-lr)
    return dict(accuracy=float(r["test_accuracy"].mean()), accuracy_sd=float(r["test_accuracy"].std()),
                auc=float(r["test_roc_auc"].mean()), auc_sd=float(r["test_roc_auc"].std()),
                n_real=len(real), n_fake=len(fake),
                top_real_features=[[str(names[i]), round(float(lr[i]), 2)] for i in order[:10]],
                top_fake_features=[[str(names[i]), round(float(lr[i]), 2)] for i in order[::-1][:10]])


def main():
    sets = json.load(open(PROC / "controls_all.json"))["sets"]
    syn = PROC / "synthetic_tokens.json"
    if syn.exists():
        j = json.load(open(syn))["sets"]
        sets["synthetic"], sets["real_for_synthetic"] = j["synthetic"], j["real"]
    REP.mkdir(parents=True, exist_ok=True)
    out = {}
    for k in [k for k in FAKES if k in sets]:
        real = sets["real_for_synthetic"] if k == "synthetic" else sets["real"]
        out[k] = run(real, sets[k])
        o = out[k]
        print(f"real vs {LABEL[k]:17s} acc {o['accuracy']:.3f}±{o['accuracy_sd']:.3f}   AUC {o['auc']:.3f}"
              f"   | top 'real' cues: {', '.join(f[0] for f in o['top_real_features'][:4])}")
    json.dump(out, open(REP / "nb_guardrail.json", "w"), indent=1)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = list(out)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    vals = [out[k]["auc"] for k in ks]
    ax.barh([LABEL[k] for k in ks], vals, xerr=[out[k]["auc_sd"] for k in ks],
            color=["#e9c46a" if k == "synthetic" else "#2a9d8f" for k in ks])
    ax.axvline(0.5, ls=":", c="k"); ax.text(0.505, len(ks) - 0.55, "chance", fontsize=8)
    for i, v in enumerate(vals):
        ax.text(min(v + 0.01, 0.93), i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0.4, 1.0); ax.invert_yaxis()
    ax.set_xlabel("Naive Bayes AUC, real vs fake (5-fold CV)")
    ax.set_title("How easily a simple classifier spots each kind of fake", fontsize=11)
    plt.tight_layout(); plt.savefig(REP / "nb_guardrail.png", dpi=140)
    print(f"-> {REP / 'nb_guardrail.png'}")


if __name__ == "__main__":
    main()
