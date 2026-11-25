"""M4 / F5 - evaluating a proposed reading (the Sanskrit decipherment carried in the
M3 corpus CSV: columns `sanskrit` / `translation`) against structural controls.

What we can test WITHOUT the decipherment's own sign-value key (not available in
machine-readable form; see report):

T1  Consistency      - do identical sign sequences always receive the same reading?
T2  Phonetic anchoring (multinomial Naive Bayes, the classifier the proposal names):
    if signs carry fixed sound values and the text is read right-to-left, the sign the
    reading STARTS from should predict the reading's first sound. We predict the first
    phoneme (SLP1 transliteration = one character per phoneme) from:
      a) the reading-initial sign            (= last sign in stored order)
      b) the reading-final sign              (wrong end - structural control)
      c) a random sign from the inscription  (structural control)
      d) the reading-initial sign, readings shuffled across inscriptions (null)
    5-fold CV accuracy, plus the majority-class baseline.
T3  Freedom per sign - for each sign seen >= 20 times in reading-initial position, how
    many different first sounds does the decipherment give it? A fixed phonetic key
    gives ONE; every extra value is freedom the method can use to fit any text.

Readings are used only as data to be tested; we make no claim about their content.

  python src/m4_guardrails/f5_reading_test.py
Output: reports/m4/f5_reading_test.json
"""
from __future__ import annotations
import sys, json, collections, re, random
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "m3_csv" / "inscriptions.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "reports" / "m4" / "f5_reading_test.json"
SEED, NPERM = 0, 200
SIGN = re.compile(r"^\d{3}$")


def signs_of(text):
    """Stored-order signs via the shared parser (common/corpus.py); None = illegible."""
    sys.path.insert(0, str(ROOT / "src"))
    from common.corpus import parse_text
    t = parse_text(text)
    return [] if t is None else [None if x == "000" else x for x in t[1:-1]]


def first_phoneme(r):
    m = re.search(r"[A-Za-z]", r)
    return m.group(0) if m else None


def cv_acc(X, y, seed=SEED):
    y = np.asarray(y)
    keep = np.isin(y, [k for k, c in collections.Counter(y).items() if c >= 5])
    X, y = [x for x, k in zip(X, keep) if k], y[keep]
    enc = OneHotEncoder(handle_unknown="ignore").fit(np.array(X).reshape(-1, 1))
    Xm = enc.transform(np.array(X).reshape(-1, 1))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(MultinomialNB(alpha=0.5), Xm, y, cv=cv).mean()), int(len(y))


def main():
    d = pd.read_csv(CSV, dtype=str)
    s = d["sanskrit"].fillna("")
    own = d[(s != "") & ~s.str.startswith("ref:")]
    # T1 consistency
    g = own.groupby("text")["sanskrit"].nunique()
    t1 = dict(inscriptions_with_own_reading=int(len(own)), distinct_texts=int(len(g)),
              texts_with_conflicting_readings=int((g > 1).sum()))

    rows = []
    for t, r in zip(own["text"], own["sanskrit"]):
        sg = signs_of(t)
        if len(sg) < 2 or sg[0] is None or sg[-1] is None:
            continue
        ph = first_phoneme(r)
        if ph:
            rows.append(dict(init=sg[-1], final=sg[0], rand=None, all=[x for x in sg if x], ph=ph))
    rng = random.Random(SEED)
    for x in rows:
        x["rand"] = rng.choice(x["all"])
    y = [x["ph"] for x in rows]
    maj = collections.Counter(y).most_common(1)[0][1] / len(y)
    acc_init, n = cv_acc([x["init"] for x in rows], y)
    acc_final, _ = cv_acc([x["final"] for x in rows], y)
    acc_rand, _ = cv_acc([x["rand"] for x in rows], y)
    perm = []
    for k in range(NPERM):
        yy = y[:]; random.Random(k).shuffle(yy)
        perm.append(cv_acc([x["init"] for x in rows], yy, seed=k)[0])
    perm = np.array(perm)
    t2 = dict(n_inscriptions=n, majority_baseline=maj,
              acc_reading_initial_sign=acc_init, acc_reading_final_sign=acc_final,
              acc_random_sign=acc_rand, acc_shuffled_readings_mean=float(perm.mean()),
              acc_shuffled_readings_p95=float(np.percentile(perm, 95)),
              p_value_vs_shuffled=float((np.sum(perm >= acc_init) + 1) / (NPERM + 1)))

    by = collections.defaultdict(collections.Counter)
    for x in rows:
        by[x["init"]][x["ph"]] += 1
    freq = {k: v for k, v in by.items() if sum(v.values()) >= 20}
    per = []
    for k, v in sorted(freq.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(v.values())
        per.append(dict(sign=k, n=tot, distinct_first_sounds=len(v),
                        top_sound=v.most_common(1)[0][0], top_share=v.most_common(1)[0][1] / tot,
                        sounds=dict(v.most_common(6))))
    t3 = dict(signs_tested=len(per),
              median_distinct_first_sounds=float(np.median([p["distinct_first_sounds"] for p in per])),
              median_top_share=float(np.median([p["top_share"] for p in per])),
              signs_with_one_value=sum(p["distinct_first_sounds"] == 1 for p in per),
              detail=per[:15])
    res = dict(T1_consistency=t1, T2_naive_bayes_phonetic_anchoring=t2, T3_freedom_per_sign=t3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: (v if k != "T3_freedom_per_sign" else {kk: vv for kk, vv in v.items() if kk != "detail"})
                      for k, v in res.items()}, indent=1))
    for p in per[:8]:
        print(f"  sign {p['sign']}: n={p['n']:3d}  {p['distinct_first_sounds']} first sounds  top '{p['top_sound']}' {p['top_share']:.0%}  {p['sounds']}")


if __name__ == "__main__":
    main()
