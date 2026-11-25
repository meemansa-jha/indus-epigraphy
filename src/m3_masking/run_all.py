"""End-to-end Milestone 3 pipeline: splits -> baselines -> search -> train -> evaluate -> save."""
import sys, json, time, os, collections
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import *
from model import *
from baselines import *
from evaluate import *

OUT = "artifacts"; os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw", "m3_csv", "inscriptions.csv")  # see DATA_SOURCES.md
SEEDS = [0, 1, 2]
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

vocab, split, info = build_splits(CSV, OUT, seed=0, min_count=2) if not os.path.exists(f'{OUT}/splits.json') else load_splits(OUT)
V = len(vocab); log("data", info)
tr, va, te = (encode_split(vocab, split[k]) for k in ("train", "val", "test"))
vex, tex_loo, tex_multi = make_loo(va), make_loo(te), make_multi(te, 0.4, seed=1)
all_test_tokens = [t for k in split["test"] for t in k[1:-1] if t != "000"]
oov = float(np.mean([vocab.stoi.get(t, UNK_ID) == UNK_ID for t in all_test_tokens]))
log(f"val LOO ex={len(vex)} test LOO ex={len(tex_loo)} multi ex={len(tex_multi)} test OOV={oov:.3f}")

# ---------------- hyper-parameter search (1 seed, val NLL) ----------------
# v2: the search really runs (v1 had typed-in numbers). Results are written after every
# configuration to artifacts/search.json, so an interrupted run resumes where it stopped.
grid = [dict(d=64, layers=1, heads=4, ff=128, dropout=0.1),
        dict(d=64, layers=2, heads=4, ff=128, dropout=0.2),
        dict(d=128, layers=1, heads=4, ff=256, dropout=0.2),
        dict(d=128, layers=2, heads=4, ff=256, dropout=0.2),
        dict(d=128, layers=2, heads=4, ff=256, dropout=0.3)]
SEARCH = f"{OUT}/search.json"
search = json.load(open(SEARCH)) if os.path.exists(SEARCH) else []
for cfg in grid:
    if any(r["cfg"] == cfg for r in search):
        continue
    _, b, h = train_model(cfg, tr, vex, V, seed=0, epochs=60, patience=8, repeats=2)
    search.append(dict(cfg=cfg, val_nll=round(float(b), 4), epochs=len(h)))
    json.dump(search, open(SEARCH, "w"), indent=1)
    log(f"search {cfg} val_nll={b:.4f} epochs={len(h)}")
best_cfg = min(search, key=lambda r: r["val_nll"])["cfg"]
log("best config (lowest validation NLL):", best_cfg)

# ---------------- final training: several seeds -> ensemble ----------------
# v2: each seed's learning curve is saved (history_seed{s}.json) and reloaded on resume,
# so re-running never wipes the learning-curve panel.
models, hists = [], []
for s in SEEDS:
    ck_path, h_path = f"{OUT}/ckpt_seed{s}.pt", f"{OUT}/history_seed{s}.json"
    if os.path.exists(ck_path) and os.path.exists(h_path):          # resume
        ck = torch.load(ck_path, map_location="cpu", weights_only=False); m = MaskedSignEncoder(ck["V"], **ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
        models.append(m); hists.append([tuple(x) for x in json.load(open(h_path))]); log(f"seed {s} loaded"); continue
    m, b, h = train_model(best_cfg, tr, vex, V, seed=s, epochs=60, patience=8, repeats=2)
    torch.save({"state": m.state_dict(), "cfg": best_cfg, "V": V}, ck_path)
    json.dump([[float(v) for v in x] for x in h], open(h_path, "w"))
    models.append(m); hists.append(h); log(f"seed {s} val_nll={b:.3f} epochs={len(h)}")


def nn_probs(ms, examples):
    lps = [nn_logprobs(m, examples) for m in ms]
    return [np.mean([np.exp(l[i]) for l in lps], axis=0) for i in range(len(examples))]


# ---------------- temperature (fit on val only) ----------------
T = fit_temperature(nn_probs(models, vex), vex)
log("temperature", T)

# ---------------- evaluation ----------------
uni, bi = UnigramModel(tr, V), WittenBellBigram(tr, V)
results = {"info": info, "test_oov_rate": oov, "best_cfg": best_cfg, "search": search,
           "temperature": T, "protocols": {}}
dfs = {}
for pname, ex in [("loo", tex_loo), ("multi", tex_multi)]:
    R = {}
    for name, fn in [(uni.name, uni.probs), (bi.name, bi.probs)]:
        df = score_examples(fn(ex), ex); R[name] = summarize(df); dfs[(pname, name)] = df
    singles = [summarize(score_examples(nn_probs([m], ex), ex)) for m in models]
    R["Transformer (single seed, mean±std)"] = {k: [float(np.mean([s[k] for s in singles])),
                                                    float(np.std([s[k] for s in singles]))] for k in singles[0]}
    ens = nn_probs(models, ex)
    d_raw = score_examples(ens, ex); R["Transformer ensemble (uncalibrated)"] = summarize(d_raw)
    d_cal = score_examples(apply_temperature(ens, T), ex); R["Transformer ensemble (calibrated)"] = summarize(d_cal)
    dfs[(pname, "raw")], dfs[(pname, "cal")] = d_raw, d_cal
    results["protocols"][pname] = R
    log(pname, json.dumps({k: (v if isinstance(v, dict) else v) for k, v in R.items()}, default=float)[:600])

cal = dfs[("loo", "cal")]; raw = dfs[("loo", "raw")]; bdf = dfs[("loo", dict_name := bi.name)]
results["breakdowns_loo_ensemble_cal"] = {
    "by_seq_len": breakdown(cal.assign(seq_len=cal.seq_len.clip(upper=6)), "seq_len"),
    "by_position": breakdown(cal, "where")}
results["breakdowns_loo_bigram"] = {
    "by_seq_len": breakdown(bdf.assign(seq_len=bdf.seq_len.clip(upper=6)), "seq_len"),
    "by_position": breakdown(bdf, "where")}
results["breakdown_multi_by_n_masked"] = breakdown(dfs[("multi", "cal")].assign(
    n_masked=dfs[("multi", "cal")].n_masked.clip(upper=5)), "n_masked")

# frequency analysis: is the model just predicting common formula signs?
freq_rank = {i: r for r, i in enumerate(range(NSPEC, V))}     # vocab is sorted by train frequency
cal2 = cal.assign(fbucket=np.where(cal.true < NSPEC + 10, "top-10 signs", np.where(cal.true < NSPEC + 50, "rank 11-50", "rank >50")))
results["accuracy_by_true_sign_frequency"] = breakdown(cal2, "fbucket")
conf_pairs = collections.Counter((vocab.itos[t], vocab.itos[p]) for t, p, c in zip(cal.true, cal.pred, cal.top1) if not c)
results["top_confusions_true_pred"] = [[a, b, n] for (a, b), n in conf_pairs.most_common(10)]

# confidence bands (calibrated ensemble, LOO test)
bands = {}
for name in ("High", "Medium", "Low"):
    m = cal.conf.map(rating) == name
    bands[name] = dict(n=int(m.sum()), accuracy=float(cal.top1[m].mean()) if m.sum() else None,
                       mean_conf=float(cal.conf[m].mean()) if m.sum() else None)
results["confidence_bands_test"] = bands
results["ece_table_calibrated"] = ece(cal.conf.values, cal.top1.values)[1]
json.dump(dict(temperature=T, rating=RATING, best_cfg=best_cfg, seeds=SEEDS), open(f"{OUT}/config.json", "w"))
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1, default=float)

# ---------------- figures ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for h, s in zip(hists, SEEDS):
    if not h: continue
    ax[0].plot([x[1] for x in h], label=f"train s{s}", alpha=.6); ax[0].plot([x[2] for x in h], "--", label=f"val s{s}")
ax[0].set(title="Learning curves (masked-LM loss / val LOO NLL)", xlabel="epoch"); ax[0].legend(fontsize=7)
for lab, d in [("uncalibrated", raw), ("calibrated", cal)]:
    t = ece(d.conf.values, d.top1.values)[1]
    ax[1].plot([r[3] for r in t], [r[4] for r in t], "o-", label=f"{lab} (ECE {ece(d.conf.values, d.top1.values)[0]:.3f})")
ax[1].plot([0, 1], [0, 1], "k:"); ax[1].set(title="Reliability (test, LOO)", xlabel="mean confidence", ylabel="accuracy"); ax[1].legend()
b1 = results["breakdowns_loo_ensemble_cal"]["by_seq_len"]; b2 = results["breakdowns_loo_bigram"]["by_seq_len"]
ks = sorted(b1, key=int); x = np.arange(len(ks))
ax[2].bar(x - .2, [b1[k]["top1"] for k in ks], .4, label="Transformer ens."); ax[2].bar(x + .2, [b2[k]["top1"] for k in ks], .4, label="WB bigram")
ax[2].set_xticks(x); ax[2].set_xticklabels([k if k != "6" else "6+" for k in ks]); ax[2].set(title="Top-1 by #readable signs", xlabel="signs in sequence"); ax[2].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/figures.png", dpi=130)
log("done")
