"""Small masked-sign Transformer encoder (TinyBERT-style, trained from scratch)."""
import math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from data import NSPEC, MASK_ID, PAD_ID

torch.set_num_threads(1)


class MaskedSignEncoder(nn.Module):
    def __init__(self, V, d=128, layers=2, heads=4, ff=256, dropout=0.2, max_len=32):
        super().__init__()
        self.V, self.cfg = V, dict(d=d, layers=layers, heads=heads, ff=ff, dropout=dropout, max_len=max_len)
        self.tok = nn.Embedding(V, d, padding_idx=PAD_ID)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.bias = nn.Parameter(torch.zeros(V))
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.tok.weight, std=d ** -0.5)
        nn.init.normal_(self.pos.weight, std=d ** -0.5)
        with torch.no_grad():
            self.tok.weight[PAD_ID].zero_()

    def forward(self, ids):
        pad = ids == PAD_ID
        pos = torch.arange(ids.size(1), device=ids.device)[None]
        h = self.drop(self.tok(ids) + self.pos(pos))
        h = self.ln(self.enc(h, src_key_padding_mask=pad))
        logits = h @ self.tok.weight.T + self.bias          # tied output embedding
        logits[..., :NSPEC] = -1e9                          # only real signs can be predicted
        return logits


def pad_batch(seqs):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD_ID] * (L - len(s)) for s in seqs])


def random_mask(batch, p, V, gen):
    """BERT-style dynamic masking (80% [MASK], 10% random sign, 10% keep) with >=1 target/seq."""
    ids = batch.clone()
    elig = ids >= NSPEC
    r = torch.rand(ids.shape, generator=gen)
    chosen = elig & (r < p)
    none = ~chosen.any(1)
    if none.any():                                   # force one target in sequences with none
        pick = (torch.rand(ids.shape, generator=gen) * elig).argmax(1)
        chosen[none, pick[none]] = True
    labels = torch.where(chosen, ids, torch.full_like(ids, -100))
    r2 = torch.rand(ids.shape, generator=gen)
    ids = torch.where(chosen & (r2 < 0.8), torch.full_like(ids, MASK_ID), ids)
    rnd = torch.randint(NSPEC, V, ids.shape, generator=gen)
    ids = torch.where(chosen & (r2 >= 0.8) & (r2 < 0.9), rnd, ids)
    return ids, labels


@torch.no_grad()
def nn_logprobs(model, examples, bs=256):
    """Return list of (n_targets, V) log-probabilities (specials = -inf)."""
    model.eval()
    out = []
    for i in range(0, len(examples), bs):
        chunk = examples[i:i + bs]
        lp = F.log_softmax(model(pad_batch([e["x"] for e in chunk])), -1)
        for j, e in enumerate(chunk):
            out.append(lp[j, e["pos"]].numpy())
    return out


def val_nll(model, examples):
    lps = nn_logprobs(model, examples)
    tot = n = 0
    for lp, e in zip(lps, examples):
        for r, y in enumerate(e["y"]):
            tot -= lp[r, y]; n += 1
    return tot / n


def train_model(cfg, train_ids, val_examples, V, seed, epochs=80, patience=10,
                lr=2e-3, wd=0.01, bs=64, mask_p=0.25, repeats=4, verbose=False):
    torch.manual_seed(seed); rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    model = MaskedSignEncoder(V, **cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(train_ids) * repeats // bs + 1))
    best, best_state, bad, hist = 1e9, None, 0, []
    for ep in range(epochs):
        model.train()
        data = train_ids * repeats
        rng.shuffle(data)
        tl = 0; nb = 0
        for i in range(0, len(data), bs):
            batch = pad_batch(data[i:i + bs])
            x, y = random_mask(batch, mask_p, V, gen)
            loss = F.cross_entropy(model(x).view(-1, V), y.view(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item(); nb += 1
        v = val_nll(model, val_examples)
        hist.append((ep, tl / nb, v))
        if verbose:
            print(f"  ep{ep:3d} train {tl/nb:.3f} val {v:.3f}", flush=True)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, hist
