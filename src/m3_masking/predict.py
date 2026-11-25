"""Context-aware sign completion with calibrated confidence ratings (Milestone 3 deliverable).

The model works internally in WELLS numbering and STORED order (left-to-right as transcribed,
which is the reverse of right-to-left reading order). v2 adds explicit conversion so that input
from other sources cannot be misread silently:

    numbering = "wells" (default) | "im77" (Milestone 1 output) | "parpola" (CISI / Milestone 2)
    order     = "stored" (default, corpus order) | "reading" (right-to-left, Milestone 1 output)

Python:
    from predict import SignCompleter
    sc = SignCompleter()
    sc.complete("+740-000-590+")                                  # corpus string: Wells, stored order
    sc.complete(["740", "?", "590"])                              # list, Wells, stored order
    sc.complete(["342", "?", "59"], numbering="im77", order="reading")   # Milestone 1 output
    sc.complete(["P385", "?", "P324"], numbering="parpola", order="reading")
    sc.complete("+740-220-590+", mask=[1])                        # hide sign #1 to test

CLI:
    python predict.py "+740-000-590+" --top-k 5
    python predict.py "342 ? 59" --numbering im77 --order reading

Results are reported in Wells AND translated back into the input numbering and order.
Signs that cannot be converted (no crosswalk entry) are treated as unreadable gaps' context
holes and listed under `unmapped`; they are never guessed.
"""
import argparse, json, os, re, sys
import numpy as np, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from data import (NSPEC, S_OK, S_BRK, E_OK, E_BRK, parse_text, load_splits)  # noqa: E402
from model import MaskedSignEncoder, pad_batch  # noqa: E402
from evaluate import apply_temperature, rating  # noqa: E402
from common.signs import im77_to_wells, wells_to_im77, parpola_to_wells  # noqa: E402

GAP = {"?", "000", "[MASK]", "*"}
NUMBERINGS = ("wells", "im77", "parpola")


class SignCompleter:
    def __init__(self, art_dir=os.path.join(HERE, "artifacts")):
        self.vocab, _, _ = load_splits(art_dir)
        cfg = json.load(open(f"{art_dir}/config.json"))
        self.T = cfg["temperature"]
        self.models = []
        for s in cfg["seeds"]:
            ck = torch.load(f"{art_dir}/ckpt_seed{s}.pt", map_location="cpu", weights_only=False)
            m = MaskedSignEncoder(ck["V"], **ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
            self.models.append(m)
        self._freq = {t: -i for i, t in enumerate(self.vocab.itos)}   # vocab is sorted by train frequency

    # ---- numbering conversion ----
    def _to_wells(self, sid, numbering):
        """-> (wells_id or None, note)"""
        if numbering == "wells":
            return str(sid).zfill(3), None
        cands = im77_to_wells(int(str(sid).lstrip("Mm"))) if numbering == "im77" else parpola_to_wells(str(sid).upper())
        if not cands:
            return None, f"{numbering} {sid}: no Wells equivalent"
        best = max(cands, key=lambda w: self._freq.get(w, -10 ** 6))
        return best, (f"{numbering} {sid} -> Wells {best} (ambiguous: {'/'.join(cands)})" if len(cands) > 1 else None)

    def _from_wells(self, w, numbering):
        if numbering == "wells":
            return w
        if numbering == "im77":
            return "/".join(map(str, wells_to_im77(w))) or f"W{w}"
        # parpola: invert parpola_to_wells
        from common.signs import _rows
        ps = [r["parpola"] for r in _rows() if w in r["wells"]]
        return "/".join(ps) or f"W{w}"

    # ---- input handling ----
    def _tokens(self, seq, numbering, order, edges):
        notes, unmapped = [], []
        if isinstance(seq, str) and re.match(r"^[+\[\]]", seq.strip()):
            if numbering != "wells" or order != "stored":
                raise ValueError("corpus-format strings (+...+) are Wells numbering in stored order; "
                                 "pass a list for other numberings/orders")
            s = re.sub(r"\[MASK\]|\*+|\?+", "000", seq.strip())
            return parse_text(s), notes, unmapped
        items = seq.split() if isinstance(seq, str) else list(seq)
        toks = []
        for t in items:
            t = str(t)
            if t in GAP:
                toks.append("000"); continue
            w, note = self._to_wells(t, numbering)
            if note: notes.append(note)
            if w is None:
                unmapped.append(t); toks.append("UNMAPPED")
            else:
                toks.append(w)
        if order == "reading":
            toks = toks[::-1]
        a, b = (S_OK, E_OK) if edges == "complete" else (S_BRK, E_BRK)
        return [a] + toks + [b], notes, unmapped

    @torch.no_grad()
    def complete(self, seq, top_k=5, mask=None, edges="complete", numbering="wells", order="stored"):
        if numbering not in NUMBERINGS or order not in ("stored", "reading"):
            raise ValueError(f"numbering must be one of {NUMBERINGS}; order 'stored' or 'reading'")
        toks, notes, unmapped = self._tokens(seq, numbering, order, edges)
        if toks is None:
            raise ValueError("empty sequence")
        n = len(toks) - 2
        if mask:                                     # positions are given in the INPUT order
            for i in mask:
                j = (n - 1 - i) if order == "reading" else i
                toks[j + 1] = "000"
        gaps = [i for i, t in enumerate(toks) if t == "000"]
        if not gaps:
            raise ValueError("no gaps found: put ? (or 000 / [MASK]) where a sign is missing, or pass mask=[...]")
        unknown = [t for t in toks[1:-1] if t not in ("000", "UNMAPPED") and t not in self.vocab.stoi]
        x = self.vocab.encode(["[UNK]" if t == "UNMAPPED" else t for t in toks])
        batch = pad_batch([x])
        probs = np.mean([F.softmax(m(batch), -1)[0, gaps].numpy() for m in self.models], axis=0)
        probs = apply_temperature([probs], self.T)[0]
        out, filled = [], list(toks)
        for r, g in enumerate(gaps):
            order_ix = np.argsort(-probs[r])[:top_k]
            cands = [dict(sign=self.vocab.itos[i], sign_in_input_numbering=self._from_wells(self.vocab.itos[i], numbering),
                          prob=float(probs[r, i])) for i in order_ix]
            pos_stored = g - 1
            pos_input = (n - 1 - pos_stored) if order == "reading" else pos_stored
            out.append(dict(position=pos_input, candidates=cands,
                            confidence=cands[0]["prob"], rating=rating(cands[0]["prob"])))
            filled[g] = cands[0]["sign"]
        body_w = filled[1:-1]
        body_in = [self._from_wells(t, numbering) if t not in ("UNMAPPED",) else "?" for t in body_w]
        if order == "reading":
            body_in = body_in[::-1]
        return dict(input="-".join(toks[1:-1]), completed="-".join(body_w),
                    completed_in_input_numbering_and_order=" ".join(body_in),
                    numbering=numbering, order=order, gaps=sorted(out, key=lambda d: d["position"]),
                    conversion_notes=notes, unmapped_input_signs=unmapped, unknown_context_signs=unknown,
                    note="Internal sequences are Wells numbering in stored order (reverse of reading order). "
                         "Each gap is predicted with all other gaps hidden; confidence is calibrated on validation data.")


def _pretty(res):
    print(f"internal (Wells, stored order): {res['input']}  ->  {res['completed']}")
    print(f"in your numbering/order ({res['numbering']}, {res['order']}): {res['completed_in_input_numbering_and_order']}")
    for k in ("conversion_notes", "unmapped_input_signs", "unknown_context_signs"):
        if res[k]: print(f"  {k}: {res[k]}")
    for g in res["gaps"]:
        c = ", ".join(f"{d['sign_in_input_numbering']} ({d['prob']:.2f})" for d in g["candidates"])
        print(f"  gap @ sign #{g['position']}: [{g['rating']} confidence {g['confidence']:.2f}]  {c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence", help='corpus string "+740-000-590+" or space-separated IDs "342 ? 59"')
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--edges", default="complete", choices=["complete", "broken"])
    ap.add_argument("--mask", type=int, nargs="*")
    ap.add_argument("--numbering", default="wells", choices=NUMBERINGS)
    ap.add_argument("--order", default="stored", choices=["stored", "reading"])
    a = ap.parse_args()
    _pretty(SignCompleter().complete(a.sequence, a.top_k, a.mask, a.edges, a.numbering, a.order))
