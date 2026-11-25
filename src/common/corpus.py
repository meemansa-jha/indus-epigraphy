"""The project's single loader for the inscription corpus (Wells numbering).

Every milestone reads the corpus through here, so parsing and deduplication are defined once.

Text format in the CSV `text` column, e.g. '+400-740-405-590[':
    '-' separates signs;  '+' = complete edge;  '[' or ']' = broken edge;
    '/' = alternative readings (the first readable one is used);  '000' = illegible sign.
Tokens are returned in STORED order (left-to-right as transcribed = reverse of reading order;
see order.py) with edge markers:  ['<S>', '400', '740', '405', '590', '<E?>']
    first token  '<S>' / '<S?>'  = stored-left edge complete / broken   (= reading END)
    last token   '<E>' / '<E?>'  = stored-right edge complete / broken  (= reading START)

Deduplication policies
    unique_sequences()  one entry per distinct SIGN sequence (Milestone 3 v2 policy; keeps the most
                        frequent edge variant; `count` = number of rows with those signs)
    rows()              every row (duplicates kept), for statistics that should count every object
"""
import collections, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "raw" / "m3_csv" / "inscriptions.csv"
S_OK, S_BRK, E_OK, E_BRK = "<S>", "<S?>", "<E>", "<E?>"
SIGN_RE = re.compile(r"^\d{3}$")


def parse_text(s):
    """'+400-740-405-590[' -> ['<S>','400','740','405','590','<E?>']  (None if empty)."""
    s = str(s).strip()
    if not s or s == "nan":
        return None
    start = S_OK if s[0] == "+" else S_BRK
    end = E_OK if s[-1] == "+" else E_BRK
    body = s.strip("+[]")
    toks = []
    for el in body.split("-"):
        alts = [a for a in el.split("/") if SIGN_RE.match(a)]
        if not alts:
            toks.append("000")           # unparseable -> treat as damaged
            continue
        good = [a for a in alts if a != "000"]
        toks.append(good[0] if good else "000")   # first readable alternative
    if not toks:
        return None
    return [start] + toks + [end]


def read_csv(csv_path=CSV):
    return pd.read_csv(csv_path, dtype=str)


def rows(csv_path=CSV):
    """-> list of (row_dict, tokens) for every row with a parseable text (duplicates kept)."""
    out = []
    for _, r in read_csv(csv_path).iterrows():
        t = parse_text(r["text"])
        if t is not None:
            out.append((r.to_dict(), t))
    return out


def unique_sequences(csv_path=CSV):
    """-> (n_rows, OrderedDict{token_tuple: {'site', 'count', 'edge_variants'}}).
    Deduplicates on the sign sequence only; texts with no readable sign are dropped."""
    d = read_csv(csv_path)
    groups = collections.OrderedDict()
    for _, r in d.iterrows():
        toks = parse_text(r["text"])
        if toks is None or all(t == "000" for t in toks[1:-1]):
            continue
        body = tuple(toks[1:-1])
        g = groups.setdefault(body, {"site": r.get("site", "-"), "edges": collections.Counter(), "count": 0})
        g["edges"][(toks[0], toks[-1])] += 1
        g["count"] += 1
    seqs = collections.OrderedDict()
    for body, g in groups.items():
        (s, e), _ = g["edges"].most_common(1)[0]
        seqs[(s,) + body + (e,)] = {"site": g["site"], "count": g["count"], "edge_variants": len(g["edges"])}
    return len(d), seqs
