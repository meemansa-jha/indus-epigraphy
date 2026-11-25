"""One place for reading direction.

The script reads RIGHT-TO-LEFT. Both corpora (the Wells CSV and CISI) are transcribed LEFT-TO-RIGHT
as the signs appear on the impression, so the first STORED sign is the last sign READ.
Evidence: the jar sign (Wells 740 = Parpola P324 = Mahadevan 342), a well-known text-ender, is the
first stored sign in 71% of right-to-left inscriptions and the last in 3%; Mahadevan's Table I
gives it 69.6% final position in reading order.

Stored tokens follow common/corpus.py: ['<S>' | '<S?>', signs ..., '<E>' | '<E?>'].
"""


def to_reading(stored):
    """stored tokens -> (signs in reading order, reading_start_complete, reading_end_complete)."""
    return list(stored[1:-1])[::-1], stored[-1] == "<E>", stored[0] == "<S>"


def to_stored(signs_reading, start_complete=True, end_complete=True):
    """signs in reading order -> stored tokens."""
    return [("<S>" if end_complete else "<S?>")] + list(signs_reading)[::-1] + [("<E>" if start_complete else "<E?>")]


def reading_tokens(stored):
    """Same tokens, reading order, edge markers swapped to the matching ends (used by word2vec / LLM2Vec)."""
    return [stored[-1]] + list(stored[1:-1])[::-1] + [stored[0]]


def segments(stored, gap="000"):
    """Milestone 2 definition: reading-order segments cut at illegible signs.
    -> list of (signs, genuine_start, genuine_end); an edge next to damage is not genuine."""
    body, start_ok, end_ok = to_reading(stored)
    out, cur, ok = [], [], start_ok
    for t in body:
        if t == gap:
            if cur: out.append((cur, ok, False))
            cur, ok = [], False
        else:
            cur.append(t)
    if cur: out.append((cur, ok, end_ok))
    return out
