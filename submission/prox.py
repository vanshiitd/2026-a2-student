# term proximity bonus on top of unigram QL (tao & zhai 2007, "MinDist")
#   bonus = w * log(1 + exp(-mindist) / alpha)
# mindist = smallest gap between occurrences of two *different* query terms in the doc.
# this is their log(alpha + exp(-d)) shifted by -log(alpha), so a doc with fewer than 2 distinct
# query terms gets exactly 0 and a one-word query is plain QL. only needs positions inside the doc,
# no collection bigram counts. w = 0 turns it off.
import math
from typing import Dict, List, Optional


def min_dist(q_set, terms: List[str]) -> Optional[int]:
    last: Dict[str, int] = {}
    best = None
    for i, t in enumerate(terms):
        if t not in q_set:
            continue
        for u, j in last.items():
            if u != t and (best is None or i - j < best):
                best = i - j
        last[t] = i
    return best


def bonus(q: List[str], docs: List[str], st, w: float, alpha: float) -> Dict[str, float]:
    qs = set(q)
    out = {}
    for d in docs:
        md = min_dist(qs, st.terms(d))
        out[d] = w * math.log1p(math.exp(-md) / alpha) if md is not None else 0.0
    return out
