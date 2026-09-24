# unigram LM stuff: analyzer, collection stats, QL scoring (dirichlet + JM)
# pure stdlib on purpose, grader only promises basic packages
import json
import math
import re
from collections import Counter
from typing import Dict, List, Optional

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# common english function words, own list. kept small on purpose
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be been before being below between both but by
can could did do does doing down during each either few for from further had has have having he her here hers him
his how i if in into is it its itself just may me might more most must my no nor not of off on once only or other
our ours out over own same shall she should so some such than that the their theirs them then there these they this
those through to too under until up upon very was we were what when where whether which while who whom whose why will
with within would you your yours
""".split())


def s_stem(w: str) -> str:
    # harman's s-stemmer, only plural endings, very conservative
    if len(w) > 3 and w.endswith("ies") and not w.endswith(("eies", "aies")):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and not w.endswith(("aes", "ees", "oes")):
        return w[:-1]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("us", "ss")):
        return w[:-1]
    return w


class Analyzer:
    """text -> list of terms. same one must be used for docs and queries."""

    def __init__(self, stop: bool = True, stem: str = "none"):
        self.stop = stop
        self.stem = stem
        self._cache: Dict[str, Optional[str]] = {}

    def _term(self, tok: str) -> Optional[str]:
        t = self._cache.get(tok, 0)
        if t != 0:
            return t
        if self.stop and tok in STOPWORDS:
            t = None
        elif self.stem == "s":
            t = s_stem(tok)
        else:
            t = tok
        self._cache[tok] = t
        return t

    def __call__(self, text: str) -> List[str]:
        out = []
        for tok in _TOKEN_RE.findall(text.lower()):
            t = self._term(tok)
            if t is not None:
                out.append(t)
        return out


class Stats:
    """collection counts built once; per doc counts done lazily for docs we actually score"""

    def __init__(self, analyzer: Analyzer):
        self.an = analyzer
        self.texts: Dict[str, str] = {}
        self.doc_len: Dict[str, int] = {}
        self.cf: Counter = Counter()
        self.total = 0
        self._tf: Dict[str, Counter] = {}

    @classmethod
    def from_jsonl(cls, path: str, analyzer: Analyzer) -> "Stats":
        st = cls(analyzer)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                st.add(obj["doc_id"], obj["text"])
        return st

    def add(self, doc_id: str, text: str) -> None:
        terms = self.an(text)
        self.texts[doc_id] = text
        self.doc_len[doc_id] = len(terms)
        self.cf.update(terms)
        self.total += len(terms)

    def tf(self, doc_id: str) -> Counter:
        c = self._tf.get(doc_id)
        if c is None:
            # unknown doc id -> treat as empty doc instead of crashing the query
            c = Counter(self.an(self.texts.get(doc_id, "")))
            self._tf[doc_id] = c
        return c

    def dl(self, doc_id: str) -> int:
        return self.doc_len.get(doc_id, 0)

    def p_c(self, term: str) -> float:
        return self.cf.get(term, 0) / self.total if self.total else 0.0

    @property
    def avgdl(self) -> float:
        return self.total / len(self.doc_len) if self.doc_len else 0.0


def ql_score(q_terms: List[str], tf: Counter, dl: int, st: Stats, method: str, param: float) -> float:
    """log P(Q|D) = sum over query tokens of log P(w|D).
    dirichlet: (tf + mu*p_c)/(dl + mu)      jm: (1-lam)*tf/dl + lam*p_c
    terms never seen in the collection are skipped (same -inf for every doc, so no effect on ranking)"""
    s = 0.0
    for w in q_terms:
        pc = st.p_c(w)
        if pc == 0.0:
            continue
        c = tf.get(w, 0)
        if method == "dirichlet":
            p = (c + param * pc) / (dl + param)
        elif method == "jm":
            p = (1 - param) * (c / dl if dl else 0.0) + param * pc
        else:
            raise ValueError(method)
        s += math.log(p)
    return s


def rank(scores: Dict[str, float], k: int):
    # ties broken by doc id so we never lean on input order
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
