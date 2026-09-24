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


# ---- porter stemmer, written from the 1980 paper's rule tables ----

def _cons(w: str, i: int) -> bool:
    c = w[i]
    if c in "aeiou":
        return False
    if c == "y":
        return i == 0 or not _cons(w, i - 1)
    return True


def _m(stem: str) -> int:
    # number of VC sequences, [C](VC)^m[V]
    m, prev_vowel = 0, False
    for i in range(len(stem)):
        if _cons(stem, i):
            if prev_vowel:
                m += 1
            prev_vowel = False
        else:
            prev_vowel = True
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _cons(stem, i) for i in range(len(stem)))


def _double_c(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _cons(w, len(w) - 1)


def _cvc(w: str) -> bool:
    # cons-vowel-cons, last cons not w/x/y
    n = len(w)
    return (n >= 3 and _cons(w, n - 3) and not _cons(w, n - 2) and _cons(w, n - 1)
            and w[-1] not in "wxy")


_STEP2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"),
          ("abli", "able"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
          ("ization", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
          ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")]
_STEP3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"),
          ("ful", ""), ("ness", "")]
_STEP4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent",
          "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"]
# longest suffix first, only that one is tried (condition checked after matching)
_STEP2.sort(key=lambda x: -len(x[0]))
_STEP3.sort(key=lambda x: -len(x[0]))
_STEP4.sort(key=len, reverse=True)


def _replace(w: str, rules, min_m: int) -> str:
    for suf, rep in rules:
        if w.endswith(suf):
            stem = w[: -len(suf)]
            return stem + rep if _m(stem) > min_m else w
    return w


def porter_stem(w: str) -> str:
    if len(w) <= 2:
        return w
    # 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    # 1b
    extra = False
    if w.endswith("eed"):
        if _m(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _has_vowel(w[:-2]):
        w, extra = w[:-2], True
    elif w.endswith("ing") and _has_vowel(w[:-3]):
        w, extra = w[:-3], True
    if extra:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _double_c(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif _m(w) == 1 and _cvc(w):
            w += "e"
    # 1c
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"
    # 2, 3
    w = _replace(w, _STEP2, 0)
    w = _replace(w, _STEP3, 0)
    # 4
    for suf in _STEP4:
        if w.endswith(suf):
            stem = w[: -len(suf)]
            if _m(stem) > 1 and (suf != "ion" or stem.endswith(("s", "t"))):
                w = stem
            break
    # 5a
    if w.endswith("e"):
        stem = w[:-1]
        m = _m(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            w = stem
    # 5b
    if _m(w) > 1 and _double_c(w) and w.endswith("l"):
        w = w[:-1]
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
        elif self.stem == "porter":
            t = porter_stem(tok)
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
