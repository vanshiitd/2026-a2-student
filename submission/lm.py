# unigram LM stuff: analyzer, collection stats, QL scoring (dirichlet + JM)
# pure stdlib on purpose, grader only promises basic packages
import json
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

# tokens = runs of [a-z0-9] after lowercasing. done with a byte translate table instead of a regex,
# ~2x faster on the full corpus and gives exactly the same tokens (non-ascii chars -> '?' -> separator)
_TABLE = bytearray(b" " * 256)
for _c in b"abcdefghijklmnopqrstuvwxyz0123456789":
    _TABLE[_c] = _c
_TABLE = bytes(_TABLE)


def tokens(text: str) -> List[str]:
    return text.lower().encode("ascii", "replace").translate(_TABLE).decode("ascii").split()


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

    def term(self, tok: str) -> Optional[str]:
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
        term = self.term
        return [t for t in map(term, tokens(text)) if t is not None]


class Stats:
    """collection counts built once. doc text is NOT kept in memory, only (offset, length) of its
    line in the corpus file; per doc term counts are read + cached lazily for docs we actually score"""

    def __init__(self, analyzer: Analyzer):
        self.an = analyzer
        self.loc: Dict[str, Tuple[int, int]] = {}
        self.doc_len: Dict[str, int] = {}
        self.cf: Counter = Counter()
        self.total = 0
        self._tf: Dict[str, Counter] = {}
        self._fd: Optional[int] = None

    @classmethod
    def from_jsonl(cls, path: str, analyzer: Analyzer) -> "Stats":
        st = cls(analyzer)
        raw = Counter()  # raw token counts over the whole collection, stemming done once per type at the end
        stop = STOPWORDS if analyzer.stop else frozenset()
        off = 0
        with open(path, "rb") as f:
            for line in f:
                n = len(line)
                if line.strip():
                    obj = json.loads(line)
                    toks = tokens(obj["text"])
                    raw.update(toks)
                    st.loc[obj["doc_id"]] = (off, n)
                    # analyzer only ever drops stopwords, so length = tokens - stopwords
                    st.doc_len[obj["doc_id"]] = len(toks) - sum(map(stop.__contains__, toks))
                off += n
        for tok, c in raw.items():
            t = analyzer.term(tok)
            if t is not None:
                st.cf[t] += c
        st.total = sum(st.cf.values())
        st._fd = os.open(path, os.O_RDONLY)
        return st

    def _text(self, doc_id: str) -> str:
        loc = self.loc.get(doc_id)
        if loc is None:
            return ""
        # pread = positional read, no shared file offset, so its safe even if the process forks
        return json.loads(os.pread(self._fd, loc[1], loc[0]))["text"]

    def tf(self, doc_id: str) -> Counter:
        c = self._tf.get(doc_id)
        if c is None:
            # unknown doc id -> treat as empty doc instead of crashing the query
            c = Counter(self.an(self._text(doc_id)))
            self._tf[doc_id] = c
        return c

    def dl(self, doc_id: str) -> int:
        return self.doc_len.get(doc_id, 0)

    def p_c(self, term: str) -> float:
        return self.cf.get(term, 0) / self.total if self.total else 0.0

    @property
    def avgdl(self) -> float:
        return self.total / len(self.doc_len) if self.doc_len else 0.0

    def __del__(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass


def p_smooth(c: int, dl: int, pc: float, method: str, param: float) -> float:
    """smoothed P(w|D). dirichlet: (c + mu*pc)/(dl + mu)    jm: (1-lam)*c/dl + lam*pc"""
    if method == "dirichlet":
        return (c + param * pc) / (dl + param)
    if method == "jm":
        return (1 - param) * (c / dl if dl else 0.0) + param * pc
    raise ValueError(method)


def ql_score(q_terms: List[str], tf: Counter, dl: int, st: Stats, method: str, param: float) -> float:
    """log P(Q|D) = sum over query tokens of log P(w|D), smoothing as in p_smooth.
    terms never seen in the collection are skipped (same -inf for every doc, so no effect on ranking)"""
    s = 0.0
    for w in q_terms:
        pc = st.p_c(w)
        if pc == 0.0:
            continue
        s += math.log(p_smooth(tf.get(w, 0), dl, pc, method, param))
    return s


def rank(scores: Dict[str, float], k: int):
    # ties broken by doc id so we never lean on input order
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
