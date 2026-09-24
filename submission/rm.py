# relevance models (lavrenko & croft): RM1, RM2, RM3 + KL/cross-entropy reranking
# seed docs F only used for estimating P(w|R); ranking is done over the candidate pool
import math
from collections import Counter
from typing import Dict, List

from submission.lm import Stats, ql_score

Model = Dict[str, float]


def _p_w_d(w: str, tf: Counter, dl: int, st: Stats, mu: float) -> float:
    # doc model used inside RM estimation. mu=0 -> plain max likelihood
    if mu <= 0:
        return tf.get(w, 0) / dl if dl else 0.0
    return (tf.get(w, 0) + mu * st.p_c(w)) / (dl + mu)


def _seed(seed_ids: List[str], st: Stats):
    # dedup, drop empty docs (nothing to learn from them)
    out = []
    for d in dict.fromkeys(seed_ids):
        if st.dl(d) > 0:
            out.append(d)
    return out


def doc_weights(q_terms: List[str], docs: List[str], st: Stats, mu: float) -> Dict[str, float]:
    """P(D|Q) over the seed set = normalised query likelihood (uniform P(D)), via log-sum-exp"""
    logs = {d: ql_score(q_terms, st.tf(d), st.dl(d), st, "dirichlet", mu) for d in docs}
    if not logs:
        return {}
    top = max(logs.values())
    ex = {d: math.exp(v - top) for d, v in logs.items()}
    z = sum(ex.values())
    return {d: v / z for d, v in ex.items()}


def rm1(q_terms: List[str], seed_ids: List[str], st: Stats, mu_rank: float, mu_est: float = 0.0) -> Model:
    """P(w|R) ∝ sum_D P(w|D) P(D|Q)   (iid sampling). terms = vocab of the seed docs"""
    docs = _seed(seed_ids, st)
    wts = doc_weights(q_terms, docs, st, mu_rank)
    model: Dict[str, float] = {}
    vocab = set()
    for d in docs:
        vocab.update(st.tf(d))
    for d in docs:
        tf, dl, wd = st.tf(d), st.dl(d), wts[d]
        for w in vocab if mu_est > 0 else tf:
            model[w] = model.get(w, 0.0) + wd * _p_w_d(w, tf, dl, st, mu_est)
    return normalise(model)


def rm2(q_terms: List[str], seed_ids: List[str], st: Stats, mu_rank: float, mu_est: float = 0.0) -> Model:
    """P(w|R) ∝ P(w) prod_i sum_D P(q_i|D) P(D|w)   (conditional sampling), P(D|w) = P(w|D)P(D)/P(w).
    query-term probs P(q_i|D) use the ranking smoother so a missing term doesnt zero the product"""
    docs = _seed(seed_ids, st)
    if not docs:
        return {}
    q = [t for t in q_terms if st.p_c(t) > 0]
    vocab = set()
    for d in docs:
        vocab.update(st.tf(d))
    pd = 1.0 / len(docs)
    pq = {d: [_p_w_d(t, st.tf(d), st.dl(d), st, mu_rank) for t in q] for d in docs}
    logm: Dict[str, float] = {}
    for w in vocab:
        pwd = {d: _p_w_d(w, st.tf(d), st.dl(d), st, mu_est) for d in docs}
        pw = sum(pwd.values()) * pd
        if pw <= 0:
            continue
        s = math.log(pw)
        for i in range(len(q)):
            inner = sum(pq[d][i] * pwd[d] * pd / pw for d in docs)
            s += math.log(inner) if inner > 0 else -1e9
        logm[w] = s
    if not logm:
        return {}
    top = max(logm.values())
    return normalise({w: math.exp(v - top) for w, v in logm.items()})


def normalise(m: Model) -> Model:
    z = sum(m.values())
    return {w: v / z for w, v in m.items()} if z > 0 else {}


def truncate(m: Model, n: int) -> Model:
    """keep top n terms (ties by term), renormalise. n<=0 keeps all"""
    if n <= 0 or len(m) <= n:
        return normalise(m)
    return normalise(dict(sorted(m.items(), key=lambda x: (-x[1], x[0]))[:n]))


def query_model(q_terms: List[str]) -> Model:
    return normalise(Counter(q_terms))


def rm3(q_terms: List[str], rel: Model, lam: float) -> Model:
    """lam * P(w|Q) + (1-lam) * P(w|R).  lam=1 -> plain query, lam=0 -> pure relevance model"""
    out = {w: lam * p for w, p in query_model(q_terms).items()}
    for w, p in rel.items():
        out[w] = out.get(w, 0.0) + (1 - lam) * p
    return out


def ce_scores(model: Model, doc_ids: List[str], st: Stats, mu: float) -> Dict[str, float]:
    """sum_w P(w|model) log P(w|D), dirichlet P(w|D). rank equivalent to -KL(model || D).
    with model = query ML this is exactly QL / |Q|, so lam=1 reproduces score_candidates order"""
    terms = [(w, p, st.p_c(w)) for w, p in model.items() if p > 0 and st.p_c(w) > 0]
    out = {}
    for d in dict.fromkeys(doc_ids):
        tf, dl = st.tf(d), st.dl(d)
        s = 0.0
        for w, p, pc in terms:
            s += p * math.log((tf.get(w, 0) + mu * pc) / (dl + mu))
        out[d] = s
    return out
