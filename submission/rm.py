# relevance models (lavrenko & croft): RM1, RM2, RM3 + KL/cross-entropy reranking
# seed docs F only used for estimating P(w|R); ranking is done over the candidate pool
# method/param = the same smoothing score_candidates uses (dirichlet mu or jm lambda)
import heapq
import math
from collections import Counter
from typing import Dict, List

from submission.lm import Stats, p_smooth, ql_score

Model = Dict[str, float]


def _seed(seed_ids: List[str], st: Stats):
    # dedup, drop empty docs (nothing to learn from them)
    return [d for d in dict.fromkeys(seed_ids) if st.dl(d) > 0]


def doc_weights(q_terms: List[str], docs: List[str], st: Stats, param: float,
                method: str = "dirichlet") -> Dict[str, float]:
    """P(D|Q) over the seed set = normalised query likelihood (uniform P(D)), via log-sum-exp"""
    logs = {d: ql_score(q_terms, st.tf(d), st.dl(d), st, method, param) for d in docs}
    if not logs:
        return {}
    top = max(logs.values())
    ex = {d: math.exp(v - top) for d, v in logs.items()}
    z = sum(ex.values())
    return {d: v / z for d, v in ex.items()}


def _seed_vocab(docs, st):
    # union of seed terms in a fixed order (doc order, then term order) -> same result every run
    return list(dict.fromkeys(w for d in docs for w in st.tf(d)))


def rm1(q_terms: List[str], seed_ids: List[str], st: Stats, param: float, mu_est: float = 0.0,
        method: str = "dirichlet") -> Model:
    """P(w|R) ∝ sum_D P(w|D) P(D|Q)   (iid sampling). terms = vocab of the seed docs.
    P(w|D) is max likelihood (mu_est=0) or dirichlet(mu_est)"""
    docs = _seed(seed_ids, st)
    wts = doc_weights(q_terms, docs, st, param, method)
    model: Dict[str, float] = {}
    get = model.get
    if mu_est <= 0:
        # ML: only terms actually in D contribute
        for d in docs:
            tf, dl, wd = st.tf(d), st.dl(d), wts[d]
            for w, c in tf.items():
                model[w] = get(w, 0.0) + wd * (c / dl)
    else:
        vocab = _seed_vocab(docs, st)
        mpc = [mu_est * st.p_c(w) for w in vocab]
        for d in docs:
            tf, wd = st.tf(d), wts[d]
            den = st.dl(d) + mu_est
            tget = tf.get
            for w, m in zip(vocab, mpc):
                model[w] = get(w, 0.0) + wd * ((tget(w, 0) + m) / den)
    return normalise(model)


def rm2(q_terms: List[str], seed_ids: List[str], st: Stats, param: float, mu_est: float = 0.0,
        method: str = "dirichlet") -> Model:
    """P(w|R) ∝ P(w) prod_i sum_D P(q_i|D) P(D|w)   (conditional sampling), P(D|w) = P(w|D)P(D)/P(w).
    query-term probs P(q_i|D) use the ranking smoother so a missing term doesnt zero the product"""
    docs = _seed(seed_ids, st)
    if not docs:
        return {}
    q = [t for t in q_terms if st.p_c(t) > 0]
    n = len(docs)
    pd = 1.0 / n
    tfs = [st.tf(d) for d in docs]
    dls = [st.dl(d) for d in docs]
    pq = [[p_smooth(tf.get(t, 0), dl, st.p_c(t), method, param) for t in q] for tf, dl in zip(tfs, dls)]
    # postings: term -> [(doc index, P(w|D))]. with ML estimation a term sits in 1-2 seed docs usually,
    # zero entries add exactly 0.0 to every sum so skipping them changes nothing
    post: Dict[str, list] = {}
    if mu_est <= 0:
        for j, (tf, dl) in enumerate(zip(tfs, dls)):
            for w, c in tf.items():
                post.setdefault(w, []).append((j, c / dl))
    else:
        for w in _seed_vocab(docs, st):
            m = mu_est * st.p_c(w)
            post[w] = [(j, (tfs[j].get(w, 0) + m) / (dls[j] + mu_est)) for j in range(n)]
    logm: Dict[str, float] = {}
    qi = range(len(q))
    for w, lst in post.items():
        pw = sum(p for _, p in lst) * pd
        if pw <= 0:
            continue
        s = math.log(pw)
        for i in qi:
            inner = sum(pq[j][i] * p * pd / pw for j, p in lst)
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
    return normalise(dict(heapq.nsmallest(n, m.items(), key=lambda x: (-x[1], x[0]))))


def query_model(q_terms: List[str]) -> Model:
    return normalise(Counter(q_terms))


def rm3(q_terms: List[str], rel: Model, lam: float) -> Model:
    """lam * P(w|Q) + (1-lam) * P(w|R).  lam=1 -> plain query, lam=0 -> pure relevance model"""
    out = {w: lam * p for w, p in query_model(q_terms).items()}
    for w, p in rel.items():
        out[w] = out.get(w, 0.0) + (1 - lam) * p
    return out


def ce_scores(model: Model, doc_ids: List[str], st: Stats, param: float,
              method: str = "dirichlet") -> Dict[str, float]:
    """sum_w P(w|model) log P(w|D). rank equivalent to -KL(model || D).
    with model = query ML this is exactly QL / |Q|, so lam=1 reproduces score_candidates order.
    dirichlet is done sparsely:
      log((tf + mu pc)/(dl + mu)) = log(mu pc) + log1p(tf/(mu pc)) - log(dl + mu)
    first part is the same for every doc, so per doc we only touch terms it actually contains"""
    terms = [(w, p, st.p_c(w)) for w, p in model.items() if p > 0]
    terms = [(w, p, pc) for w, p, pc in terms if pc > 0]
    out = {}
    if method == "dirichlet":
        mu = param
        const = sum(p * math.log(mu * pc) for _, p, pc in terms)
        mass = sum(p for _, p, _ in terms)
        info = {w: (p, mu * pc) for w, p, pc in terms}
        for d in dict.fromkeys(doc_ids):
            tf = st.tf(d)
            s = const - mass * math.log(st.dl(d) + mu)
            if len(tf) < len(info):
                for w, c in tf.items():
                    x = info.get(w)
                    if x is not None:
                        s += x[0] * math.log1p(c / x[1])
            else:
                for w, (p, m) in info.items():
                    c = tf.get(w)
                    if c:
                        s += p * math.log1p(c / m)
            out[d] = s
        return out
    for d in dict.fromkeys(doc_ids):
        tf, dl = st.tf(d), st.dl(d)
        out[d] = sum(p * math.log(p_smooth(tf.get(w, 0), dl, pc, method, param)) for w, p, pc in terms)
    return out
