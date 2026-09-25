"""
submission/feedback.py -- THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5,
"Submission Interface & Conformance Checking") -- do not rename them,
change their signatures, or move them out of this file.

    prepare(corpus_path: str) -> None
        Called once, before anything else. Load the corpus and build
        whatever collection-wide statistics your unigram LM and
        relevance model need (see submission/lm_utils.py::CollectionStats
        for why this is a cheap, one-time pass and does not require
        building an index). No build/load process split this time --
        grading is in-process, so it is fine to keep everything in
        module-level state here and read it in the two functions below.

    score_candidates(query: str, candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]
        Rerank the PROVIDED candidate pool -- typically the top-100 from
        an undisclosed reference retriever (assignment Section 6) -- using
        your own unigram query-likelihood LM (Section 3.1). You are never
        asked to rank the whole corpus; candidate_doc_ids is the entire
        universe of documents you need to consider for this call. Return
        up to k (doc_id, score) pairs, sorted by score descending, drawn
        ONLY from candidate_doc_ids. This is graded directly as Track A.

    relevance_model_feedback(query: str, pseudo_relevant_doc_ids: List[str], candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]
        RM1/RM2/RM3-based reranking (Section 3.2). Two DIFFERENT lists
        come in, doing two different jobs -- do not confuse them:
          - pseudo_relevant_doc_ids: the (possibly noise-perturbed) seed
            set used to ESTIMATE the relevance model. Supplied BY THE
            HARNESS -- sometimes exactly your own score_candidates()
            top-k', sometimes a version of that with a fraction swapped
            for off-topic documents (query drift stress-testing;
            assignment Section 7, Track C). You have no legitimate way to
            tell which, and must not try to detect it -- see
            docs/SUBMISSION_INTERFACE.md, "call-order independence".
          - candidate_doc_ids: the pool to RERANK using your estimated
            model -- typically the SAME candidate pool passed to
            score_candidates() for this query, unperturbed. Your
            returned doc_ids must come from this list, not from
            pseudo_relevant_doc_ids and not from outside either list.
        Return up to k (doc_id, score) pairs, sorted by score descending.

This file ships with a trivial, fully-working baseline: score_candidates()
does real Dirichlet-smoothed query-likelihood reranking of whatever
candidate pool it's given (so it is not literally a no-op), and
relevance_model_feedback() ignores pseudo_relevant_doc_ids entirely and
just reranks candidate_doc_ids the same way score_candidates() would. It
exercises the full interface correctly end-to-end from your first commit,
including the harness's noise-injection stress test (it will score
identically at every noise level, since it never looks at its seed input
-- which is itself a legitimate, if unambitious, point on the Track C
"retention" axis: you cannot drift if you never expand). Replace the
feedback logic; keep the same function shapes.
"""
from typing import List, Optional, Tuple

from submission.lm import Analyzer, Stats, ql_score, rank
from submission import rm

# ---- knobs (tuned on dev, see report) ----
SMOOTHING = "dirichlet"   # "dirichlet" or "jm"
DIRICHLET_MU = 250.0
JM_LAMBDA = 0.7
STOPWORDS = True
STEMMER = "porter"        # "none", "s" or "porter"

FB_MODEL = "rm3"          # "rm1", "rm2" or "rm3"
RM3_BASE = "rm1"          # which relevance model rm3 interpolates
FB_TERMS = 20             # expansion terms kept (0 = all)
FB_LAMBDA = 0.85          # rm3 weight on the original query (high on purpose, drift)
RM_EST_MU = 0.0           # doc model smoothing inside RM estimation (0 = max likelihood)

_STATS: Optional[Stats] = None


def prepare(corpus_path: str) -> None:
    global _STATS
    _STATS = Stats.from_jsonl(corpus_path, Analyzer(stop=STOPWORDS, stem=STEMMER))


def _need_stats() -> Stats:
    if _STATS is None:
        raise RuntimeError("call prepare(corpus_path) first")
    return _STATS


def _smooth_param() -> float:
    return DIRICHLET_MU if SMOOTHING == "dirichlet" else JM_LAMBDA


def score_candidates(query: str, candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]:
    st = _need_stats()
    q = st.an(query)
    param = _smooth_param()
    scores = {}
    for d in dict.fromkeys(candidate_doc_ids):  # dedup, keep order
        scores[d] = ql_score(q, st.tf(d), st.dl(d), st, SMOOTHING, param)
    return rank(scores, k)


def relevance_model_feedback(
    query: str,
    pseudo_relevant_doc_ids: List[str],
    candidate_doc_ids: List[str],
    k: int = 10,
) -> List[Tuple[str, float]]:
    st = _need_stats()
    q = st.an(query)
    rel = _feedback_model(q, pseudo_relevant_doc_ids, st)
    if not rel:
        # nothing usable in the seed -> just the query
        return score_candidates(query, candidate_doc_ids, k)
    model = rm.rm3(q, rel, FB_LAMBDA) if FB_MODEL == "rm3" else rel
    return rank(rm.ce_scores(model, candidate_doc_ids, st, _smooth_param(), SMOOTHING), k)


def _feedback_model(q, seed, st):
    """truncated relevance model from the seed (rm3 interpolation happens after this)"""
    kind = FB_MODEL if FB_MODEL != "rm3" else RM3_BASE
    if kind == "rm1":
        rel = rm.rm1(q, seed, st, _smooth_param(), RM_EST_MU, SMOOTHING)
    elif kind == "rm2":
        rel = rm.rm2(q, seed, st, _smooth_param(), RM_EST_MU, SMOOTHING)
    else:
        raise ValueError(kind)
    return rm.truncate(rel, FB_TERMS) if rel else rel
