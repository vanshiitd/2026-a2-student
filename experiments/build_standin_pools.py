# stand-in candidate pools till the official candidates_dev.jsonl comes out
# staff said on piazza we can make our own with write_candidates(), so thats what this does.
#   bm25 : plain BM25 (k1=0.9, b=0.4), top-100  -> "unremarkable" first pass
#   weak : raw tf*idf, no length norm, top-100 -> worse + biased to long docs, to test pool sensitivity
# only query terms are counted so one pass over the corpus is enough, no index needed
# usage: python -m experiments.build_standin_pools --data data/full --out data/full/pools
import argparse
import math
import os
from collections import Counter

from harness.candidates_io import write_candidates
from harness.trec_io import read_queries
from submission.corpus_utils import load_corpus
from submission.lm_utils import tokenize

K1, B = 0.9, 0.4
DEPTH = 100


def top_k(scores, k):
    # tie break on doc_id so output is deterministic
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    queries = read_queries(os.path.join(args.data, "queries_dev.tsv"))
    q_terms = {qid: tokenize(t) for qid, t in queries}
    vocab = set(w for ts in q_terms.values() for w in ts)

    corpus = load_corpus(os.path.join(args.data, "corpus.jsonl"))
    n_docs = len(corpus)
    doc_len = {}
    tf = {}  # doc_id -> {term: count}, only query terms
    df = Counter()
    for doc_id, text in corpus:
        toks = tokenize(text)
        doc_len[doc_id] = len(toks)
        c = {w: n for w, n in Counter(toks).items() if w in vocab}
        if c:
            tf[doc_id] = c
            df.update(c.keys())
    avgdl = sum(doc_len.values()) / n_docs

    bm25_pools, weak_pools = {}, {}
    for qid, terms in q_terms.items():
        qtf = Counter(terms)
        bm, wk = {}, {}
        for doc_id, c in tf.items():
            s_bm = s_wk = 0.0
            for w, qn in qtf.items():
                n = c.get(w)
                if not n:
                    continue
                idf_bm = math.log(1 + (n_docs - df[w] + 0.5) / (df[w] + 0.5))
                norm = K1 * (1 - B + B * doc_len[doc_id] / avgdl)
                s_bm += qn * idf_bm * n * (K1 + 1) / (n + norm)
                s_wk += qn * n * math.log(n_docs / df[w])
            bm[doc_id] = s_bm
            wk[doc_id] = s_wk
        bm25_pools[qid] = top_k(bm, DEPTH)
        weak_pools[qid] = top_k(wk, DEPTH)

    os.makedirs(args.out, exist_ok=True)
    write_candidates(os.path.join(args.out, "standin_bm25.jsonl"), bm25_pools)
    write_candidates(os.path.join(args.out, "standin_weak.jsonl"), weak_pools)
    print(f"wrote pools for {len(q_terms)} queries to {args.out}")


if __name__ == "__main__":
    main()
