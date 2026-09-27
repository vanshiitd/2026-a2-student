# score_candidates (track A) sweep over the knobs that survived: proximity weight and dirichlet mu,
# driven through the real submission code so what we measure is what ships.
# the ideas that lost (porter2, sdm bigrams, query df-stopping) were deleted, their numbers are in the findings log.
# usage: python -m experiments.sweep_qlx --data data/full --pool data/full/candidates_dev.jsonl --out runs/qlx/official.json
import argparse
import json
import os

import submission.feedback as sub
from experiments.evalkit import load_dataset, mean, paired_bootstrap
from harness.metrics import average_precision, ndcg_at_k

PROX = [0.0, 0.5, 1.0, 2.0, 3.0]
MUS = [250.0, 350.0, 500.0, 750.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = load_dataset(args.data, args.pool)
    sub.prepare(d["corpus"])
    base = (sub.PROX_WEIGHT, sub.DIRICHLET_MU)
    res = {}
    for w in PROX:
        for mu in MUS:
            sub.PROX_WEIGHT, sub.DIRICHLET_MU = w, mu
            nd, av = {}, {}
            for qid, text in d["queries"]:
                top = [x for x, _ in sub.score_candidates(text, d["pools"][qid], 10)]
                qr = d["qrels"].get(qid, {})
                nd[qid], av[qid] = ndcg_at_k(top, qr), average_precision(top, qr)
            res[f"prox={w},mu={int(mu)}"] = {"ndcg": nd, "ap": av}
    sub.PROX_WEIGHT, sub.DIRICHLET_MU = base
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"))
    ref = res[f"prox={base[0]},mu={int(base[1])}"]
    print(f"[{d['pool_name']} @ {args.data}] shipped (prox={base[0]}, mu={int(base[1])}) nDCG@10 {mean(ref['ndcg'].values()):.4f}")
    for k, r in res.items():
        dn, p = paired_bootstrap(ref["ndcg"], r["ndcg"], n=2000)
        print(f"  {k:<18} nDCG {mean(r['ndcg'].values()):.4f} ({dn:+.4f}, p={p:.3f})  MAP {mean(r['ap'].values()):.4f}")


if __name__ == "__main__":
    main()
