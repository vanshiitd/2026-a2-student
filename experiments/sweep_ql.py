# QL-only sweep: analyzer (stopwords x stemmer) x smoothing (dirichlet mu / jm lambda)
# prepares once per analyzer, keeps per query ndcg/ap so we can do cv + bootstrap after
# usage: python -m experiments.sweep_ql --data data/full --pool data/full/pools/standin_bm25.jsonl [--pool ...] --out runs/ql_sweep/full.json
import argparse
import json
import os

import submission.feedback as sub
from experiments.evalkit import cv_select, load_dataset, mean, paired_bootstrap
from harness.metrics import average_precision, ndcg_at_k

MUS = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]
LAMS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ANALYZERS = [(False, "none"), (True, "none"), (False, "s"), (True, "s")]


def cfg_name(stop, stem, method, param):
    return f"stop={int(stop)},stem={stem},{method}={param}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pool", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mus", default=None, help="comma list, overrides default grid")
    ap.add_argument("--lams", default=None)
    ap.add_argument("--analyzers", default=None, help="e.g. 1s,1none  (stop flag + stemmer)")
    args = ap.parse_args()
    global MUS, LAMS, ANALYZERS
    if args.mus:
        MUS = [int(x) for x in args.mus.split(",")]
    if args.lams:
        LAMS = [float(x) for x in args.lams.split(",")]
    if args.analyzers:
        ANALYZERS = [(a[0] == "1", a[1:]) for a in args.analyzers.split(",")]

    datasets = [load_dataset(args.data, p) for p in args.pool]
    res = {d["pool_name"]: {} for d in datasets}  # pool -> cfg -> {"ndcg": {qid: v}, "ap": {...}}

    for stop, stem in ANALYZERS:
        sub.STOPWORDS, sub.STEMMER = stop, stem
        sub.prepare(datasets[0]["corpus"])
        for method, params in (("dirichlet", MUS), ("jm", LAMS)):
            sub.SMOOTHING = method
            for p in params:
                if method == "dirichlet":
                    sub.DIRICHLET_MU = float(p)
                else:
                    sub.JM_LAMBDA = float(p)
                name = cfg_name(stop, stem, method, p)
                for d in datasets:
                    nd, av = {}, {}
                    for qid, text in d["queries"]:
                        top = [x for x, _ in sub.score_candidates(text, d["pools"][qid], 10)]
                        qr = d["qrels"].get(qid, {})
                        nd[qid], av[qid] = ndcg_at_k(top, qr), average_precision(top, qr)
                    res[d["pool_name"]][name] = {"ndcg": nd, "ap": av}
        print(f"done analyzer stop={stop} stem={stem}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)

    base = cfg_name(False, "none", "dirichlet", 1500)
    for d in datasets:
        if base not in res[d["pool_name"]]:
            base = next(iter(res[d["pool_name"]]))
        pn, qids = d["pool_name"], [q for q, _ in d["queries"]]
        r = res[pn]
        pool_order = {q: ndcg_at_k(d["pools"][q][:10], d["qrels"].get(q, {})) for q in qids}
        print(f"\n== {pn}  (pool order nDCG@10 {mean(pool_order.values()):.4f}) ==")
        # grid table: rows = analyzer, cols = params
        for method, params in (("dirichlet", MUS), ("jm", LAMS)):
            print(f"{method:<18}" + "".join(f"{p:>8}" for p in params))
            for stop, stem in ANALYZERS:
                row = [mean(r[cfg_name(stop, stem, method, p)]["ndcg"].values()) for p in params]
                print(f"stop={int(stop)} stem={stem:<5}  " + "".join(f"{v:8.4f}" for v in row))
        cfgs = list(r)
        best = max(cfgs, key=lambda c: mean(r[c]["ndcg"].values()))
        cv, picks = cv_select(cfgs, lambda c, qs: mean(r[c]["ndcg"][q] for q in qs), qids)
        top_picks = sorted(picks.items(), key=lambda x: -x[1])[:3]
        print(f"best in-sample: {best}  nDCG {mean(r[best]['ndcg'].values()):.4f}  MAP {mean(r[best]['ap'].values()):.4f}")
        print(f"5-fold CV x10 (select on train, score test): {cv:.4f}   picks {top_picks}")
        diff, p = paired_bootstrap(r[base]["ndcg"], r[best]["ndcg"])
        print(f"best vs starter ({base}): {diff:+.4f}  p={p:.4f}")
        diff, p = paired_bootstrap(pool_order, r[best]["ndcg"])
        print(f"best vs pool order: {diff:+.4f}  p={p:.4f}")


if __name__ == "__main__":
    main()
