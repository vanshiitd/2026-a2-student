# RM sweep: base model x lambda x fb terms x estimation smoothing, run through the eval kit noise suite
# prepare + QL rankings + seed conditions done once in the parent, then forked workers (copy on write)
# each take one (base, estmu, terms) group. inside a group the truncated relevance model depends only on
# (query, seed), so its memoized across all lambdas -> estimated once instead of once per lambda.
# usage: python -m experiments.sweep_rm --data data/full --pool data/full/pools/standin_bm25.jsonl --suite full --out runs/rm_sweep/covid_bm25.json
import argparse
import itertools
import json
import multiprocessing as mp
import os
import time

import submission.feedback as sub
from experiments.evalkit import evaluate, load_dataset, prepare_base

BASES = ["rm1", "rm2"]
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TERMS = [10, 20, 50, 100, 0]
EST_MUS = [0.0, 250.0]

_G = {}  # data / base / suite, set in parent before fork


def cfg_name(base, lam, terms, est_mu):
    return f"{base},lam={lam},terms={terms},estmu={int(est_mu)}"


def csv_list(cast):
    return lambda s: [cast(x) for x in s.split(",")]


def run_group(group):
    base, est_mu, terms, lambdas = group
    sub.FB_MODEL, sub.RM3_BASE, sub.RM_EST_MU, sub.FB_TERMS = "rm3", base, est_mu, terms
    real = sub._feedback_model
    memo = {}

    def cached(q, seed, st):
        key = (tuple(q), tuple(seed))
        if key not in memo:
            memo[key] = real(q, seed, st)
        return memo[key]

    sub._feedback_model = cached
    out = []
    try:
        for lam in lambdas:
            sub.FB_LAMBDA = lam
            res = evaluate(sub, _G["data"], _G["suite"], do_prepare=False, base=_G["base"])
            out.append((cfg_name(base, lam, terms, est_mu), res["summary"], res["perq"], res["max_call_s"]))
    finally:
        sub._feedback_model = real
    return out


def n_workers(requested, n_groups):
    avail = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, min(requested or avail, avail, n_groups))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--suite", default="full", choices=["full", "lite", "practice"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bases", type=csv_list(str), default=BASES)
    ap.add_argument("--lambdas", type=csv_list(float), default=LAMBDAS)
    ap.add_argument("--terms", type=csv_list(int), default=TERMS)
    ap.add_argument("--estmus", type=csv_list(float), default=EST_MUS)
    ap.add_argument("--workers", type=int, default=0, help="0 = all available cores")
    args = ap.parse_args()

    data = load_dataset(args.data, args.pool)
    t0 = time.time()
    sub.prepare(data["corpus"])
    _G.update(data=data, suite=args.suite, base=prepare_base(sub, data, args.suite))
    groups = [(b, e, t, args.lambdas) for b, e, t in itertools.product(args.bases, args.estmus, args.terms)]
    workers = n_workers(args.workers, len(groups))
    print(f"prepared in {time.time() - t0:.1f}s; {len(groups)} groups x {len(args.lambdas)} lambdas on {workers} workers",
          flush=True)

    results = {}
    if workers == 1:
        it = map(run_group, groups)
    else:
        pool = mp.get_context("fork").Pool(workers)
        it = pool.imap_unordered(run_group, groups)
    done = 0
    for rows in it:
        for name, summary, perq, max_call in rows:
            results[name] = {"summary": summary, "perq": perq, "max_call_s": max_call}
            done += 1
            print(f"[{done}/{len(groups) * len(args.lambdas)} {time.time() - t0:5.0f}s] {name:<36} "
                  f"clean {summary['clean_ndcg']:.4f}  ret_all {summary['retention_all']:.4f}  "
                  f"ret_hard {summary.get('retention_hard', 0):.4f}  max {max_call:.3f}s", flush=True)
    if workers > 1:
        pool.close()
        pool.join()

    # keep grid order in the output file
    order = [cfg_name(b, l, t, e) for b, l, t, e in itertools.product(args.bases, args.lambdas, args.terms, args.estmus)]
    out = {"pool": data["pool_name"], "suite": args.suite, "configs": {n: results[n] for n in order},
           "ql_map": _G["base"]["ql_map"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"saved {args.out}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
