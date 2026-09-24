# RM sweep: base model x lambda x fb terms x estimation smoothing, run through the eval kit noise suite
# prepares once, then just flips knobs in submission.feedback. keeps per query numbers.
# usage: python -m experiments.sweep_rm --data data/full --pool data/full/pools/standin_bm25.jsonl --suite full --out runs/rm_sweep/covid_bm25.json
import argparse
import itertools
import json
import os
import time

import submission.feedback as sub
from experiments.evalkit import evaluate, load_dataset

BASES = ["rm1", "rm2"]
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TERMS = [10, 20, 50, 100, 0]
EST_MUS = [0.0, 250.0]


def cfg_name(base, lam, terms, est_mu):
    return f"{base},lam={lam},terms={terms},estmu={int(est_mu)}"


def csv_list(cast):
    return lambda s: [cast(x) for x in s.split(",")]


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
    args = ap.parse_args()

    data = load_dataset(args.data, args.pool)
    sub.prepare(data["corpus"])
    sub.FB_MODEL = "rm3"
    out = {"pool": data["pool_name"], "suite": args.suite, "configs": {}}
    t0 = time.time()
    grid = list(itertools.product(args.bases, args.lambdas, args.terms, args.estmus))
    for i, (base, lam, terms, est_mu) in enumerate(grid):
        sub.RM3_BASE, sub.FB_LAMBDA, sub.FB_TERMS, sub.RM_EST_MU = base, lam, terms, est_mu
        name = cfg_name(base, lam, terms, est_mu)
        res = evaluate(sub, data, args.suite, do_prepare=False)
        out["configs"][name] = {
            "summary": res["summary"], "perq": res["perq"], "max_call_s": res["max_call_s"]}
        s = res["summary"]
        print(f"[{i + 1}/{len(grid)} {time.time() - t0:5.0f}s] {name:<46} "
              f"clean {s['clean_ndcg']:.4f}  ret_all {s['retention_all']:.4f}  "
              f"ret_hard {s.get('retention_hard', 0):.4f}  max {res['max_call_s']:.3f}s", flush=True)
    out["ql_map"] = res["ql_map"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
