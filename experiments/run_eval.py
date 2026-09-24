# run the eval kit on the current submission and dump per-query json
# usage:
#   python -m experiments.run_eval --data data/full --pool data/full/pools/standin_bm25.jsonl --suite full --tag base
#   --set NAME=VALUE overrides a module level constant in submission.feedback (e.g. --set DIRICHLET_MU=1000)
import argparse
import json
import os

import submission.feedback as sub
from experiments.evalkit import evaluate, load_dataset


def parse_val(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def print_summary(s, max_call):
    print(f"QL        nDCG@10 {s['ql_ndcg']:.4f}  MAP@10 {s['ql_map']:.4f}")
    print(f"FB clean  nDCG@10 {s['clean_ndcg']:.4f}" +
          (f"   (pool-top10 seed: {s['clean_pool_ndcg']:.4f})" if "clean_pool_ndcg" in s else ""))
    for g, v in s["noisy"].items():
        print(f"  {g:<16} {v:.4f}")
    print("retention  " + "  ".join(f"{k[10:]}={v:.4f}" for k, v in s.items() if k.startswith("retention_")))
    print(f"slowest call {max_call:.3f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--suite", default="practice", choices=["practice", "full"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    for kv in args.set:
        name, val = kv.split("=", 1)
        if not hasattr(sub, name):
            raise SystemExit(f"submission.feedback has no {name}")
        setattr(sub, name, parse_val(val))

    data = load_dataset(args.data, args.pool)
    res = evaluate(sub, data, args.suite)
    print(f"[{data['pool_name']} / {args.suite} / {len(data['queries'])} queries]")
    print_summary(res["summary"], res["max_call_s"])

    if args.tag:
        res["overrides"] = args.set
        out = os.path.join("runs", "eval", f"{args.tag}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
