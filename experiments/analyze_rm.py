# reads sweep_rm output: lambda curves, clean-vs-retention pareto frontier, bootstrap vs plain QL
# usage: python -m experiments.analyze_rm runs/rm_sweep/covid_bm25.json [--terms 50 --estmu 0]
import argparse
import json

from experiments.evalkit import mean, paired_bootstrap


def noisy_mean(s):
    return mean(s["noisy"].values())


def pareto(items):
    # items: [(name, clean, retention)] -> ones not dominated on both
    out = []
    for n, c, r in items:
        if not any(c2 >= c and r2 >= r and (c2 > c or r2 > r) for _, c2, r2 in items):
            out.append((n, c, r))
    return sorted(out, key=lambda x: -x[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--terms", type=int, default=None, help="only show lambda curves for this fb terms")
    ap.add_argument("--estmu", type=int, default=None)
    args = ap.parse_args()
    res = json.load(open(args.path))
    cfgs = res["configs"]
    any_s = next(iter(cfgs.values()))["summary"]
    ql = next(iter(cfgs.values()))["perq"]["ql"]
    print(f"# {res['pool']} / {res['suite']}   plain QL nDCG@10 {any_s['ql_ndcg']:.4f}\n")

    # group by everything except lambda
    groups = {}
    for name, v in cfgs.items():
        parts = name.split(",")
        lam = next(p for p in parts if p.startswith("lam="))
        key = tuple(p for p in parts if p != lam)
        groups.setdefault(key, []).append((float(lam.split("=")[1]), name, v["summary"]))
    has_pool = "clean_pool_ndcg" in any_s
    for key, rows in sorted(groups.items()):
        kv = dict(p.split("=") for p in key[1:])
        if (args.terms is not None and int(kv["terms"]) != args.terms) or \
                (args.estmu is not None and int(kv["estmu"]) != args.estmu):
            continue
        print("## " + " ".join(key))
        print("| λ | clean | " + ("pool-seed | " if has_pool else "") + "uniform@0.5 | hard@max | noisy mean | ret_practice | ret_all |")
        print("|---|---|" + ("---|" if has_pool else "") + "---|---|---|---|---|")
        for lam, name, s in sorted(rows):
            hard = [v for g, v in s["noisy"].items() if g.startswith("hard@")]
            print(f"| {lam:.1f} | {s['clean_ndcg']:.4f} | " + (f"{s['clean_pool_ndcg']:.4f} | " if has_pool else "")
                  + f"{s['noisy'].get('uniform@0.5', float('nan')):.4f} | {hard[-1] if hard else float('nan'):.4f} | "
                  f"{noisy_mean(s):.4f} | {s['retention_practice']:.4f} | {s['retention_all']:.4f} |")
        print()

    items = [(n, v["summary"]["clean_ndcg"], v["summary"]["retention_all"]) for n, v in cfgs.items()]
    print("## pareto frontier (clean nDCG vs retention_all)\n")
    print("| config | clean | ret_all | noisy mean | clean vs QL (p) | noisy-mean vs QL (p) |\n|---|---|---|---|---|---|")
    for n, c, r in pareto(items):
        pq = cfgs[n]["perq"]
        noisy_q = {q: mean(pq[k][q] for k in pq if "@" in k) for q in ql}
        d1, p1 = paired_bootstrap(ql, pq["clean"], n=2000)
        d2, p2 = paired_bootstrap(ql, noisy_q, n=2000)
        print(f"| {n} | {c:.4f} | {r:.4f} | {noisy_mean(cfgs[n]['summary']):.4f} | {d1:+.4f} ({p1:.3f}) | {d2:+.4f} ({p2:.3f}) |")


if __name__ == "__main__":
    main()
