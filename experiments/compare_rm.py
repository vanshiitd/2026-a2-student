# cross dataset view of RM sweeps: only configs run on every dataset, gains vs plain QL on each
# usage: python -m experiments.compare_rm covid=runs/rm_sweep/a.json,runs/rm_sweep/b.json scifact=... [--top 25]
import argparse
import json

from experiments.evalkit import mean


def load(paths):
    cfgs = {}
    for p in paths.split(","):
        for n, v in json.load(open(p))["configs"].items():
            cfgs[n] = v["summary"]
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="name=path1,path2")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-ret", type=float, default=0.0, help="drop configs with retention_all below this anywhere")
    ap.add_argument("--filter", default="", help="substring every shown config must contain")
    args = ap.parse_args()

    data = {}
    for s in args.sets:
        name, paths = s.split("=", 1)
        data[name] = load(paths)
    common = set.intersection(*(set(d) for d in data.values()))
    rows = []
    for c in common:
        if args.filter and args.filter not in c:
            continue
        per = {}
        for n, d in data.items():
            s = d[c]
            per[n] = (s["clean_ndcg"] - s["ql_ndcg"], mean(s["noisy"].values()) - s["ql_ndcg"], s["retention_all"])
        if min(r for _, _, r in per.values()) < args.min_ret:
            continue
        rows.append((c, per))
    # robust first: worst-case noisy gain, then mean clean gain
    rows.sort(key=lambda x: (-min(v[1] for v in x[1].values()), -mean(v[0] for v in x[1].values())))
    names = list(data)
    print(f"{len(common)} common configs; showing top {args.top} by worst-case noisy gain vs QL\n")
    print("| config | " + " | ".join(f"{n} Δclean / Δnoisy / ret" for n in names) + " | worst Δnoisy | mean Δclean |")
    print("|---|" + "---|" * (len(names) + 2))
    for c, per in rows[:args.top]:
        cells = [f"{per[n][0]:+.4f} / {per[n][1]:+.4f} / {per[n][2]:.3f}" for n in names]
        print(f"| {c} | " + " | ".join(cells) + f" | {min(v[1] for v in per.values()):+.4f} | {mean(v[0] for v in per.values()):+.4f} |")


if __name__ == "__main__":
    main()
