# eval kit: runs a submission module over a dataset + pool under many seed conditions,
# keeps per-query numbers so we can do CV / paired bootstrap on them later.
#
# suites
#   practice : exactly what harness/run_harness.py does (own QL top-10 seed, uniform 0.25/0.5, same rng seeds)
#   full     : practice-style clean seed + pool top-10 seed + 3 noise types x 4 levels x 3 seeds
#   lite     : same but levels 0.25/0.5 and one seed (for the bigger transfer sets)
# noise types (all draw replacements from the same pool, like the public recipe)
#   uniform  : public recipe (harness.noise_injection)
#   hard     : judged/unjudged NON-relevant docs that our own QL ranks highest (plausible but wrong)
#   adjacent : next-ranked docs after the seed by our own QL, relevance ignored
# hard uses qrels -> only a stress test, never goes anywhere near submission/
import os
import random
import time
from typing import Callable, Dict, List

from harness import noise_injection
from harness.candidates_io import read_candidates
from harness.metrics import average_precision, ndcg_at_k
from harness.run_harness import PRF_DEPTH, _validate_and_sort_results
from harness.trec_io import read_qrels, read_queries

NOISE_TYPES = ["uniform", "hard", "adjacent"]
FULL_LEVELS = [0.1, 0.25, 0.5, 0.75]
FULL_SEEDS = [0, 1, 2]
SUITE_GRID = {"full": (FULL_LEVELS, FULL_SEEDS), "lite": ([0.25, 0.5], [0])}
NEIGHBOUR_DEPTH = 20  # hard/adjacent pick from this many next-best docs


def load_dataset(data_dir: str, pool_path: str = None) -> Dict:
    pool_path = pool_path or os.path.join(data_dir, "candidates_dev.jsonl")
    pools = {q: [d for d, _ in lst] for q, lst in read_candidates(pool_path).items()}
    queries = [(q, t) for q, t in read_queries(os.path.join(data_dir, "queries_dev.tsv")) if q in pools]
    return {
        "corpus": os.path.join(data_dir, "corpus.jsonl"),
        "queries": queries,
        "qrels": read_qrels(os.path.join(data_dir, "qrels_dev.txt")),
        "pools": pools,
        "pool_name": os.path.basename(pool_path),
    }


def _swap(clean: List[str], cands: List[str], frac: float, rng: random.Random) -> List[str]:
    # same swap logic as the public recipe, just a different place to draw from
    n_rep = round(len(clean) * frac)
    if n_rep == 0 or not cands:
        return list(clean)
    out = list(clean)
    pos = rng.sample(range(len(clean)), n_rep)
    picks = rng.sample(cands, min(n_rep, len(cands)))
    for p, d in zip(pos, picks):
        out[p] = d
    return out


def make_noisy(kind, clean, pool, ranked, qrels_q, frac, rng):
    if kind == "uniform":
        return noise_injection.perturb_pseudo_relevant_set(clean, pool, frac, rng)
    seen = set(clean)
    rest = [d for d in ranked if d not in seen]
    if kind == "hard":
        cands = [d for d in rest if qrels_q.get(d, 0) <= 0][:NEIGHBOUR_DEPTH]
    elif kind == "adjacent":
        cands = rest[:NEIGHBOUR_DEPTH]
    else:
        raise ValueError(kind)
    return _swap(clean, cands, frac, rng)


def build_conditions(data: Dict, own_ranked: Dict[str, List[str]], suite: str) -> Dict[str, Dict[str, List[str]]]:
    """{condition_name: {qid: seed_doc_ids}}. 'clean' is always there."""
    conds: Dict[str, Dict[str, List[str]]] = {"clean": {}}
    for qid, _ in data["queries"]:
        pool = data["pools"][qid]
        clean = own_ranked[qid][:PRF_DEPTH]
        conds["clean"][qid] = clean
        if suite == "practice":
            # identical draws to run_harness.run()
            sd = noise_injection.stable_seed(qid, base_seed=0)
            for level, ids in noise_injection.sweep(clean, pool, seed=sd):
                if level > 0:
                    conds.setdefault(f"uniform@{level}", {})[qid] = ids
            continue
        conds.setdefault("clean_pool", {})[qid] = pool[:PRF_DEPTH]
        qr = data["qrels"].get(qid, {})
        levels, seed_list = SUITE_GRID[suite]
        for kind in NOISE_TYPES:
            for level in levels:
                for s in seed_list:
                    rng = random.Random(noise_injection.stable_seed(qid, base_seed=1000 * (s + 1)) + int(level * 100))
                    ids = make_noisy(kind, clean, pool, own_ranked[qid], qr, level, rng)
                    conds.setdefault(f"{kind}@{level}#{s}", {})[qid] = ids
    return conds


def evaluate(sub, data: Dict, suite: str = "practice", do_prepare: bool = True) -> Dict:
    """sub = module with prepare/score_candidates/relevance_model_feedback.
    returns {'perq': {name: {qid: ndcg}}, 'ql_map': {qid: ap}, 'summary': {...}, 'max_call_s': float}"""
    if do_prepare:
        sub.prepare(data["corpus"])
    qrels = data["qrels"]
    perq: Dict[str, Dict[str, float]] = {"ql": {}}
    ql_map: Dict[str, float] = {}
    own_ranked: Dict[str, List[str]] = {}
    max_call = 0.0

    def timed(fn: Callable, *a):
        nonlocal max_call
        t0 = time.perf_counter()
        r = fn(*a)
        max_call = max(max_call, time.perf_counter() - t0)
        return r

    for qid, text in data["queries"]:
        pool = data["pools"][qid]
        valid = set(pool)
        full = _validate_and_sort_results(timed(sub.score_candidates, text, pool, len(pool)), qid, len(pool), valid)
        own_ranked[qid] = [d for d, _ in full]
        top = own_ranked[qid][:10]
        perq["ql"][qid] = ndcg_at_k(top, qrels.get(qid, {}))
        ql_map[qid] = average_precision(top, qrels.get(qid, {}))

    conds = build_conditions(data, own_ranked, suite)
    text_of = dict(data["queries"])
    for name, seeds in conds.items():
        perq[name] = {}
        for qid, seed_ids in seeds.items():
            pool = data["pools"][qid]
            res = _validate_and_sort_results(
                timed(sub.relevance_model_feedback, text_of[qid], seed_ids, pool, 10), qid, 10, set(pool))
            perq[name][qid] = ndcg_at_k([d for d, _ in res], qrels.get(qid, {}))

    return {"perq": perq, "ql_map": ql_map, "summary": summarize(perq, ql_map), "max_call_s": max_call,
            "suite": suite, "pool": data["pool_name"]}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def retention(clean: float, noisy: List[float]) -> float:
    # same formula as harness/leaderboard.retention_ratio
    if clean <= 0 or not noisy:
        return 0.0
    return max(0.0, min(1.0, mean(noisy) / clean))


def summarize(perq: Dict[str, Dict[str, float]], ql_map: Dict[str, float], qids: List[str] = None) -> Dict:
    """aggregate numbers, optionally on a subset of qids (for CV folds)"""
    qids = qids if qids is not None else list(perq["ql"].keys())
    agg = {name: mean(v[q] for q in qids) for name, v in perq.items()}
    out = {"ql_ndcg": agg["ql"], "ql_map": mean(ql_map[q] for q in qids), "clean_ndcg": agg["clean"]}
    if "clean_pool" in agg:
        out["clean_pool_ndcg"] = agg["clean_pool"]
    # group noisy conditions by type@level (averaging seeds)
    groups: Dict[str, List[float]] = {}
    for name, v in agg.items():
        if "@" in name:
            groups.setdefault(name.split("#")[0], []).append(v)
    levels = {g: mean(vs) for g, vs in sorted(groups.items())}
    out["noisy"] = levels
    practice = [levels[g] for g in ("uniform@0.25", "uniform@0.5") if g in levels]
    out["retention_practice"] = retention(agg["clean"], practice)
    for kind in NOISE_TYPES:
        ks = [v for g, v in levels.items() if g.startswith(kind + "@")]
        if ks:
            out[f"retention_{kind}"] = retention(agg["clean"], ks)
    out["retention_all"] = retention(agg["clean"], list(levels.values()))
    return out


# ---- stats ----

def paired_bootstrap(a: Dict[str, float], b: Dict[str, float], n: int = 10000, seed: int = 0):
    """two-sided paired bootstrap on per-query diffs (b - a). returns (mean_diff, p)"""
    qids = sorted(set(a) & set(b))
    diffs = [b[q] - a[q] for q in qids]
    obs = mean(diffs)
    centred = [d - obs for d in diffs]
    rng = random.Random(seed)
    m = len(centred)
    hits = 0
    for _ in range(n):
        s = mean(centred[rng.randrange(m)] for _ in range(m))
        if abs(s) >= abs(obs):
            hits += 1
    return obs, (hits + 1) / (n + 1)


def cv_select(configs: List[str], score_fn: Callable[[str, List[str]], float], qids: List[str],
              k: int = 5, repeats: int = 10, seed: int = 0):
    """honest estimate of 'pick best config on train folds, report on test fold'.
    score_fn(cfg, qids) -> float (higher better), so it can be ndcg, retention or any combo.
    returns (mean test score, {cfg: times picked})"""
    rng = random.Random(seed)
    test_scores, picks = [], {}
    for _ in range(repeats):
        order = list(qids)
        rng.shuffle(order)
        folds = [order[i::k] for i in range(k)]
        for i in range(k):
            test = folds[i]
            train = [q for j, f in enumerate(folds) if j != i for q in f]
            best = max(configs, key=lambda c: score_fn(c, train))
            picks[best] = picks.get(best, 0) + 1
            test_scores.append(score_fn(best, test))
    return mean(test_scores), picks
