# checks the eval kit gives the exact same numbers as harness/run_harness.py (practice suite),
# using a seed-sensitive dummy feedback fn so the noise draws actually matter
import os

import pytest

import submission.feedback as sub
from experiments import evalkit
from harness import run_harness

TOY = os.path.join(os.path.dirname(__file__), "..", "data", "toy")


def _seed_first(query, pseudo_relevant_doc_ids, candidate_doc_ids, k=10):
    # seed docs first in seed order, then rest of pool; depends on exact seed contents
    pool = set(candidate_doc_ids)
    order = [d for d in pseudo_relevant_doc_ids if d in pool]
    order += [d for d in candidate_doc_ids if d not in set(order)]
    return [(d, -float(i)) for i, d in enumerate(order[:k])]


@pytest.fixture
def seed_first(monkeypatch):
    monkeypatch.setattr(sub, "relevance_model_feedback", _seed_first)


def test_practice_suite_matches_harness(seed_first):
    h = run_harness.run(os.path.join(TOY, "corpus.jsonl"), os.path.join(TOY, "queries_dev.tsv"),
                        os.path.join(TOY, "qrels_dev.txt"), os.path.join(TOY, "candidates_dev.jsonl"))
    e = evalkit.evaluate(sub, evalkit.load_dataset(TOY), "practice")["summary"]
    assert e["ql_ndcg"] == pytest.approx(h["score_candidates_ndcg@10"])
    assert e["ql_map"] == pytest.approx(h["score_candidates_map@10"])
    assert e["clean_ndcg"] == pytest.approx(h["clean_feedback_ndcg@10"])
    for level, v in h["noisy_feedback_ndcg@10_by_level"].items():
        assert e["noisy"][f"uniform@{level}"] == pytest.approx(v)
    assert e["retention_practice"] == pytest.approx(h["practice_retention_ratio"])
    # sanity: the dummy really is seed sensitive, else this test proves nothing
    assert len(set(e["noisy"].values()) | {e["clean_ndcg"]}) > 1


def test_full_suite_seeds_stay_inside_pool_and_keep_length():
    data = evalkit.load_dataset(TOY)
    sub.prepare(data["corpus"])
    ranked = {q: [d for d, _ in sub.score_candidates(t, data["pools"][q], 100)] for q, t in data["queries"]}
    conds = evalkit.build_conditions(data, ranked, "full")
    assert "clean_pool" in conds and len(conds) == 2 + 3 * 4 * 3
    for name, seeds in conds.items():
        for qid, ids in seeds.items():
            assert len(ids) == len(conds["clean"][qid])
            assert set(ids) <= set(data["pools"][qid])
            assert len(set(ids)) == len(ids)


def test_hard_noise_only_injects_nonrelevant():
    rng = evalkit.random.Random(0)
    clean = ["a", "b", "c", "d"]
    ranked = clean + ["r1", "n1", "r2", "n2"]
    qr = {"r1": 1, "r2": 2, "n1": 0}
    out = evalkit.make_noisy("hard", clean, ranked, ranked, qr, 0.5, rng)
    assert set(out) - set(clean) <= {"n1", "n2"}


def test_paired_bootstrap_and_cv_basics():
    a = {str(i): 0.5 for i in range(30)}
    b = {str(i): 0.6 for i in range(30)}
    diff, p = evalkit.paired_bootstrap(a, b, n=500)
    assert diff == pytest.approx(0.1) and p < 0.01
    same, p2 = evalkit.paired_bootstrap(a, a, n=500)
    assert same == 0 and p2 == 1.0
    score, picks = evalkit.cv_select(["x", "y"], lambda c, qs: 1.0 if c == "y" else 0.0, list(a), k=5, repeats=2)
    assert score == 1.0 and picks == {"y": 10}
