# hand checkable proximity bonus tests (tao & zhai MinDist, shifted so no-pair docs get exactly 0)
import json
import math

import pytest

import submission.feedback as fb
from submission.prox import min_dist


def test_min_dist_by_hand():
    q = {"a", "b"}
    assert min_dist(q, ["a", "x", "x", "b"]) == 3
    assert min_dist(q, ["b", "a"]) == 1
    assert min_dist(q, ["a", "x", "b", "x", "x", "a", "b"]) == 1       # closest pair wins
    assert min_dist(q, ["a", "x", "a"]) is None                        # same term twice doesnt count
    assert min_dist(q, ["x", "y"]) is None


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    # d1 = "cat dog fish" (cat and fish 2 apart), d2 = "cat x x x fish" (4 apart), d3 = "cat cat" (no pair)
    p = tmp_path / "corpus.jsonl"
    with open(p, "w") as f:
        for i, t in enumerate(["cat dog fish", "cat x x x fish", "cat cat"], 1):
            f.write(json.dumps({"doc_id": f"d{i}", "text": t}) + "\n")
    for name, val in [("STOPWORDS", False), ("STEMMER", "none"), ("SMOOTHING", "dirichlet"),
                      ("DIRICHLET_MU", 2.0), ("PROX_WEIGHT", 2.0), ("PROX_ALPHA", 0.3), ("FB_LAMBDA", 1.0)]:
        monkeypatch.setattr(fb, name, val)
    fb.prepare(str(p))
    return fb


def test_score_is_ql_plus_bonus_by_hand(tiny):
    with_prox = dict(tiny.score_candidates("cat fish", ["d1", "d2", "d3"], 10))
    tiny.PROX_WEIGHT = 0.0
    plain = dict(tiny.score_candidates("cat fish", ["d1", "d2", "d3"], 10))
    assert with_prox["d1"] - plain["d1"] == pytest.approx(2.0 * math.log(1 + math.exp(-2) / 0.3))
    assert with_prox["d2"] - plain["d2"] == pytest.approx(2.0 * math.log(1 + math.exp(-4) / 0.3))
    assert with_prox["d3"] == pytest.approx(plain["d3"])  # no pair -> exactly plain QL


def test_one_word_query_is_plain_ql(tiny):
    a = tiny.score_candidates("cat", ["d1", "d2", "d3"], 10)
    tiny.PROX_WEIGHT = 0.0
    assert a == tiny.score_candidates("cat", ["d1", "d2", "d3"], 10)


def test_feedback_at_lambda_one_matches_score_candidates_with_prox(tiny):
    for seed in (["d1"], ["d2", "d3"], []):
        a = [d for d, _ in tiny.score_candidates("cat fish", ["d1", "d2", "d3"], 10)]
        b = [d for d, _ in tiny.relevance_model_feedback("cat fish", seed, ["d1", "d2", "d3"], 10)]
        assert a == b
