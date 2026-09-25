# hand checkable RM1 / RM2 / RM3 tests, same 2 doc corpus as test_lm.py
#   d1 = "cat cat dog" (len 3)   d2 = "dog fish" (len 2)   p(cat)=.4 p(dog)=.4 p(fish)=.2
# ranking smoother dirichlet mu=2, RM estimation uses max likelihood P(w|D)
import json
import math
import os

import pytest

import submission.feedback as fb
from harness.candidates_io import read_candidates
from harness.trec_io import read_queries
from submission import rm

TOY = os.path.join(os.path.dirname(__file__), "..", "data", "toy")


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    p = tmp_path / "corpus.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"doc_id": "d1", "text": "cat cat dog"}) + "\n")
        f.write(json.dumps({"doc_id": "d2", "text": "dog fish"}) + "\n")
    for name, val in [("STOPWORDS", False), ("STEMMER", "none"), ("SMOOTHING", "dirichlet"),
                      ("DIRICHLET_MU", 2.0), ("RM_EST_MU", 0.0), ("FB_TERMS", 0)]:
        monkeypatch.setattr(fb, name, val)
    fb.prepare(str(p))
    return fb._STATS


def test_doc_weights_are_normalised_query_likelihood(tiny):
    # "cat": P(Q|d1) = (2+.8)/5 = .56, P(Q|d2) = .8/4 = .2
    w = rm.doc_weights(["cat"], ["d1", "d2"], tiny, 2.0)
    assert w["d1"] == pytest.approx(.56 / .76)
    assert w["d2"] == pytest.approx(.2 / .76)


def test_rm1_by_hand(tiny):
    # query "cat fish": P(Q|d1) = .56*.08, P(Q|d2) = .2*.35
    a, b = .56 * .08, .2 * .35
    w1, w2 = a / (a + b), b / (a + b)
    m = rm.rm1(["cat", "fish"], ["d1", "d2"], tiny, 2.0)
    assert m["cat"] == pytest.approx(w1 * 2 / 3)
    assert m["dog"] == pytest.approx(w1 / 3 + w2 * .5)
    assert m["fish"] == pytest.approx(w2 * .5)
    assert sum(m.values()) == pytest.approx(1.0)


def test_rm2_by_hand(tiny):
    # P(w) prod_i sum_D P(q_i|D) P(w|D) P(D) / P(w), P(D)=.5
    # q probs: d1 cat .56 fish .08 ; d2 cat .2 fish .35
    def score(pw1, pw2):
        pw = .5 * (pw1 + pw2)
        cat = (.56 * pw1 * .5 + .2 * pw2 * .5) / pw
        fish = (.08 * pw1 * .5 + .35 * pw2 * .5) / pw
        return pw * cat * fish
    raw = {"cat": score(2 / 3, 0), "dog": score(1 / 3, .5), "fish": score(0, .5)}
    z = sum(raw.values())
    m = rm.rm2(["cat", "fish"], ["d1", "d2"], tiny, 2.0)
    for w in raw:
        assert m[w] == pytest.approx(raw[w] / z)
    # rm2 differs from rm1 here (dog gets more, it co-occurs with both query words)
    assert m["dog"] > rm.rm1(["cat", "fish"], ["d1", "d2"], tiny, 2.0)["dog"]


def test_rm1_equals_rm2_for_one_term_query(tiny):
    m1 = rm.rm1(["cat"], ["d1", "d2"], tiny, 2.0)
    m2 = rm.rm2(["cat"], ["d1", "d2"], tiny, 2.0)
    assert m1 == pytest.approx(m2)


def test_rm3_mix_and_ce_ranking_by_hand(tiny, monkeypatch):
    monkeypatch.setattr(fb, "FB_MODEL", "rm3")
    monkeypatch.setattr(fb, "RM3_BASE", "rm1")
    monkeypatch.setattr(fb, "FB_LAMBDA", 0.5)
    w1, w2 = .56 / .76, .2 / .76
    r = {"cat": w1 * 2 / 3, "dog": w1 / 3 + w2 * .5, "fish": w2 * .5}
    mix = {"cat": .5 + .5 * r["cat"], "dog": .5 * r["dog"], "fish": .5 * r["fish"]}
    assert rm.rm3(["cat"], r, 0.5) == pytest.approx(mix)
    # dirichlet mu=2 doc models: d1 cat .56 dog .36 fish .08 ; d2 cat .2 dog .45 fish .35
    e1 = mix["cat"] * math.log(.56) + mix["dog"] * math.log(.36) + mix["fish"] * math.log(.08)
    e2 = mix["cat"] * math.log(.2) + mix["dog"] * math.log(.45) + mix["fish"] * math.log(.35)
    res = dict(fb.relevance_model_feedback("cat", ["d1", "d2"], ["d1", "d2"], 10))
    assert res["d1"] == pytest.approx(e1)
    assert res["d2"] == pytest.approx(e2)


def test_truncate_keeps_top_and_renormalises():
    m = rm.truncate({"a": .5, "b": .3, "c": .2}, 2)
    assert m == pytest.approx({"a": .625, "b": .375})


def test_lambda_one_reproduces_score_candidates_on_toy(monkeypatch):
    monkeypatch.setattr(fb, "FB_MODEL", "rm3")
    monkeypatch.setattr(fb, "FB_LAMBDA", 1.0)
    fb.prepare(os.path.join(TOY, "corpus.jsonl"))
    cands = read_candidates(os.path.join(TOY, "candidates_dev.jsonl"))
    for qid, text in read_queries(os.path.join(TOY, "queries_dev.tsv")):
        pool = [d for d, _ in cands[qid]]
        a = [d for d, _ in fb.score_candidates(text, pool, 10)]
        b = [d for d, _ in fb.relevance_model_feedback(text, pool[5:10], pool, 10)]
        assert a == b


@pytest.mark.parametrize("model", ["rm1", "rm2", "rm3"])
def test_seed_outside_pool_unknown_or_empty(tiny, monkeypatch, model):
    monkeypatch.setattr(fb, "FB_MODEL", model)
    # seed doc d2 not in pool, ghost unknown -> only pool ids come back
    res = fb.relevance_model_feedback("cat", ["d2", "ghost", "d2"], ["d1"], 10)
    assert [d for d, _ in res] == ["d1"]
    # empty / useless seed falls back to plain QL
    assert fb.relevance_model_feedback("cat", [], ["d1", "d2"], 10) == fb.score_candidates("cat", ["d1", "d2"], 10)
    assert fb.relevance_model_feedback("cat", ["ghost"], ["d1", "d2"], 10) == fb.score_candidates("cat", ["d1", "d2"], 10)



def test_lambda_one_reproduces_score_candidates_with_jm_too(monkeypatch):
    monkeypatch.setattr(fb, "FB_MODEL", "rm3")
    monkeypatch.setattr(fb, "FB_LAMBDA", 1.0)
    monkeypatch.setattr(fb, "SMOOTHING", "jm")
    fb.prepare(os.path.join(TOY, "corpus.jsonl"))
    cands = read_candidates(os.path.join(TOY, "candidates_dev.jsonl"))
    for qid, text in read_queries(os.path.join(TOY, "queries_dev.tsv")):
        pool = [d for d, _ in cands[qid]]
        a = [d for d, _ in fb.score_candidates(text, pool, 10)]
        b = [d for d, _ in fb.relevance_model_feedback(text, pool[5:10], pool, 10)]
        assert a == b
