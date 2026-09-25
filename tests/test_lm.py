# hand checkable QL tests on a 2 doc corpus
#   d1 = "cat cat dog"  (len 3)     d2 = "dog fish"  (len 2)
#   collection: cat 2, dog 2, fish 1, total 5 -> p(cat)=.4 p(dog)=.4 p(fish)=.2
import json
import math

import pytest

import submission.feedback as fb
from submission.lm import Analyzer, s_stem


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    p = tmp_path / "corpus.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"doc_id": "d1", "text": "Cat cat, dog."}) + "\n")
        f.write(json.dumps({"doc_id": "d2", "text": "dog fish"}) + "\n")
    monkeypatch.setattr(fb, "STOPWORDS", False)
    monkeypatch.setattr(fb, "STEMMER", "none")
    return str(p)


def test_dirichlet_by_hand(tiny, monkeypatch):
    monkeypatch.setattr(fb, "SMOOTHING", "dirichlet")
    monkeypatch.setattr(fb, "DIRICHLET_MU", 2.0)
    fb.prepare(tiny)
    res = dict(fb.score_candidates("cat", ["d1", "d2"], 10))
    # d1: (2 + 2*.4)/(3+2) = .56     d2: (0 + .8)/(2+2) = .2
    assert res["d1"] == pytest.approx(math.log(0.56))
    assert res["d2"] == pytest.approx(math.log(0.2))


def test_jm_by_hand_and_order(tiny, monkeypatch):
    monkeypatch.setattr(fb, "SMOOTHING", "jm")
    monkeypatch.setattr(fb, "JM_LAMBDA", 0.5)
    fb.prepare(tiny)
    res = fb.score_candidates("cat fish", ["d1", "d2"], 10)
    # d1: cat .5*2/3+.5*.4 = .5333, fish 0+.1 = .1      d2: cat .2, fish .5*.5+.1 = .35
    d1 = math.log(0.5 * 2 / 3 + 0.2) + math.log(0.1)
    d2 = math.log(0.2) + math.log(0.35)
    assert [d for d, _ in res] == ["d2", "d1"]
    assert dict(res)["d1"] == pytest.approx(d1)
    assert dict(res)["d2"] == pytest.approx(d2)


def test_repeated_query_term_counts_twice(tiny, monkeypatch):
    monkeypatch.setattr(fb, "SMOOTHING", "dirichlet")
    monkeypatch.setattr(fb, "DIRICHLET_MU", 2.0)
    fb.prepare(tiny)
    one = dict(fb.score_candidates("cat", ["d1"], 1))["d1"]
    two = dict(fb.score_candidates("cat cat", ["d1"], 1))["d1"]
    assert two == pytest.approx(2 * one)


def test_unseen_term_skipped_and_empty_query(tiny):
    fb.prepare(tiny)
    a = dict(fb.score_candidates("cat", ["d1", "d2"], 10))
    b = dict(fb.score_candidates("cat zebra", ["d1", "d2"], 10))
    assert a == pytest.approx(b)
    assert fb.score_candidates("zebra", ["d1", "d2"], 10) != []  # all zero, still a valid ranking
    assert len(fb.score_candidates("", ["d1", "d2"], 10)) == 2  # nothing to score on, still a valid list


def test_only_pool_docs_no_dups_k_cap(tiny):
    fb.prepare(tiny)
    res = fb.score_candidates("dog", ["d2", "d2", "d1"], 1)
    assert len(res) == 1 and res[0][0] in {"d1", "d2"}
    res = fb.score_candidates("dog", ["d2", "d2", "d1", "ghost"], 10)
    assert sorted(d for d, _ in res) == ["d1", "d2", "ghost"]  # unknown id scored as empty doc, not a crash


def test_analyzer():
    an = Analyzer(stop=True, stem="s")
    assert an("What are the Studies of COVID-19 viruses?") == ["study", "covid", "19", "viruse"]
    assert Analyzer(stop=False)("The cats") == ["the", "cats"]
    assert [s_stem(w) for w in ["studies", "glasses", "virus", "cells", "is"]] == ["study", "glasse", "virus", "cell", "is"]


def test_fast_tokenizer_matches_regex():
    import re
    from submission.lm import tokens
    rx = re.compile(r"[a-z0-9]+")
    for s in ["COVID-19 (SARS-CoV-2)!", "naïve café 5µm", "K-mer İstanbul ǅ", "a\tb\nc", "", "  x  ",
              "β-coronavirus 2019-nCoV; RT-qPCR@37°C", "日本語 text 123abc"]:
        assert tokens(s) == rx.findall(s.lower()), s
