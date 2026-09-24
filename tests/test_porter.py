# porter stemmer vs the worked examples given step by step in the original paper
import pytest

from submission.lm import Analyzer, porter_stem

CASES = {
    # 1a
    "caresses": "caress", "ponies": "poni", "ties": "ti", "caress": "caress", "cats": "cat",
    # 1b
    "feed": "feed", "agreed": "agre", "plastered": "plaster", "bled": "bled", "motoring": "motor",
    "sing": "sing", "conflated": "conflat", "troubled": "troubl", "sized": "size", "hopping": "hop",
    "tanned": "tan", "falling": "fall", "hissing": "hiss", "fizzed": "fizz", "failing": "fail", "filing": "file",
    # 1c
    "happy": "happi", "sky": "sky",
    # 2 -> 5 full pipeline
    "relational": "relat", "conditional": "condit", "rational": "ration", "valenci": "valenc",
    "digitizer": "digit", "conformabli": "conform", "radicalli": "radic", "differentli": "differ",
    "vileli": "vile", "analogousli": "analog", "vietnamization": "vietnam", "predication": "predic",
    "operator": "oper", "feudalism": "feudal", "decisiveness": "decis", "hopefulness": "hope",
    "callousness": "callous", "formaliti": "formal", "sensitiviti": "sensit", "sensibiliti": "sensibl",
    "triplicate": "triplic", "formative": "form", "formalize": "formal", "electriciti": "electr",
    "electrical": "electr", "hopeful": "hope", "goodness": "good",
    "revival": "reviv", "allowance": "allow", "inference": "infer", "airliner": "airlin",
    "gyroscopic": "gyroscop", "adjustable": "adjust", "defensible": "defens", "irritant": "irrit",
    "replacement": "replac", "adjustment": "adjust", "dependent": "depend", "adoption": "adopt",
    "homologou": "homolog", "communism": "commun", "activate": "activ", "angulariti": "angular",
    "homologous": "homolog", "effective": "effect", "bowdlerize": "bowdler",
    "probate": "probat", "rate": "rate", "cease": "ceas", "controll": "control", "roll": "roll",
    "generalizations": "gener", "oscillators": "oscil",
    # short words untouched
    "is": "is", "as": "as",
}


@pytest.mark.parametrize("word,expected", sorted(CASES.items()))
def test_paper_examples(word, expected):
    assert porter_stem(word) == expected


def test_analyzer_porter_on_covid_query():
    an = Analyzer(stop=True, stem="porter")
    assert an("What are the effects of vaccines on infections?") == ["effect", "vaccin", "infect"]
