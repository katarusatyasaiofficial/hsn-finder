"""Run with:  pytest -q"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsn_finder import HSNMatcher, strength_label  # noqa: E402
from hsn_finder.chapters import CHAPTER_TITLES  # noqa: E402
from hsn_finder.synonyms import expand_query  # noqa: E402
from hsn_finder.text import normalize, stem  # noqa: E402


@pytest.fixture(scope="module")
def matcher():
    return HSNMatcher.from_csv()


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(ROOT / "data" / "hsn_codes.csv", dtype=str)


# ---- data ------------------------------------------------------------------
def test_dataset_shape(df):
    assert len(df) == 12604
    assert df["code"].is_unique
    assert df["code"].str.fullmatch(r"\d{8}").all()          # leading zeros restored
    assert (df["heading"] == df["code"].str[:4]).all()


def test_every_chapter_has_a_title(df):
    missing = set(df["chapter"]) - set(CHAPTER_TITLES)
    assert not missing, f"chapters without a title: {missing}"


def test_eval_sets_reference_real_headings(df):
    headings = set(df["heading"])
    for name in ("dev", "test", "holdout"):
        q = pd.read_csv(ROOT / "eval" / f"{name}_queries.csv", dtype=str)
        for acc in q["accepted_headings"]:
            assert set(acc.split("|")) <= headings, (name, acc)


def test_no_gst_rate_columns(df):
    # Deliberate: source rates are unreliable (see README), so none are shipped.
    assert not any("gst" in c.lower() or "rate" in c.lower() for c in df.columns)


# ---- text ------------------------------------------------------------------
@pytest.mark.parametrize(
    "word,expected",
    [("tiles", "tile"), ("batteries", "battery"), ("boxes", "box"), ("mangoes", "mango"),
     ("glass", "glass"), ("shoes", "shoe"), ("bus", "bus"), ("gas", "gas")],
)
def test_stemmer(word, expected):
    assert stem(word) == expected


def test_normalize_drops_stopwords():
    assert normalize("Cotton T-Shirts for the Men") == "cotton t shirt men"


def test_synonym_expansion():
    assert "telephones for cellular networks" in expand_query("mobile phone")
    assert expand_query("zzzz unknown thing") == "zzzz unknown thing"


# ---- matcher ---------------------------------------------------------------
def test_basic_lookup(matcher):
    res = matcher.search("stainless steel kitchen sink", k=5)
    assert res[0].heading == "7324"
    assert res[0].chapter == "73"
    assert "iron or steel" in res[0].chapter_title.lower()


@pytest.mark.parametrize(
    "query,heading",
    [("basmati rice", "1006"), ("turmeric powder", "0910"), ("mobile phone", "8517"),
     ("portland cement", "2523"), ("ceramic floor tiles", "6907")],
)
def test_expected_heading_in_top5(matcher, query, heading):
    assert heading in [m.heading for m in matcher.search(query, k=5)]


def test_typo_is_corrected(matcher):
    assert matcher.fix_spelling("staniless steel kitchen sink") == "stainless steel kitchen sink"
    res = matcher.search("staniless steel kitchen sink", k=5)
    assert "7324" in [m.heading for m in res]


def test_common_words_not_miscorrected(matcher):
    assert matcher.fix_spelling("something nice") == "something nice"


@pytest.mark.parametrize("query", ["", "   ", "xyzzy qqq", "hello how are you", "123456"])
def test_gibberish_returns_no_match(matcher, query):
    assert matcher.search(query) == []


def test_results_sorted_and_complete(matcher):
    res = matcher.search("cotton fabric", k=5)
    scores = [m.score for m in res]
    assert scores == sorted(scores, reverse=True)
    for m in res:
        assert len(m.code) == 8 and m.code.startswith(m.heading) and m.heading.startswith(m.chapter)
        assert m.path and m.description


def test_top_headings_are_distinct(matcher):
    heads = [h for h, _ in matcher.top_headings("steel pipe", n=5)]
    assert len(heads) == len(set(heads))


def test_strength_labels():
    assert strength_label(0.9) == "High"
    assert strength_label(0.5) == "Medium"
    assert strength_label(0.2) == "Low"
