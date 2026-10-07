"""Lightweight text normalisation shared by the index and the queries."""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words that appear in nearly every tariff description and carry no signal,
# on top of the usual English stop words. (IDF already down-weights them, but
# removing them keeps bigrams clean.)
DOMAIN_STOPWORDS = {
    "other", "others", "whether", "not", "elsewhere", "specified", "included",
    "including", "excluding", "thereof", "nes", "nec", "parts", "part",
}

STOPWORDS = (set(ENGLISH_STOP_WORDS) | DOMAIN_STOPWORDS) - {"men", "women", "man", "woman"}

_TOKEN = re.compile(r"[a-z0-9]+")


def stem(token: str) -> str:
    """Very small suffix stripper (plural handling only).

    Deliberately simple and dependency-free: 'tiles' -> 'tile',
    'batteries' -> 'battery', 'boxes' -> 'box', 'glass' stays 'glass'.
    """
    if len(token) <= 3 or token.isdigit():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("oes") and len(token) > 5:   # mangoes, potatoes, tomatoes
        return token[:-2]
    if token.endswith(("ss", "us", "is")):
        return token
    if token.endswith("s"):
        return token[:-1]
    return token


def normalize(text: str, drop_stopwords: bool = True) -> str:
    """Lower-case, tokenise on alphanumerics, drop stop words, stem plurals."""
    tokens = _TOKEN.findall(str(text).lower())
    out = []
    for t in tokens:
        if drop_stopwords and t in STOPWORDS:
            continue
        out.append(stem(t))
    return " ".join(out)
