"""TF-IDF based HSN code matcher.

How it works
------------
1. Every HSN code's official description is normalised and indexed with
   TF-IDF over word unigrams + bigrams.
2. (optional) A second index over character n-grams makes matching robust to
   spelling variants, run-together words and unusual plurals.
3. (optional) A heading-level index (all descriptions under a 4-digit heading
   concatenated) lets a strong heading match lift all its sub-codes. This is
   what helps when the seller's words match the *heading* wording but the
   code-specific text is generic ("OTHER").
4. Scores are blended and the top-k codes returned with their hierarchy.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .synonyms import SYNONYMS, expand_query
from .text import STOPWORDS, normalize, stem

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "hsn_codes.csv"


# Words after which a product phrase turns into qualifiers ("shirt FOR men").
_CUT_WORDS = {"for", "with", "of", "in", "from", "used", "made", "using", "without"}
_CONJ = {"and", "or"}
# Quantity / size units that often trail a product phrase ("bulb 9 watt").
_UNITS = {
    "w", "watt", "watts", "kw", "hp", "v", "volt", "kg", "g", "gm", "mg", "ml", "ltr",
    "litre", "liter", "mm", "cm", "inch", "ft", "mtr", "meter", "gsm", "pcs", "piece",
    "pack", "amp", "mah", "kva",
}


def head_noun(query: str) -> str | None:
    """Best guess of the product's head noun (English phrases are head-final).

    "stainless steel kitchen sink" -> "sink"; "cotton t-shirt for men" -> "shirt";
    "LED bulb 9 watt" -> "bulb". Returns None for lists ("bolts and nuts")
    or single-word queries, where boosting would not help.
    """
    import re

    toks = re.findall(r"[a-z0-9]+", query.lower())
    for i, t in enumerate(toks):
        if t in _CUT_WORDS:
            toks = toks[:i]
            break
    if any(t in _CONJ for t in toks):
        return None
    toks = [t for t in toks if not t.isdigit() and t not in _UNITS and t not in STOPWORDS]
    if len(toks) < 2:
        return None
    return stem(toks[-1])


def strength_label(score: float) -> str:
    """Human-friendly label for a similarity score.

    Thresholds come from the calibration table in eval/RESULTS.md: rank-1 was
    correct ~90% of the time at score >= 0.6, ~75% at 0.45-0.6, and ~70% below
    (small samples; treat as indicative, not as probabilities).
    """
    if score >= 0.6:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


@dataclass
class Match:
    code: str
    description: str
    path: str
    chapter: str
    chapter_title: str
    heading: str
    subheading: str
    score: float

    @property
    def code_formatted(self) -> str:
        """8-digit code displayed as 'XXXX XX XX'."""
        return f"{self.code[:4]} {self.code[4:6]} {self.code[6:]}"


@dataclass
class Config:
    char_weight: float = 0.3      # share of the code-level score from char n-grams
    heading_weight: float = 0.3   # share of the final score from the heading-level index
    use_synonyms: bool = True     # seller-vocabulary -> tariff-vocabulary bridge
    head_boost: int = 0           # extra repetitions of the head noun (no consistent gain; off)
    spell_fix: bool = True        # correct unknown words to the closest known word
    min_score: float = 0.15       # below this top score, report "no confident match"
    word_ngrams: tuple = (1, 2)
    char_ngrams: tuple = (3, 5)


@dataclass
class HSNMatcher:
    df: pd.DataFrame
    config: Config = field(default_factory=Config)

    # ---- construction -------------------------------------------------
    @classmethod
    def from_csv(cls, path: str | Path = DEFAULT_DATA, config: Config | None = None) -> "HSNMatcher":
        df = pd.read_csv(path, dtype=str).fillna("")
        return cls(df=df, config=config or Config())

    def __post_init__(self) -> None:
        cfg = self.config
        self.df = self.df.reset_index(drop=True)
        docs = self.df["description"].map(normalize)

        self.word_vec = TfidfVectorizer(
            ngram_range=cfg.word_ngrams, sublinear_tf=True, min_df=1
        )
        self.word_mat = self.word_vec.fit_transform(docs)

        # Vocabulary for spelling correction: tariff words + seller-vocabulary words
        self._vocab = Counter()
        for d in docs:
            self._vocab.update(d.split())
        for key in SYNONYMS:
            self._vocab.update(key.split())
        self._by_initial: dict[str, list[str]] = {}
        for w in self._vocab:
            if len(w) >= 4:
                self._by_initial.setdefault(w[0], []).append(w)

        self.char_vec = None
        if cfg.char_weight > 0:
            self.char_vec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=cfg.char_ngrams, sublinear_tf=True, min_df=2
            )
            self.char_mat = self.char_vec.fit_transform(docs)

        self.head_vec = None
        if cfg.heading_weight > 0:
            grp = (
                pd.DataFrame({"heading": self.df["heading"], "doc": docs})
                .groupby("heading")["doc"]
                .apply(lambda s: " ".join(s))
            )
            self.heading_index = {h: i for i, h in enumerate(grp.index)}
            self.head_vec = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)
            self.head_mat = self.head_vec.fit_transform(grp.values)
            self.code_to_head_idx = self.df["heading"].map(self.heading_index).to_numpy()

    # ---- query preparation -------------------------------------------
    def fix_spelling(self, query: str) -> str:
        """Replace unknown words (len >= 5) by the closest known word.

        A word is "known" if its stemmed form occurs in the tariff text or in
        the seller-vocabulary table. Candidates must share the first letter
        (typos rarely hit it) and be >= 0.8 similar; the most frequent wins.
        """
        out = []
        for tok in re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]+", query):
            low = tok.lower()
            if (
                len(low) >= 5
                and low.isalpha()
                and low not in STOPWORDS
                and stem(low) not in self._vocab
            ):
                cands = get_close_matches(stem(low), self._by_initial.get(low[0], []), n=3, cutoff=0.8)
                if cands:
                    tok = max(cands, key=lambda c: self._vocab[c])
            out.append(tok)
        return "".join(out)

    # ---- scoring ------------------------------------------------------
    def _scores(self, query: str) -> np.ndarray:
        cfg = self.config
        if cfg.spell_fix:
            query = self.fix_spelling(query)
        q = expand_query(query) if cfg.use_synonyms else query
        qn = normalize(q)
        if cfg.head_boost > 0:
            head = head_noun(query)
            if head:
                qn = f"{qn} " + " ".join([head] * cfg.head_boost)
        if not qn:
            return np.zeros(len(self.df))

        word = (self.word_mat @ self.word_vec.transform([qn]).T).toarray().ravel()
        score = word
        if self.char_vec is not None:
            char = (self.char_mat @ self.char_vec.transform([qn]).T).toarray().ravel()
            score = (1 - cfg.char_weight) * word + cfg.char_weight * char
        if self.head_vec is not None:
            head = (self.head_mat @ self.head_vec.transform([qn]).T).toarray().ravel()
            score = (1 - cfg.heading_weight) * score + cfg.heading_weight * head[self.code_to_head_idx]
        return score

    def search(self, query: str, k: int = 5) -> list[Match]:
        """Return the top-k matching HSN codes for a plain-English description."""
        if not query or not query.strip():
            return []
        scores = self._scores(query)
        if scores.max() < self.config.min_score:
            return []
        top = np.argsort(-scores)[:k]
        out = []
        for i in top:
            if scores[i] <= 0:
                continue
            r = self.df.iloc[i]
            out.append(
                Match(
                    code=r["code"], description=r["description"], path=r["path"],
                    chapter=r["chapter"], chapter_title=r["chapter_title"],
                    heading=r["heading"], subheading=r["subheading"],
                    score=float(scores[i]),
                )
            )
        return out

    def top_headings(self, query: str, n: int = 5) -> list[tuple[str, float]]:
        """Best 4-digit headings, scored by their strongest code (for the UI)."""
        scores = self._scores(query)
        s = pd.Series(scores).groupby(self.df["heading"]).max().sort_values(ascending=False)
        return [(h, float(v)) for h, v in s.head(n).items() if v > 0]
