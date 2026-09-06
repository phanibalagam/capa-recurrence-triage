"""
Precedent retrieval: find historical deviations like this one, and show what was
done about them and whether it worked.

This is the function that is deployable in a quarter and needs no model training.
It is also the one investigators actually want: not "what is the root cause" but
"has this happened here before, what did we do, and did it hold?"
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PrecedentIndex:
    def __init__(self, min_df: int = 2, ngram: tuple = (1, 2)):
        self.vec = TfidfVectorizer(min_df=min_df, ngram_range=ngram,
                                   stop_words="english", sublinear_tf=True)
        self.matrix = None
        self.df: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> "PrecedentIndex":
        self.df = df.reset_index(drop=True)
        self.matrix = self.vec.fit_transform(self.df["searchable"])
        return self

    def search(self, text: str, k: int = 5,
               same_site: str | None = None,
               same_area: str | None = None) -> pd.DataFrame:
        q = self.vec.transform([text])
        sims = cosine_similarity(q, self.matrix).ravel()

        mask = pd.Series(True, index=self.df.index)
        if same_site:
            mask &= self.df["unit"].eq(same_site)
        if same_area:
            mask &= self.df["area"].eq(same_area)
        if mask.sum() < k:            # filters too tight - fall back to global
            mask = pd.Series(True, index=self.df.index)

        cand = self.df[mask].copy()
        cand["similarity"] = sims[mask.values]
        return cand.nlargest(k, "similarity")[[
            "record_id", "date", "unit", "area", "severity", "cause_category",
            "action_type", "action_class", "recurred_within_365d",
            "group_size", "narrative", "similarity"]]


def precedent_summary(hits: pd.DataFrame) -> dict:
    """What the retrieved precedent says as a block, not one row at a time."""
    if hits.empty:
        return {}
    return {
        "n": len(hits),
        "most_common_cause": hits["cause_category"].mode().iat[0],
        "most_common_action": hits["action_type"].mode().iat[0],
        "share_recurred": float(hits["recurred_within_365d"].mean()),
        "cited_ids": hits["record_id"].tolist(),
    }
