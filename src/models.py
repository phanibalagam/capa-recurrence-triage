"""
Two models, both with the same discipline as project 06: temporal split, an
honest baseline, and a gate that can fail.

  RootCauseClassifier   narrative text -> root cause category. Advisory only.
  RecurrenceModel       deviation + chosen CAPA -> will this come back?

The recurrence model is the one with teeth. It does not tell you the root cause;
it tells you that the CAPA you are about to close is the kind that comes back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class Scored:
    name: str
    metrics: dict
    baseline: dict
    beats_baseline: bool


class RootCauseClassifier:
    """TF-IDF + logistic regression. Deliberately simple and inspectable."""

    def __init__(self):
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2),
                                      stop_words="english", sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, C=4.0,
                                       class_weight="balanced")),
        ])

    def fit(self, df: pd.DataFrame):
        self.pipe.fit(df["searchable"], df["cause_category"])
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.pipe.predict(texts))

    def predict_top(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        p = self.pipe.predict_proba([text])[0]
        classes = self.pipe.named_steps["clf"].classes_
        order = np.argsort(-p)[:k]
        return [(classes[i], float(p[i])) for i in order]

    def evaluate(self, test: pd.DataFrame, train: pd.DataFrame) -> Scored:
        pred = self.predict(test["searchable"].tolist())
        y = test["cause_category"]
        dummy = DummyClassifier(strategy="most_frequent").fit(
            train["searchable"], train["cause_category"])
        d_pred = dummy.predict(test["searchable"])
        m = {"accuracy": round(accuracy_score(y, pred), 4),
             "macro_f1": round(f1_score(y, pred, average="macro"), 4)}
        b = {"accuracy": round(accuracy_score(y, d_pred), 4),
             "macro_f1": round(f1_score(y, d_pred, average="macro", zero_division=0), 4)}
        return Scored("root_cause", m, b, m["macro_f1"] > b["macro_f1"])

    def report(self, test: pd.DataFrame) -> str:
        return classification_report(test["cause_category"],
                                     self.predict(test["searchable"].tolist()),
                                     zero_division=0)


class RecurrenceModel:
    """
    Predicts whether a deviation with this CAPA recurs within 365 days.

    Features are deliberately the things known AT CAPA APPROVAL - process area,
    severity, root cause, CAPA type, site - because that is the moment the
    prediction has to be useful. A model that needs the effectiveness check
    result predicts the past.
    """

    # Common-schema columns, so the identical model runs on either corpus.
    # group_size is CRITICAL on real data: a device that gets reported more is
    # mechanically more likely to be reported again, whatever action was taken.
    # Leaving it out lets reporting volume masquerade as an effect of the action.
    FEATURES = ["area", "severity", "cause_category", "action_type", "unit"]
    # Backward-looking reporting volume only. group_size is NOT here: it counts
    # future members of the group and therefore leaks the outcome.
    ADJUSTED = FEATURES + ["n_prior_in_group", "prior_365"]
    LEAKY_DEMO = FEATURES + ["group_size"]   # used only to demonstrate the leak

    def __init__(self):
        self.enc = OneHotEncoder(handle_unknown="ignore")
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    def fit(self, df: pd.DataFrame):
        X = self.enc.fit_transform(df[self.FEATURES])
        self.clf.fit(X, df["recurred_within_365d"])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self.enc.transform(df[self.FEATURES]))[:, 1]

    def evaluate(self, test: pd.DataFrame, train: pd.DataFrame) -> Scored:
        p = self.predict_proba(test)
        y = test["recurred_within_365d"]
        base_rate = float(train["recurred_within_365d"].mean())
        m = {"roc_auc": round(float(roc_auc_score(y, p)), 4),
             "pr_auc": round(float(average_precision_score(y, p)), 4),
             "base_rate": round(float(y.mean()), 4)}
        b = {"roc_auc": 0.5, "pr_auc": round(float(y.mean()), 4),
             "note": f"train base rate {base_rate:.3f}"}
        return Scored("recurrence", m, b,
                      m["roc_auc"] > 0.60 and m["pr_auc"] > b["pr_auc"] * 1.15)

    def capa_risk_table(self, df: pd.DataFrame, by: str = "action_type") -> pd.DataFrame:
        """
        Observed recurrence by action, in the form a quality head can act on.

        UNADJUSTED by construction. On real data this table is confounded by how
        much a device or product line gets reported; read it alongside the
        adjusted estimate in experiments.py rather than on its own.
        """
        t = (df.groupby(by)
               .agg(n=("recurred_within_365d", "size"),
                    recurrence_rate=("recurred_within_365d", "mean"),
                    median_group_size=("group_size", "median"))
               .sort_values("recurrence_rate", ascending=False))
        t["recurrence_rate"] = (t["recurrence_rate"] * 100).round(1)
        return t
