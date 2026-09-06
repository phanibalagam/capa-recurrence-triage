"""End-to-end triage: precedent -> root cause -> CAPA -> recurrence risk -> draft."""

from __future__ import annotations

import pandas as pd

from src.draft import TriageResult, compose_narrative
from src.models import RecurrenceModel, RootCauseClassifier
from src.retrieve import PrecedentIndex, precedent_summary

# Which CAPA type historically holds, per root cause. Learned from the training
# data at fit time rather than hard-coded, so it tracks your actual experience.
class TriageSystem:
    def __init__(self, train: pd.DataFrame,
                 high_risk_threshold: float = 0.35):
        self.train = train
        self.index = PrecedentIndex().fit(train)
        self.rc = RootCauseClassifier().fit(train)
        self.rm = RecurrenceModel().fit(train)
        self.threshold = high_risk_threshold
        self.best_capa = self._learn_best_capa(train)

    @staticmethod
    def _learn_best_capa(train: pd.DataFrame) -> dict[str, str]:
        """For each cause category, the action with the lowest observed
        recurrence among those used at least 15 times. Unadjusted, and
        advisory only."""
        fallback = train["action_type"].mode().iat[0]
        out = {}
        for cause, g in train.groupby("cause_category"):
            t = g.groupby("action_type").agg(n=("recurred_within_365d", "size"),
                                             rate=("recurred_within_365d", "mean"))
            t = t[t["n"] >= 15]
            out[cause] = t["rate"].idxmin() if len(t) else fallback
        return out

    def triage(self, description: str, site: str, process_area: str,
               severity: str = "Major", k: int = 5,
               narrative_backend: str = "none",
               deviation_id: str = "NEW") -> TriageResult:
        hits = self.index.search(description, k=k, same_site=site,
                                 same_area=process_area)
        causes = self.rc.predict_top(description, k=3)
        top_cause = causes[0][0]
        capa = self.best_capa.get(top_cause, "Procedure revision")

        risk_row = pd.DataFrame([{
            "area": process_area, "severity": severity,
            "cause_category": top_cause, "action_type": capa, "unit": site}])
        risk = float(self.rm.predict_proba(risk_row)[0])

        # The warning that earns the project its keep: compare the CAPA the
        # quality unit is most likely to choose against the one that holds.
        likely = (self.train[self.train["cause_category"] == top_cause]
                  ["action_type"].mode())
        flag = ""
        if len(likely) and likely.iat[0] != capa:
            lr = self.train[(self.train["cause_category"] == top_cause) &
                            (self.train["action_type"] == likely.iat[0])
                            ]["recurred_within_365d"].mean()
            br = self.train[(self.train["cause_category"] == top_cause) &
                            (self.train["action_type"] == capa)
                            ]["recurred_within_365d"].mean()
            flag = (f"'{likely.iat[0]}' is the CAPA usually chosen for "
                    f"{top_cause} here and recurs {lr:.0%} of the time; "
                    f"'{capa}' recurs {br:.0%} of the time.")
        elif risk >= self.threshold:
            flag = (f"recurrence risk above the {self.threshold:.0%} review "
                    f"threshold - consider escalating the CAPA type.")

        res = TriageResult(
            deviation_id=deviation_id, description=description,
            predicted_root_causes=causes, precedent=hits,
            precedent_summary=precedent_summary(hits),
            recommended_capa=capa, recurrence_risk=risk, risk_flag=flag)
        res.narrative = compose_narrative(res, backend=narrative_backend)
        return res
