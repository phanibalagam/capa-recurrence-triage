"""
Load either corpus into one common analysis frame.

Two sources, one schema:

  synthetic  data/deviations.jsonl        generated pharmaceutical deviations
  maude      data/deviations_maude.jsonl  real openFDA device reports (CC0)

The common column names below are what every model and experiment reads, so the
identical pipeline runs on both. That is deliberate: the synthetic corpus is for
learning the method, and the real corpus is what any reported result comes from.

  COMMON            SYNTHETIC                MAUDE
  record_id         deviation_id             report_number
  date              date_opened              date_received
  cause_category    root_cause_category      product_problem
  action_type       capa_type                remedial_action
  action_class      (mapped from capa_type)  action_class
  unit              site                     manufacturer
  area              process_area             product_code
  severity          severity                 event_type
  narrative         investigation_narrative  narrative

Recurrence, group size and observability are derived for MAUDE
(see build_corpus.py) and planted for the synthetic corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SYNTHETIC_PATH = "data/deviations.jsonl"
MAUDE_PATH = "data/deviations_maude.jsonl"

# The synthetic CAPA vocabulary, mapped onto the same interpretable taxonomy
# used for MAUDE remedial actions, so the two corpora can be discussed together.
SYNTH_ACTION_CLASS = {
    "Engineering control": "Engineering / design",
    "Procedure revision": "Communication",
    "Retraining only": "Communication",
    "Supplier corrective action": "Engineering / design",
    "Enhanced monitoring": "Surveillance",
    "No action - not required": "Other",
}

# group_key identifies the cluster within which the outcome is defined. The
# outcome (did a comparable report follow?) is a property of the group, not of the
# record, so records are NOT independent and any resampling must be by group.
COMMON = ["record_id", "date", "cause_category", "action_type", "action_class",
          "unit", "area", "severity", "narrative", "recurred_within_365d",
          "n_prior_in_group", "prior_365", "group_size", "observable", "source",
          "group_key"]

# group_size counts every member of a record's group, including reports that
# arrive AFTER it, so it encodes the outcome. It is kept for description and
# excluded from every model. Adjustment uses backward-looking counts only.
LEAKY = {"group_size"}


def _read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `python run.py setup --source "
            f"{'maude' if 'maude' in str(p) else 'synthetic'}` first.")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_synthetic(path: str) -> pd.DataFrame:
    df = pd.DataFrame(_read_jsonl(path))
    out = pd.DataFrame({
        "record_id": df["deviation_id"],
        "date": pd.to_datetime(df["date_opened"]),
        "cause_category": df["root_cause_category"],
        "action_type": df["capa_type"],
        "action_class": df["capa_type"].map(SYNTH_ACTION_CLASS).fillna("Other"),
        "unit": df["site"],
        "area": df["process_area"],
        "severity": df["severity"],
        "narrative": df["investigation_narrative"],
        "recurred_within_365d": df["recurred_within_365d"].astype(int),
    })
    # The synthetic corpus has no group structure and no censoring: every row is
    # observable by construction. Group size is constant so it cannot confound.
    out["group_size"] = 1
    out["n_prior_in_group"] = 0
    out["prior_365"] = 0
    out["observable"] = 1
    out["source"] = "synthetic"
    return out


def _load_maude(path: str) -> pd.DataFrame:
    df = pd.DataFrame(_read_jsonl(path))
    out = pd.DataFrame({
        "record_id": df["report_number"],
        "date": pd.to_datetime(df["date_received"]),
        "cause_category": df["product_problem"],
        "action_type": df["remedial_action"],
        "action_class": df["action_class"],
        "unit": df["manufacturer"],
        "area": df["product_code"],
        "severity": df["event_type"],
        "narrative": df["narrative"],
        "recurred_within_365d": df["recurred_within_365d"].astype(int),
        "group_size": df["group_size"].astype(int),      # descriptive only
        "n_prior_in_group": df["n_prior_in_group"].astype(int),
        "prior_365": df.get("prior_365", 0),
        "observable": df["observable"].astype(int),
    })
    out["source"] = "maude"
    out.attrs["is_fixture"] = bool(df.get("is_fixture", pd.Series([False])).any())
    return out


def load(source: str = "synthetic", path: str | None = None,
         observable_only: bool = True) -> pd.DataFrame:
    """
    observable_only drops rows without a full 365-day follow-up window. Leave it
    on for anything that reports a recurrence figure; a corpus that includes
    right-censored rows understates recurrence, and understates it worst for the
    most recent reports.
    """
    if source == "synthetic":
        df = _load_synthetic(path or SYNTHETIC_PATH)
    elif source == "maude":
        df = _load_maude(path or MAUDE_PATH)
    else:
        raise ValueError(f"unknown source {source!r}; use 'synthetic' or 'maude'")

    if observable_only:
        n0 = len(df)
        df = df[df["observable"] == 1].copy()
        df.attrs["dropped_censored"] = n0 - len(df)

    # The cluster the outcome is defined within: (product code, manufacturer,
    # primary problem) for MAUDE, which store as (area, unit, cause_category).
    df["group_key"] = (df["area"].astype(str) + "|" + df["unit"].astype(str)
                       + "|" + df["cause_category"].astype(str))

    df["searchable"] = (df["area"].astype(str) + " " + df["cause_category"].astype(str)
                        + " " + df["narrative"].astype(str))
    return df.sort_values("date").reset_index(drop=True)


def temporal_split(df: pd.DataFrame, frac_train: float = 0.75):
    """Train on the past, test on the future."""
    cut = int(len(df) * frac_train)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
