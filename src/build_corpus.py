"""
Turn cached openFDA MAUDE reports into the analysis corpus.

The important part of this file is the recurrence derivation, because it is what
lets the study report an observational finding on real public data rather than
recovering a signal someone planted.

RECURRENCE, DERIVED
    No dataset carries "did the problem come back". It is constructed here:

      group key   (device_report_product_code, manufacturer, primary problem)
      recurrence  another report in the same group with date_received in
                  (D, D + window], where D is this report's receipt date

    Two methodological details decide whether the resulting number means
    anything, and both are handled explicitly.

    1. RIGHT CENSORING. A report received near the end of the observation
       window has had less than `window` days in which to recur, so including
       it drags the estimate down for recent reports specifically. Only reports
       with a full window of follow-up inside the corpus are marked observable,
       and only observable reports enter the analysis.

    2. REPORTING VOLUME, MEASURED BACKWARDS ONLY. A group that gets reported
       more is mechanically more likely to contain a later report, whatever the
       action was, so reporting volume must be adjusted for.

       But it must be measured using ONLY the past. `group_size` counts every
       member of a group including reports that arrive after this one, so it
       contains the outcome: "does a later report exist" is most of what
       group_size measures. Adjusting for it does not remove confounding, it
       conditions on the answer.

       The covariates used for adjustment are therefore `n_prior_in_group`
       (reports before this one, all time) and `prior_365` (reports in the same
       group in the year before this one). `group_size` is retained as a
       DESCRIPTIVE statistic only and is excluded from every model.

    Neither is optional. Skip the first and the estimate is biased; skip the
    second and any association with remedial action is confounded by how much
    a device gets reported.

    python -m src.build_corpus --raw data/maude_raw.jsonl --out data/deviations_maude.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# MAUDE's remedial_action vocabulary, grouped into an interpretable taxonomy.
# The grouping is ours and is stated in the data card; raw values are retained.
ACTION_CLASS = {
    "Recall": "Engineering / design",
    "Repair": "Engineering / design",
    "Replace": "Engineering / design",
    "Modification/Adjustment": "Engineering / design",
    "Relabeling": "Communication",
    "Notification": "Communication",
    "Inspection": "Surveillance",
    "Patient Monitoring": "Surveillance",
    "Other": "Other",
    "None": "Other",
}

NARRATIVE_TYPES = {"Description of Event or Problem",
                   "Additional Manufacturer Narrative"}


def _date(s: str | None):
    if not s or len(s) < 8:
        return None
    try:
        return datetime.strptime(s[:8], "%Y%m%d").date()
    except ValueError:
        return None


def _first(xs, default=""):
    return xs[0] if isinstance(xs, list) and xs else default


def normalise(rec: dict) -> dict | None:
    """One raw MAUDE report -> one canonical record, or None if unusable."""
    received = _date(rec.get("date_received"))
    problems = [p for p in (rec.get("product_problems") or []) if p]
    actions = [a for a in (rec.get("remedial_action") or []) if a]
    if not (received and problems and actions):
        return None

    dev = _first(rec.get("device") or [], {}) or {}
    ofda = dev.get("openfda") or {}

    narrative = " ".join(
        t.get("text", "").strip()
        for t in (rec.get("mdr_text") or [])
        if t.get("text") and t.get("text_type_code") in NARRATIVE_TYPES).strip()
    if len(narrative) < 40:
        return None

    action = actions[0]
    return {
        "report_number": rec.get("report_number", ""),
        "date_received": received.isoformat(),
        "event_type": rec.get("event_type", ""),
        "product_problem": problems[0],
        "all_problems": problems,
        "remedial_action": action,
        "action_class": ACTION_CLASS.get(action, "Other"),
        "all_actions": actions,
        "manufacturer": (rec.get("manufacturer_name")
                         or dev.get("manufacturer_d_name") or "UNKNOWN").strip().upper(),
        "product_code": dev.get("device_report_product_code", "") or "UNKNOWN",
        "brand_name": (dev.get("brand_name") or "").strip(),
        "device_class": ofda.get("device_class", ""),
        "specialty": ofda.get("medical_specialty_description", ""),
        "narrative": narrative,
        # provenance, per the hybrid-data rule: every field is attributable
        "provenance": "openFDA device/event (MAUDE), CC0 1.0",
        "synthetic": False,
        "is_fixture": bool(rec.get("_fixture")),
    }


def collapse_batch_filings(rows: list[dict]) -> list[dict]:
    """
    Collapse each (group, receipt date) to a single filing event.

    THE UNIT OF ANALYSIS IS A FILING EVENT, NOT A REPORT.

    MAUDE contains mass filings: a manufacturer submits thousands of reports for
    one problem on one device on a single day. In the corpus this was written
    against, one such filing accounted for 2,413 of the 4,528 normalised reports, all on 2022-01-01, all one product code, manufacturer and
    problem.

    That breaks the outcome. Recurrence is "another report in this group with a
    receipt date strictly after this one, within a year", so reports sharing a
    date cannot recur against each other, and a batch of 2,413 same-day reports
    scores zero recurrence by construction. Treating them as 2,413 independent
    observations does not just understate variance; it reverses the direction of
    the estimate. Before this collapse the unadjusted odds ratio for
    engineering-type actions was 0.229 (they appeared far LESS likely to recur).
    After it, on the same data, it is 1.874.

    One record per (group, date) is kept, carrying `n_reports`: how many reports
    that filing event stands for. Everything downstream - the outcome, the prior
    counts, the observability cut - is then derived from filing events.

    `n_reports` is what makes the superseded record-level analysis reproducible
    without the 172 MB raw download. Reports collapsed into one event share the
    group key, the receipt date, the action class and, by construction, the
    outcome, so replicating each event `n_reports` times reconstructs exactly the
    record-level table the first analysis was fitted on. `run.py experiments
    --record-level` does that, and reproduces the 0.229 this paper reports as the
    error. See METHODS.md section 6.3.
    """
    first: dict[tuple, dict] = {}
    for r in sorted(rows, key=lambda x: x["date_received"]):
        key = (r["product_code"], r["manufacturer"], r["product_problem"],
               r["date_received"])
        if key in first:
            first[key]["n_reports"] += 1
            continue
        r["n_reports"] = 1
        first[key] = r
    return list(first.values())


def derive_recurrence(rows: list[dict], window_days: int = 365) -> list[dict]:
    """
    Adds: recurred_within_365d, days_to_next, n_prior_in_group, prior_365,
    group_size, observable.

    Operates on filing events; see collapse_batch_filings.

    Only `observable` rows have a full follow-up window. Only `n_prior_in_group`
    and `prior_365` are safe to use as model inputs - `group_size` looks into the
    future and is descriptive only.
    """
    for r in rows:
        r["_d"] = datetime.fromisoformat(r["date_received"]).date()
    corpus_end = max(r["_d"] for r in rows)
    cutoff = corpus_end - timedelta(days=window_days)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["product_code"], r["manufacturer"], r["product_problem"])].append(r)

    for key, members in groups.items():
        members.sort(key=lambda r: r["_d"])
        dates = [m["_d"] for m in members]
        for i, r in enumerate(members):
            d = r["_d"]
            later = [x for x in dates[i + 1:] if d < x <= d + timedelta(days=window_days)]
            r["recurred_within_365d"] = int(bool(later))
            r["days_to_next"] = (later[0] - d).days if later else None
            earlier = [x for x in dates[:i] if d - timedelta(days=window_days) <= x < d]
            r["n_prior_in_group"] = i          # backward-looking, safe
            r["prior_365"] = len(earlier)      # backward-looking, safe
            r["group_size"] = len(members)     # LEAKY: includes future members
            r["observable"] = int(d <= cutoff)
            r["group_key"] = "|".join(key)

    for r in rows:
        del r["_d"]
        r["derived_fields"] = ["recurred_within_365d", "days_to_next",
                               "n_prior_in_group", "prior_365", "group_size",
                               "observable", "action_class"]
        r["leaky_fields"] = ["group_size"]   # descriptive only, never a model input
    return rows


def summarise(rows: list[dict]) -> None:
    obs = [r for r in rows if r["observable"]]
    print(f"\n  records normalised      {len(rows)}")
    print(f"  with full follow-up     {len(obs)}  "
          f"({len(obs) / max(len(rows), 1):.0%})  <- analysis set")
    if not obs:
        return
    print(f"  overall recurrence      "
          f"{sum(r['recurred_within_365d'] for r in obs) / len(obs):.1%}")
    print(f"  distinct groups         "
          f"{len({r['group_key'] for r in obs})}")

    print("\n  recurrence by action class (observable rows only):")
    by = defaultdict(list)
    for r in obs:
        by[r["action_class"]].append(r["recurred_within_365d"])
    for k, v in sorted(by.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"    {k:<24} n={len(v):>6}  {sum(v) / len(v):>6.1%}")
    print("\n  NOTE: unadjusted. Reporting volume confounds this table; see the "
          "adjusted\n        analysis in src/experiments.py before quoting any of "
          "it. Adjustment\n        uses backward-looking counts only - group_size "
          "includes future\n        reports and is descriptive, never a model "
          "input.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/maude_raw.jsonl")
    ap.add_argument("--out", default="data/deviations_maude.jsonl")
    ap.add_argument("--window", type=int, default=365)
    ap.add_argument("--allow-fixture", action="store_true",
                    help="permit building from the offline test fixture")
    a = ap.parse_args()

    raw_path = Path(a.raw)
    if not raw_path.exists():
        raise SystemExit(
            f"{raw_path} not found.\n"
            "Run  python -m src.fetch_maude  on a machine with access to\n"
            "api.fda.gov, or pass --raw data/maude_fixture.jsonl --allow-fixture\n"
            "to exercise the pipeline offline.")

    raw = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if raw and raw[0].get("_fixture") and not a.allow_fixture:
        raise SystemExit(
            "Refusing to build an analysis corpus from the test fixture.\n"
            "The fixture is not data. Pass --allow-fixture if you are testing "
            "the pipeline.")

    rows = [n for n in (normalise(r) for r in raw) if n]
    if not rows:
        raise SystemExit("no usable records: none had narrative + problem + action")

    n_reports = len(rows)
    rows = collapse_batch_filings(rows)
    print(f"  {n_reports} reports -> {len(rows)} filing events "
          f"({n_reports - len(rows)} same-day duplicates within a group collapsed)")

    rows = derive_recurrence(rows, a.window)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} records -> {out}")
    if rows[0]["is_fixture"]:
        print("  *** built from the TEST FIXTURE - not data ***")
    summarise(rows)


if __name__ == "__main__":
    main()
