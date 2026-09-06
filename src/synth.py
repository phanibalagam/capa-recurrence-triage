"""
Generate a SYNTHETIC deviation corpus.

Nothing here is real. Deviation ids are prefixed DEV-SYN- so no record can be
mistaken for an actual quality event.

Two signals are deliberately planted, because a model trained on noise teaches
you nothing:

  1. Root cause is inferable from the narrative text (each root cause has its own
     vocabulary), so the classifier has something real to learn.

  2. Recurrence depends on the CAPA type, not just the deviation. Retraining-only
     CAPAs against Human Error root causes recur at roughly three times the rate
     of engineering controls. This mirrors a well-documented pattern in quality
     systems, and it is the finding the recurrence model should rediscover.

If your real data does not contain a learnable signal, the honest outcome is a
model that does not beat the base rate - and the gauntlet in project 06 shows
you how to prove that before anyone ships it.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from src.schema import CAPA_TYPES, PROCESS_AREAS, ROOT_CAUSES, SEVERITIES, Deviation

SITES = ["Malvern PA", "Leiden NL", "Cork IE", "Singapore", "Latina IT"]
PRODUCTS = ["mAb-104 DS", "mAb-104 DP", "Vaccine-22 DP", "SmallMol-7 Tablet",
            "CellTx-3 Final Product", "Peptide-19 DS"]

# Root-cause-specific vocabulary. The classifier learns these.
NARRATIVE = {
    "Human Error": [
        "Operator performed the {step} step out of sequence; the batch record entry "
        "was completed from memory after the fact rather than contemporaneously.",
        "Second-person verification of the {step} addition was not performed. The "
        "operator signed both the performer and verifier fields.",
        "Analyst transcribed the {step} result into the worksheet incorrectly; the "
        "original instrument printout showed a different value.",
        "Technician omitted the {step} line clearance check before changeover, and "
        "the preceding lot's label reel remained on the machine.",
    ],
    "Equipment / Facility": [
        "The {equip} failed mid-cycle. Maintenance history shows the unit was "
        "operating {n} days past its scheduled preventive maintenance interval.",
        "Calibration of the {equip} was found out of tolerance at the next scheduled "
        "check; drift of {pct}% was recorded against acceptance criteria.",
        "A seal on the {equip} degraded, producing a pressure excursion during the "
        "{step} operation. The seal was not on the wear-part replacement schedule.",
        "The {equip} control loop oscillated outside the validated range for {n} "
        "minutes before the alarm annunciated.",
    ],
    "Material / Component": [
        "Incoming {mat} lot failed the identity check on retest. The supplier's CoA "
        "reported conforming results for the same lot.",
        "Stopper lot showed elevated particulate on 100% inspection; {n} units were "
        "rejected against an alert limit of {pct}%.",
        "Single-use bag from the {mat} lot developed a leak during the {step} step. "
        "Two further bags from the same lot were quarantined.",
        "The {mat} raw material was received with a shortened remaining shelf life "
        "and was used {n} days before its retest date.",
    ],
    "Procedure / Documentation": [
        "SOP for the {step} operation specified a hold time inconsistent with the "
        "validated range documented in the process description.",
        "The batch record lacked a step for {step} verification, though the process "
        "validation report required it.",
        "Two controlled documents governing the {step} operation gave conflicting "
        "acceptance criteria; the older revision remained in circulation at the line.",
        "The work instruction referenced a form revision that had been superseded "
        "{n} months earlier.",
    ],
    "Environmental / Utility": [
        "Grade A differential pressure fell below the alert limit for {n} minutes "
        "during the {step} operation in the filling suite.",
        "Environmental monitoring recovered {n} CFU at a Grade B location adjacent "
        "to the {step} operation, exceeding the action limit.",
        "WFI conductivity trended above the alert limit for {n} consecutive readings "
        "before the loop was taken out of service.",
        "HVAC recovery after the door interlock event took {n} minutes, longer than "
        "the qualified recovery time.",
    ],
    "Method / Analytical": [
        "System suitability failed on the {equip} for the {step} assay; the "
        "resolution criterion was not met across {n} consecutive injections.",
        "The assay result for the {step} test fell outside the method's validated "
        "range and was reported without an OOS investigation being raised.",
        "Reference standard used for the {step} assay was {n} days past its "
        "requalification date.",
        "Method transfer acceptance criteria for the {step} test were met at the "
        "sending site but not at the receiving site, with a bias of {pct}%.",
    ],
}

STEPS = {
    "Upstream / Bioreactor": ["media addition", "inoculation", "feed bolus", "harvest"],
    "Downstream / Purification": ["column loading", "buffer exchange", "viral filtration",
                                  "pool sampling"],
    "Fill-Finish": ["stopper placement", "line clearance", "fill weight check",
                    "lyophilisation loading"],
    "QC Analytical": ["potency", "bioburden", "endotoxin", "purity by HPLC"],
    "Warehouse / Cold Chain": ["receipt inspection", "temperature excursion review",
                               "quarantine release", "shipper conditioning"],
    "Utilities / HVAC": ["pressure cascade check", "filter integrity test",
                         "WFI loop sampling", "differential pressure trending"],
    "Packaging & Labelling": ["label reconciliation", "carton coding",
                              "serialisation check", "line clearance"],
}
EQUIP = ["autoclave AC-04", "chromatography skid CS-11", "filling line FL-02",
         "HPLC system HP-7", "lyophiliser LY-01", "bioreactor BR-2000",
         "capper CP-03", "particle counter PC-15"]
MATERIALS = ["Protein A resin", "polysorbate 80", "single-use bioreactor bag",
             "glass vial", "chlorobutyl stopper", "sucrose"]

IMMEDIATE = [
    "Line stopped and product segregated pending assessment.",
    "Affected lot quarantined; QA notified within one hour.",
    "Operation halted and the batch placed on hold.",
    "Equipment taken out of service and tagged.",
]

CAPA_TEXT = {
    "Retraining only": "Retrain the {area} operators on the affected procedure and "
                       "document completion in the training system.",
    "Procedure revision": "Revise the governing SOP to add the missing verification "
                          "step and re-issue at the point of use.",
    "Engineering control": "Install an interlock preventing the operation from "
                           "proceeding until the preceding step is confirmed complete.",
    "Supplier corrective action": "Raise a supplier corrective action request and "
                                  "increase incoming inspection to 100% for three lots.",
    "Enhanced monitoring": "Add the parameter to the continued process verification "
                           "plan and trend monthly for twelve months.",
    "No action - not required": "No CAPA required; event assessed as having no product "
                                "impact and no systemic cause.",
}

# Planted signal: base recurrence probability by CAPA type.
RECURRENCE_P = {
    "Retraining only": 0.42,
    "No action - not required": 0.38,
    "Enhanced monitoring": 0.22,
    "Procedure revision": 0.18,
    "Supplier corrective action": 0.15,
    "Engineering control": 0.09,
}

# CAPA type that quality units actually tend to choose, per root cause.
CAPA_PRIOR = {
    "Human Error": ["Retraining only"] * 6 + ["Procedure revision"] * 3 + ["Engineering control"],
    "Equipment / Facility": ["Engineering control"] * 4 + ["Enhanced monitoring"] * 3 + ["Procedure revision"] * 2 + ["Retraining only"],
    "Material / Component": ["Supplier corrective action"] * 5 + ["Enhanced monitoring"] * 3 + ["Procedure revision"] * 2,
    "Procedure / Documentation": ["Procedure revision"] * 6 + ["Retraining only"] * 3 + ["Engineering control"],
    "Environmental / Utility": ["Enhanced monitoring"] * 4 + ["Engineering control"] * 3 + ["Procedure revision"] * 2 + ["Retraining only"],
    "Method / Analytical": ["Procedure revision"] * 4 + ["Retraining only"] * 3 + ["Enhanced monitoring"] * 2 + ["No action - not required"],
}


def make(rng: random.Random, i: int, start: date) -> Deviation:
    area = rng.choice(PROCESS_AREAS)
    rc = rng.choice(ROOT_CAUSES)
    step = rng.choice(STEPS[area])
    narrative = rng.choice(NARRATIVE[rc]).format(
        step=step, equip=rng.choice(EQUIP), mat=rng.choice(MATERIALS),
        n=rng.randint(2, 96), pct=round(rng.uniform(0.4, 12.0), 1), area=area)

    severity = rng.choices(SEVERITIES, weights=[0.6, 0.32, 0.08])[0]

    # Planted label noise, with a real mechanism. Quality systems default to
    # "Human Error" when an investigation runs out of time, so a share of records
    # carry that code regardless of what the narrative describes. This is why
    # classifier accuracy on real QMS data has a ceiling set by your coding
    # consistency, not by your model.
    recorded_rc = rc
    if rc != "Human Error" and rng.random() < 0.14:
        recorded_rc = "Human Error"

    capa_type = rng.choice(CAPA_PRIOR[recorded_rc])

    p = RECURRENCE_P[capa_type]
    # Critical events get more scrutiny, and recur slightly less often.
    if severity == "Critical":
        p *= 0.7
    ec_effective = rng.random() > (p * 0.8)
    recurred = int(rng.random() < p)

    opened = start + timedelta(days=rng.randint(0, 1400))
    ec_date = opened + timedelta(days=rng.randint(120, 260))

    return Deviation(
        deviation_id=f"DEV-SYN-{i:05d}",
        date_opened=opened.isoformat(),
        site=rng.choice(SITES),
        process_area=area,
        product=rng.choice(PRODUCTS),
        severity=severity,
        description=f"During {step} in {area}, an unplanned event was identified "
                    f"by the {rng.choice(['operator', 'shift supervisor', 'QA on the floor', 'reviewer at batch record review'])}.",
        immediate_action=rng.choice(IMMEDIATE),
        investigation_narrative=narrative,
        root_cause_category=recorded_rc,
        root_cause_detail=narrative.split(".")[0] + ".",
        capa_id=f"CAPA-SYN-{i:05d}",
        capa_type=capa_type,
        capa_description=CAPA_TEXT[capa_type].format(area=area),
        effectiveness_check_date=ec_date.isoformat(),
        effectiveness_check_outcome="Effective" if ec_effective else "Not effective",
        recurred_within_365d=recurred,
        related_deviation_ids=[],
    )


def generate(n: int = 1400, seed: int = 11) -> list[Deviation]:
    """Corpus as a list of records, for the experiment harness."""
    rng = random.Random(seed)
    start = date(2021, 1, 1)
    rows = [make(rng, i, start) for i in range(n)]
    by_key: dict[tuple, list[str]] = {}
    for d in rows:
        by_key.setdefault((d.site, d.process_area, d.root_cause_category), []).append(d.deviation_id)
    for d in rows:
        if d.recurred_within_365d:
            peers = [x for x in by_key[(d.site, d.process_area, d.root_cause_category)]
                     if x != d.deviation_id]
            d.related_deviation_ids = rng.sample(peers, min(2, len(peers)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="data/deviations.jsonl")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    start = date(2021, 1, 1)
    rows = [make(rng, i, start) for i in range(a.n)]

    # link recurrences to a plausible prior event at the same site + area + cause
    by_key: dict[tuple, list[str]] = {}
    for d in rows:
        by_key.setdefault((d.site, d.process_area, d.root_cause_category), []).append(d.deviation_id)
    for d in rows:
        if d.recurred_within_365d:
            peers = [x for x in by_key[(d.site, d.process_area, d.root_cause_category)]
                     if x != d.deviation_id]
            d.related_deviation_ids = rng.sample(peers, min(2, len(peers)))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for d in rows:
            fh.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")
    rate = sum(d.recurred_within_365d for d in rows) / len(rows)
    print(f"wrote {len(rows)} SYNTHETIC deviations to {out}")
    print(f"overall 365-day recurrence rate: {rate:.1%}")
    print("note: ~14% of non-human-error events are coded 'Human Error' anyway,")
    print("      which is the coding-consistency ceiling the classifier runs into.")


if __name__ == "__main__":
    main()
