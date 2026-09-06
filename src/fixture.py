"""
A MAUDE-SHAPED FIXTURE for offline development and testing.

This is NOT data. It exists so the transformer, the recurrence derivation and
the models can be exercised and unit-tested on a machine with no access to
api.fda.gov. Records carry `report_number` values prefixed `FIXTURE-` and a
`_fixture: true` marker, and `build_corpus.py` refuses to write an analysis
corpus from fixture input unless explicitly asked.

Never report a number computed from this. Run `src/fetch_maude.py` for real data.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

# Field vocabularies below mirror the shapes openFDA returns. The values are
# plausible rather than sampled. Again, this is a fixture, not a sample.
REMEDIAL_ACTIONS = ["Recall", "Repair", "Replace", "Relabeling", "Notification",
                    "Inspection", "Modification/Adjustment", "Patient Monitoring",
                    "Other"]
PRODUCT_PROBLEMS = ["Device Alarm System", "Fluid/Blood Leak", "Battery Problem",
                    "Material Deformation", "Software Problem",
                    "Connection Problem", "Contamination", "Display Failure",
                    "Occlusion", "Calibration Problem"]
EVENT_TYPES = ["Malfunction", "Injury", "Death", "Other"]
PRODUCT_CODES = ["FRN", "LZG", "MNS", "OYC", "DXN", "KRG", "QFG", "NIQ"]
MANUFACTURERS = ["Fixture Medical Devices Inc", "Example Instruments Ltd",
                 "Placeholder Diagnostics Corp", "Sample Therapeutics LLC",
                 "Testbed Surgical Systems"]
BRANDS = ["FIXTURE PUMP 200", "EXAMPLE MONITOR X", "PLACEHOLDER ANALYSER",
          "SAMPLE INFUSION SET", "TESTBED CATHETER"]
SPECIALTIES = ["Cardiovascular", "General Hospital", "Anesthesiology",
               "Clinical Chemistry", "Radiology"]

NARRATIVE = [
    "It was reported that the {problem} occurred during use. The device was "
    "removed from service and returned to the manufacturer for evaluation. "
    "No patient harm was reported.",
    "The customer reported a {problem} approximately {n} minutes into the "
    "procedure. The procedure was completed using a backup device. "
    "Investigation is ongoing.",
    "During routine operation the user observed a {problem}. The device was "
    "taken out of service. Evaluation of the returned device confirmed the "
    "reported condition.",
    "A {problem} was identified by the clinical staff prior to use. The unit "
    "was quarantined and replaced. Analysis found the condition consistent "
    "with previously reported events.",
]
MFR_NARRATIVE = [
    "Evaluation of the returned device confirmed the reported condition. "
    "The root cause was determined to be within the {problem} subsystem. "
    "Corrective action has been initiated.",
    "The device was evaluated and no anomaly was found. The reported "
    "condition could not be reproduced under test conditions.",
    "Analysis is ongoing. A supplemental report will be filed when the "
    "evaluation is complete.",
]


def make(rng: random.Random, i: int, start: date) -> dict:
    problem = rng.choice(PRODUCT_PROBLEMS)
    code = rng.choice(PRODUCT_CODES)
    mfr = rng.choice(MANUFACTURERS)
    received = start + timedelta(days=rng.randint(0, 2200))

    return {
        "_fixture": True,
        "report_number": f"FIXTURE-{i:07d}",
        "event_type": rng.choices(EVENT_TYPES, weights=[0.72, 0.2, 0.03, 0.05])[0],
        "date_received": received.strftime("%Y%m%d"),
        "date_of_event": (received - timedelta(days=rng.randint(1, 60))).strftime("%Y%m%d"),
        "product_problems": [problem] + ([rng.choice(PRODUCT_PROBLEMS)]
                                         if rng.random() < 0.25 else []),
        "product_problem_flag": "Y",
        "remedial_action": [rng.choice(REMEDIAL_ACTIONS)],
        "manufacturer_name": mfr,
        "manufacturer_country": "US",
        "source_type": ["Manufacturer report"],
        "device": [{
            "brand_name": rng.choice(BRANDS),
            "generic_name": problem.split()[0].upper() + " DEVICE",
            "manufacturer_d_name": mfr,
            "device_report_product_code": code,
            "model_number": f"M{rng.randint(100, 999)}",
            "openfda": {"device_class": rng.choice(["2", "2", "3"]),
                        "medical_specialty_description": rng.choice(SPECIALTIES)},
        }],
        "mdr_text": [
            {"text_type_code": "Description of Event or Problem",
             "text": rng.choice(NARRATIVE).format(problem=problem.lower(),
                                                  n=rng.randint(2, 90))},
            {"text_type_code": "Additional Manufacturer Narrative",
             "text": rng.choice(MFR_NARRATIVE).format(problem=problem.lower())},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", default="data/maude_fixture.jsonl")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    start = date(2018, 1, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for i in range(a.n):
            fh.write(json.dumps(make(rng, i, start), ensure_ascii=False) + "\n")
    print(f"wrote {a.n} FIXTURE records -> {out}")
    print("These are not data. Do not report numbers computed from them.")


if __name__ == "__main__":
    main()
