"""
The canonical record, and how to map a source onto it.

TWO SOURCES MAP ONTO ONE ANALYSIS SCHEMA (see src/store.py):

  common          synthetic QMS record      openFDA MAUDE report
  --------------  ------------------------  ---------------------------
  record_id       deviation_id              report_number
  date            date_opened               date_received
  cause_category  root_cause_category       product_problems[0]
  action_type     capa_type                 remedial_action[0]
  unit            site                      manufacturer_name
  area            process_area              device[0].device_report_product_code
  severity        severity                  event_type
  narrative       investigation_narrative   mdr_text[].text

The definitions below describe the pharmaceutical QMS record. For the MAUDE
mapping and the derived fields, see DATA_CARD.md.

Everything in this project reads this schema. Swapping synthetic data for real
data is a mapping exercise, not a rewrite - see MAPPING below.

The field that carries the most value and is most often missing is
`recurred_within_365d` (and the `related_deviation_ids` that justify it). The
link between a CAPA and its later recurrence is the asset. Build that linkage
even if you build nothing else in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ROOT_CAUSES = [
    "Human Error",
    "Equipment / Facility",
    "Material / Component",
    "Procedure / Documentation",
    "Environmental / Utility",
    "Method / Analytical",
]

CAPA_TYPES = [
    "Retraining only",
    "Procedure revision",
    "Engineering control",
    "Supplier corrective action",
    "Enhanced monitoring",
    "No action - not required",
]

SEVERITIES = ["Minor", "Major", "Critical"]

PROCESS_AREAS = [
    "Upstream / Bioreactor",
    "Downstream / Purification",
    "Fill-Finish",
    "QC Analytical",
    "Warehouse / Cold Chain",
    "Utilities / HVAC",
    "Packaging & Labelling",
]

FIELDS = [
    "deviation_id", "date_opened", "site", "process_area", "product",
    "severity", "description", "immediate_action", "investigation_narrative",
    "root_cause_category", "root_cause_detail", "capa_id", "capa_type",
    "capa_description", "effectiveness_check_date",
    "effectiveness_check_outcome", "recurred_within_365d",
    "related_deviation_ids",
]

# Typical source fields in common QMS platforms. Verify against your own
# configuration - these are conventional names, not a guarantee.
MAPPING = {
    "deviation_id": "TrackWise: PR_NUMBER | Veeva QMS: quality_event__v.name__v",
    "date_opened": "TrackWise: DATE_OPENED | Veeva: date_opened__v",
    "process_area": "usually a site-configured picklist; may need a lookup table",
    "severity": "TrackWise: SEVERITY | Veeva: classification__v",
    "description": "the free-text event description at intake",
    "investigation_narrative": "the investigation summary / RCA write-up",
    "root_cause_category": "the coded root cause picklist - your label column",
    "capa_type": "CAPA action classification picklist",
    "effectiveness_check_outcome": "EC result field, usually Effective / Not effective",
    "recurred_within_365d": "DERIVED. Not a field in any QMS. Compute it: a later "
                            "deviation at the same site, same process area, same "
                            "root cause category, within 365 days of EC closure.",
}


@dataclass
class Deviation:
    deviation_id: str
    date_opened: str
    site: str
    process_area: str
    product: str
    severity: str
    description: str
    immediate_action: str = ""
    investigation_narrative: str = ""
    root_cause_category: str = ""
    root_cause_detail: str = ""
    capa_id: str = ""
    capa_type: str = ""
    capa_description: str = ""
    effectiveness_check_date: str = ""
    effectiveness_check_outcome: str = ""
    recurred_within_365d: int = 0
    related_deviation_ids: list[str] = field(default_factory=list)

    def searchable(self) -> str:
        return " ".join([self.process_area, self.product, self.description,
                         self.investigation_narrative, self.root_cause_detail])


if __name__ == "__main__":
    print("Canonical fields:")
    for f in FIELDS:
        print(f"  {f}")
    print("\nMapping notes:")
    for k, v in MAPPING.items():
        print(f"  {k:<28} {v}")
