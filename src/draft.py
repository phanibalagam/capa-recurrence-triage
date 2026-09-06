"""
Draft the investigation narrative and the CAPA proposal, grounded in precedent.

Every statement in the draft is tied to specific historical deviation ids. A
draft that cannot name the precedent it is reasoning from is not reviewable, and
an investigator will (rightly) throw it away.

The default composer uses no LLM at all - it assembles the draft from retrieved
precedent and model output. Pass a backend to get prose instead of structure;
the precedent block and the risk warning are appended either way, so the
reviewable content does not depend on the generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.llm import ContextBlock, get_backend

SYSTEM = """You draft pharmaceutical deviation investigation narratives for a
quality unit reviewer.

Rules:
1. Use only the precedent records supplied. Cite each as [DEV-SYN-xxxxx].
2. Never state a root cause as established. Write "the evidence is consistent
   with X" and name what would confirm or rule it out.
3. Never invent dates, lot numbers, quantities or names.
4. Keep to four short paragraphs: event, investigation to date, precedent,
   proposed CAPA with justification.
5. End with "Prepared by automated triage - requires quality unit review."."""

FOOTER = "Prepared by automated triage - requires quality unit review."

@dataclass
class TriageResult:
    deviation_id: str
    description: str
    predicted_root_causes: list[tuple[str, float]] = field(default_factory=list)
    precedent: pd.DataFrame | None = None
    precedent_summary: dict = field(default_factory=dict)
    recommended_capa: str = ""
    recurrence_risk: float = 0.0
    risk_flag: str = ""
    narrative: str = ""

    def render(self) -> str:
        L = []
        L.append(f"DEVIATION TRIAGE  -  {self.deviation_id}")
        L.append("=" * 78)
        L.append(f"\nEVENT\n  {self.description}")

        L.append("\nPREDICTED ROOT CAUSE (advisory - does not disposition)")
        for cause, p in self.predicted_root_causes:
            bar = "#" * int(round(p * 30))
            L.append(f"  {cause:<26} {p:>6.1%}  {bar}")

        if self.precedent is not None and not self.precedent.empty:
            s = self.precedent_summary
            L.append(f"\nPRECEDENT  ({s['n']} similar historical events)")
            L.append(f"  most common cause      : {s['most_common_cause']}")
            L.append(f"  most common action     : {s['most_common_action']}")
            L.append(f"  recurred within 365d   : {s['share_recurred']:.0%}")
            L.append("")
            for _, r in self.precedent.iterrows():
                L.append(f"  [{r['record_id']}] {r['date'].date()} "
                         f"{str(r['unit'])[:16]:<16} {str(r['cause_category'])[:24]:<24} "
                         f"action: {str(r['action_type'])[:22]:<22} "
                         f"{'RECURRED' if r['recurred_within_365d'] else 'held'}")

        L.append(f"\nPROPOSED CAPA\n  {self.recommended_capa}")
        L.append(f"\nRECURRENCE RISK\n  {self.recurrence_risk:.1%} probability of a "
                 f"similar event within 365 days")
        if self.risk_flag:
            L.append(f"  ** {self.risk_flag}")

        if self.narrative:
            L.append("\nDRAFT NARRATIVE\n" + "\n".join(
                "  " + ln for ln in self.narrative.splitlines()))
        L.append("\n" + "-" * 78)
        L.append(FOOTER)
        return "\n".join(L)

def compose_narrative(result: TriageResult, backend: str = "none") -> str:
    """
    backend="none" assembles a structured draft with no model. Any other backend
    generates prose from the same precedent, which is then appended to - never
    substituted for - the structured precedent block above.
    """
    if result.precedent is None or result.precedent.empty:
        return ""

    ids = result.precedent["record_id"].tolist()
    top_cause = result.predicted_root_causes[0][0] if result.predicted_root_causes else "undetermined"

    if backend == "none":
        cited = ", ".join(f"[{i}]" for i in ids[:3])
        return (
            f"The reported event is consistent with a {top_cause.lower()} root "
            f"cause. Comparable events at this site and process area were coded "
            f"the same way in {cited}.\n"
            f"Investigation to date has not established causation. Confirming "
            f"evidence would be the contemporaneous record for the affected step "
            f"and the equipment or training history covering the same period.\n"
            f"Of the {len(result.precedent)} precedent events retrieved, "
            f"{result.precedent_summary['share_recurred']:.0%} recurred within "
            f"365 days of effectiveness check closure.\n"
            f"Proposed CAPA: {result.recommended_capa}")

    llm = get_backend(backend)
    blocks = [ContextBlock(source_id=r["record_id"],
                           text=f"{r['narrative']} "
                                f"Cause coded {r['cause_category']}. "
                                f"Action was {r['action_type']}. "
                                f"{'It recurred within 365 days.' if r['recurred_within_365d'] else 'It did not recur.'}")
              for _, r in result.precedent.iterrows()]
    return llm.complete(system=SYSTEM,
                        prompt=f"Draft the investigation narrative for this event: "
                               f"{result.description}",
                        context_blocks=blocks)
