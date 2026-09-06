#!/usr/bin/env python
"""
Recurrence-aware triage - command line entry point.

Two corpora, one pipeline:

  --source maude       real openFDA device reports (CC0). Recurrence is DERIVED
                       from the data. This is what any reported result uses.
  --source synthetic   generated pharmaceutical deviations with a planted
                       signal. For learning the method, not for reporting.

    python run.py fetch                          pull MAUDE (needs api.fda.gov)
    python run.py setup --source maude           build the corpus from the cache
    python run.py setup --source synthetic       generate the teaching corpus
    python run.py capa-risk --source maude
    python run.py evaluate --source maude
    python run.py experiments --source maude --adjusted
    python run.py triage "alarm did not sound" --source maude
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_fetch(a):
    from src import fetch_maude
    if a.diagnose:
        fetch_maude.diagnose()
        return
    argv = ["fetch_maude", "--n", str(a.n), "--page-size", str(a.page_size)]
    if a.no_search:
        argv.append("--no-search")
    sys.argv = argv
    fetch_maude.main()


def cmd_setup(a):
    if a.source == "synthetic":
        from src import synth
        sys.argv = ["synth", "--n", str(a.n)]
        synth.main()
        return
    from src import build_corpus, fixture
    raw = Path(a.raw or "data/maude_raw.jsonl")
    argv = ["build_corpus", "--raw", str(raw)]
    if not raw.exists():
        if not a.fixture:
            raise SystemExit(
                f"{raw} not found.\n\n"
                "Either:\n"
                "  python run.py fetch            (needs access to api.fda.gov)\n"
                "  python run.py setup --source maude --fixture   (offline test only)")
        sys.argv = ["fixture", "--n", "3000"]
        fixture.main()
        argv = ["build_corpus", "--raw", "data/maude_fixture.jsonl", "--allow-fixture"]
    sys.argv = argv
    build_corpus.main()


def cmd_capa_risk(a):
    from src.store import load
    from src.models import RecurrenceModel
    df = load(a.source)
    t = RecurrenceModel().capa_risk_table(df, by=a.by)
    print(f"\n  OBSERVED 365-DAY RECURRENCE BY {a.by.upper()}   (source={a.source})")
    print("  " + "-" * 66)
    print(t.to_string())
    print("  " + "-" * 66)
    if len(t) >= 2:
        hi, lo = t.index[0], t.index[-1]
        ratio = t.loc[hi, "recurrence_rate"] / max(t.loc[lo, "recurrence_rate"], 0.1)
        print(f"  '{hi}' recurs {ratio:.1f}x as often as '{lo}'.")
    if a.source == "maude":
        print("\n  UNADJUSTED. On real data this table is confounded by how much a")
        print("  device gets reported. Run `python run.py experiments --adjusted`")
        print("  before quoting any of it.")


def cmd_evaluate(a):
    from src.store import load, temporal_split
    from src.models import RecurrenceModel, RootCauseClassifier
    df = load(a.source)
    tr, te = temporal_split(df)
    print(f"source={a.source}  n={len(df)}  "
          f"temporal split: train {len(tr)} ({tr.date.min().date()} -> "
          f"{tr.date.max().date()}), test {len(te)}")
    rc = RootCauseClassifier().fit(tr)
    s = rc.evaluate(te, tr)
    print("\n" + "=" * 72)
    print("  CAUSE CATEGORY CLASSIFIER  (advisory - never dispositions)")
    print("=" * 72)
    print(f"  accuracy {s.metrics['accuracy']:.3f}   macro-F1 {s.metrics['macro_f1']:.3f}")
    print(f"  most-frequent baseline: accuracy {s.baseline['accuracy']:.3f}   "
          f"macro-F1 {s.baseline['macro_f1']:.3f}")
    print(f"  GATE beats baseline: {s.beats_baseline}")
    if a.detail:
        print("\n" + rc.report(te))
    rm = RecurrenceModel().fit(tr)
    s2 = rm.evaluate(te, tr)
    print("=" * 72)
    print("  RECURRENCE MODEL  (advisory)")
    print("=" * 72)
    print(f"  ROC-AUC {s2.metrics['roc_auc']:.3f}   PR-AUC {s2.metrics['pr_auc']:.3f}"
          f"   (test base rate {s2.metrics['base_rate']:.3f})")
    print(f"  GATE ROC-AUC>0.60 and PR-AUC 15% over base rate: {s2.beats_baseline}")
    print("=" * 72)


def cmd_experiments(a):
    from src.experiments import (leave_one_group_out, ablation, adjusted, repeated,
                                 stratified, record_level_replication)
    if a.record_level:
        record_level_replication(a.source, n_boot=max(a.n_boot, 800))
    elif a.logo:
        leave_one_group_out(a.source)
    elif a.stratified:
        stratified(a.source, n_boot=a.n_boot)
    elif a.adjusted:
        adjusted(a.source, n_boot=a.n_boot)
    elif a.ablation:
        ablation(a.source, seeds=a.seeds)
    else:
        repeated(a.source, seeds=a.seeds,
                 n_boot=min(a.n_boot, 60) if a.source == "maude" else a.n_boot,
                 fit_classifier=not a.fast)


def cmd_triage(a):
    from src.store import load, temporal_split
    from src.triage import TriageSystem
    df = load(a.source)
    tr, _ = temporal_split(df)
    ts = TriageSystem(tr)
    r = ts.triage(a.description,
                  site=a.unit or tr["unit"].mode().iat[0],
                  process_area=a.area or tr["area"].mode().iat[0],
                  severity=a.severity or tr["severity"].mode().iat[0],
                  k=a.k, narrative_backend=a.backend, deviation_id=a.id)
    print("\n" + r.render())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def src(p):
        p.add_argument("--source", default="maude", choices=["synthetic", "maude"])
        return p

    s = sub.add_parser("fetch", help="download MAUDE reports from openFDA")
    s.add_argument("--n", type=int, default=4000, help="reports per year")
    s.add_argument("--diagnose", action="store_true",
                   help="climb from the simplest request to the real one")
    s.add_argument("--no-search", action="store_true")
    s.add_argument("--page-size", type=int, default=1000)
    s.set_defaults(func=cmd_fetch)

    s = src(sub.add_parser("setup", help="build the analysis corpus"))
    s.add_argument("--n", type=int, default=1400, help="synthetic corpus size")
    s.add_argument("--raw", default=None)
    s.add_argument("--fixture", action="store_true",
                   help="build from the offline test fixture (not data)")
    s.set_defaults(func=cmd_setup)

    s = src(sub.add_parser("capa-risk", help="observed recurrence by action"))
    s.add_argument("--by", default="action_type",
                   choices=["action_type", "action_class"])
    s.set_defaults(func=cmd_capa_risk)

    s = src(sub.add_parser("evaluate", help="both models on a temporal split"))
    s.add_argument("--detail", action="store_true")
    s.set_defaults(func=cmd_evaluate)

    s = src(sub.add_parser("experiments", help="repeated runs, adjustment, ablation"))
    s.add_argument("--seeds", type=int, default=10)
    s.add_argument("--n-boot", type=int, default=200)
    s.add_argument("--adjusted", action="store_true")
    s.add_argument("--stratified", action="store_true",
                   help="test the device-withdrawal alternative explanation")
    s.add_argument("--ablation", action="store_true")
    s.add_argument("--logo", action="store_true",
                   help="leave-one-group-out sensitivity of the adjusted odds ratio")
    s.add_argument("--record-level", action="store_true",
                   help="regenerate the superseded record-level estimates the "
                        "paper reports as errors")
    s.add_argument("--fast", action="store_true",
                   help="skip the expensive per-bootstrap classifier refit")
    s.set_defaults(func=cmd_experiments)

    s = src(sub.add_parser("triage", help="triage one new record"))
    s.add_argument("description")
    s.add_argument("--unit", default=None)
    s.add_argument("--area", default=None)
    s.add_argument("--severity", default=None)
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--id", default="NEW")
    s.add_argument("--backend", default="none",
                   choices=["none", "extractive", "anthropic", "openai", "ollama"])
    s.set_defaults(func=cmd_triage)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
