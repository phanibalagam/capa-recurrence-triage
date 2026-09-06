"""
Experiment harness. Runs on either corpus; the numbers that get reported come
from the real one.

    python -m src.experiments --source maude
    python -m src.experiments --source maude --adjusted
    python -m src.experiments --source synthetic --ablation

WHAT EACH EXPERIMENT ANSWERS

  repeated()  How stable is the headline? On the synthetic corpus the corpus
              itself is regenerated per seed. On MAUDE the corpus is fixed and
              real, so uncertainty comes from bootstrap resampling of the
              records instead. Both report mean with a 95% interval.

  adjusted()  THE ONE THAT MATTERS ON REAL DATA. The unadjusted association
              between remedial action and recurrence is confounded: a device
              that gets reported more often is mechanically more likely to be
              reported again, whatever action was taken. This fits the
              association with and without group-size adjustment and reports
              both, so the reader can see how much of the raw effect survives.

  ablation()  Which feature group carries the recurrence signal.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from src.models import RecurrenceModel, RootCauseClassifier
from src.store import load, temporal_split

RESULTS = Path("results")

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 19: 2.093, 24: 2.064, 29: 2.045, 49: 2.010}


def _t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96


def summarise(values: list[float]) -> dict:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    n = len(vals)
    if n == 0:
        return {"mean": None, "ci95_halfwidth": None, "sd": None, "n": 0}
    m = statistics.fmean(vals)
    if n < 2:
        return {"mean": round(m, 4), "ci95_halfwidth": None, "sd": None, "n": n}
    sd = statistics.stdev(vals)
    return {"mean": round(m, 4), "ci95_halfwidth": round(_t95(n - 1) * sd / math.sqrt(n), 4),
            "sd": round(sd, 4), "n": n}


def _percentile_ci(vals: list[float]) -> dict:
    a = np.asarray([v for v in vals if np.isfinite(v)])
    if a.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    return {"mean": round(float(a.mean()), 4),
            "lo": round(float(np.percentile(a, 2.5)), 4),
            "hi": round(float(np.percentile(a, 97.5)), 4), "n": int(a.size)}


def _frames(source: str, seeds: int, n_boot: int):
    """Synthetic: a fresh corpus per seed. MAUDE: bootstrap the real corpus."""
    if source == "synthetic":
        from src.synth import generate
        out = []
        for s in range(seeds):
            rows = [d.__dict__ for d in generate(n=1400, seed=300 + s)]
            df = pd.DataFrame(rows)
            df["date_opened"] = pd.to_datetime(df["date_opened"])
            tmp = Path("data/_seed_tmp.jsonl")
            tmp.write_text("\n".join(json.dumps(r, default=str) for r in rows),
                           encoding="utf-8")
            out.append(load("synthetic", path=str(tmp)))
        tmp.unlink(missing_ok=True)
        return out, "regenerated corpora"
    base = load("maude")
    rng = np.random.default_rng(0)
    return ([base.sample(frac=1.0, replace=True, random_state=int(rng.integers(1e9)))
             .sort_values("date").reset_index(drop=True) for _ in range(n_boot)],
            "bootstrap resamples of the real corpus")


# --------------------------------------------------------------------------

def repeated(source: str = "maude", seeds: int = 10, n_boot: int = 60,
             verbose: bool = True, fit_classifier: bool = True) -> dict:
    """
    n_boot defaults to 60 on real data: each resample refits a multinomial
    classifier over hundreds of problem categories, which is the expensive part.
    Pass fit_classifier=False to skip it and bootstrap only the cheap estimates.
    """
    frames, kind = _frames(source, seeds, n_boot)
    base = load(source)

    acc, auc, ratio, worst, best = [], [], [], [], []
    for df in frames:
        tr, te = temporal_split(df)
        if te["cause_category"].nunique() < 2 or tr["cause_category"].nunique() < 2:
            continue
        if fit_classifier:
            try:
                acc.append(RootCauseClassifier().fit(tr).evaluate(te, tr).metrics["accuracy"])
            except Exception:
                pass
        try:
            auc.append(RecurrenceModel().fit(tr).evaluate(te, tr).metrics["roc_auc"])
        except Exception:
            pass
        t = df.groupby("action_class")["recurred_within_365d"].agg(["size", "mean"])
        t = t[t["size"] >= 30].sort_values("mean", ascending=False)
        if len(t) >= 2:
            hi, lo = t["mean"].iat[0] * 100, t["mean"].iat[-1] * 100
            worst.append(hi); best.append(lo); ratio.append(hi / max(lo, 0.1))

    out = {
        "source": source, "uncertainty_from": kind,
        "n_records": int(len(base)),
        "cause_accuracy": summarise(acc),
        "recurrence_roc_auc": summarise(auc),
        "worst_action_class_pct": _percentile_ci(worst),
        "best_action_class_pct": _percentile_ci(best),
        "recurrence_ratio": _percentile_ci(ratio),
    }

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"  REPEATED EVALUATION  ·  source={source}  ·  {kind}")
        print("=" * 80)
        print(f"  records in analysis set        {out['n_records']}")
        for k in ["cause_accuracy", "recurrence_roc_auc"]:
            v = out[k]
            ci = f" ± {v['ci95_halfwidth']}" if v.get("ci95_halfwidth") else ""
            print(f"  {k:<30} {v['mean']}{ci}")
        for k in ["worst_action_class_pct", "best_action_class_pct", "recurrence_ratio"]:
            v = out[k]
            print(f"  {k:<30} {v['mean']}  [{v['lo']}, {v['hi']}]")
        print("=" * 80)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"repeated_{source}.json").write_text(json.dumps(
        {"experiment": "repeated_evaluation", "results": out}, indent=2))
    return out


# --------------------------------------------------------------------------

# Backward-looking only. See src/build_corpus.py for why group_size is excluded:
# it counts future members of the group and so encodes the outcome.
ADJ_CONTROLS = ["n_prior_in_group", "prior_365"]
# The control set of the FIRST, WRONG adjustment, kept so that estimate stays
# reproducible. group_size is the leaky term: it counts every member of a group
# including reports that arrive after the index record.
LEAKY_CONTROLS = ["group_size", "n_prior_in_group"]

# The competing explanation this module exists to test.
#
# A recall REMOVES DEVICES FROM SERVICE. A withdrawn device cannot generate
# further reports, so "fewer subsequent reports after a recall" is not evidence
# that the action worked - it may be evidence that the devices are gone. Recall
# sits inside the Engineering/design class, so it could be carrying the entire
# headline association on its own.
#
# Splitting engineering into withdrawal and in-service actions tests that
# directly. If the association holds among actions that leave the device in the
# field, mechanical removal cannot be the explanation.
WITHDRAWAL_ACTIONS = ["Recall"]
IN_SERVICE_ACTIONS = ["Repair", "Replace", "Modification/Adjustment"]


def adjusted(source: str = "maude", n_boot: int = 400, verbose: bool = True) -> dict:
    """
    Association between action class and recurrence, unadjusted and adjusted
    for how much the group gets reported.

    Reported as an odds ratio for "Engineering / design" against
    "Communication", the contrast the project is actually about.
    """
    df = load(source).copy()
    df = df[df["action_class"].isin(["Engineering / design", "Communication"])].copy()
    if df["action_class"].nunique() < 2:
        raise SystemExit("need both action classes present")
    y = df["recurred_within_365d"].to_numpy()
    is_eng = (df["action_class"] == "Engineering / design").to_numpy().astype(float)

    # Three fits, deliberately. The leaky one is computed and reported because
    # the contrast between it and the valid one is the study's main lesson:
    # group_size counts reports arriving after the index record, so adjusting
    # for it conditions on the outcome and produces a defensible-looking number.
    ARMS = {"unadjusted": [],
            "adjusted": ADJ_CONTROLS,
            "leaky_adjusted": LEAKY_CONTROLS}

    def _or(rows, controls) -> float:
        yy, ee = y[rows], is_eng[rows]
        if len(np.unique(yy)) < 2 or len(np.unique(ee)) < 2:
            return float("nan")
        X = ee.reshape(-1, 1)
        if controls:
            ctrl = df.iloc[rows][controls].to_numpy(dtype=float)
            ctrl = np.log1p(ctrl)          # heavy-tailed counts
            X = np.hstack([X, ctrl])
        m = LogisticRegression(max_iter=2000).fit(X, yy)
        return float(np.exp(m.coef_[0][0]))

    all_rows = np.arange(len(df))
    point = {k: _or(all_rows, c) for k, c in ARMS.items()}

    rng = np.random.default_rng(0)
    boots = {k: [] for k in ARMS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        for k, c in ARMS.items():
            boots[k].append(_or(idx, c))

    out = {"source": source, "n": int(len(df)),
           "contrast": "Engineering / design vs Communication",
           "outcome": "recurrence within 365 days",
           "controls": ADJ_CONTROLS,
           "unadjusted_odds_ratio": {**_percentile_ci(boots["unadjusted"]),
                                     "point": round(point["unadjusted"], 4)},
           "adjusted_odds_ratio": {**_percentile_ci(boots["adjusted"]),
                                   "point": round(point["adjusted"], 4)},
           "leaky_controls": LEAKY_CONTROLS,
           "leaky_adjusted_odds_ratio": {
               **_percentile_ci(boots["leaky_adjusted"]),
               "point": round(point["leaky_adjusted"], 4),
               "warning": ("INVALID. group_size counts reports that arrive "
                           "after the index record, so this conditions on the "
                           "outcome. Reported only as the contrast that "
                           "identified the error.")}}

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"  ADJUSTED ASSOCIATION  ·  source={source}  ·  n={out['n']}")
        print("=" * 80)
        print(f"  contrast   {out['contrast']}")
        print(f"  outcome    {out['outcome']}")
        print(f"  controls   {', '.join(ADJ_CONTROLS)} (log1p)\n")
        for k in ["unadjusted_odds_ratio", "leaky_adjusted_odds_ratio",
                  "adjusted_odds_ratio"]:
            v = out[k]
            tag = "  [LEAKY - DO NOT QUOTE]" if k.startswith("leaky") else ""
            print(f"  {k:<30} OR {v['point']:.3f}   bootstrap 95% "
                  f"[{v['lo']}, {v['hi']}]{tag}")
        print("-" * 80)
        u, a = out["unadjusted_odds_ratio"], out["adjusted_odds_ratio"]
        crosses = a["lo"] is not None and a["lo"] <= 1.0 <= a["hi"]
        print(f"  An odds ratio below 1 means engineering-type actions are followed by")
        print(f"  FEWER recurrences than communication-type actions.")
        if crosses:
            print(f"  The adjusted interval CROSSES 1: after adjusting for how much a")
            print(f"  group gets reported, the association is not distinguishable from")
            print(f"  no effect. Report that, not the unadjusted number.")
        else:
            print(f"  The adjusted interval excludes 1, so the association survives")
            print(f"  adjustment for reporting volume.")
        print("=" * 80)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"adjusted_{source}.json").write_text(json.dumps(
        {"experiment": "adjusted_association", "results": out}, indent=2))
    return out


# --------------------------------------------------------------------------

def _adjusted_or(df: pd.DataFrame, exposed_mask: np.ndarray,
                 n_boot: int = 400) -> dict:
    """Adjusted odds ratio for `exposed_mask` vs the rest of `df`."""
    y = df["recurred_within_365d"].to_numpy()
    e = exposed_mask.astype(float)
    ctrl = np.log1p(df[ADJ_CONTROLS].to_numpy(dtype=float))

    def fit(rows) -> float:
        yy, ee, cc = y[rows], e[rows], ctrl[rows]
        if len(np.unique(yy)) < 2 or len(np.unique(ee)) < 2:
            return float("nan")
        X = np.hstack([ee.reshape(-1, 1), cc])
        m = LogisticRegression(max_iter=2000).fit(X, yy)
        return float(np.exp(m.coef_[0][0]))

    point = fit(np.arange(len(df)))
    rng = np.random.default_rng(0)
    boots = [fit(rng.integers(0, len(df), len(df))) for _ in range(n_boot)]
    return {**_percentile_ci(boots),
            "point": None if not np.isfinite(point) else round(point, 4)}


def stratified(source: str = "maude", n_boot: int = 400,
               verbose: bool = True) -> dict:
    """
    Does the association survive when the device stays in service?

    Three contrasts, each against Communication actions, each adjusted for
    reporting volume:

      all engineering        the headline
      withdrawal (recall)    device leaves the field
      in service             repair, replace, modification

    The third is the one that matters. If it holds, the finding is about the
    action. If only the second holds, we measured device withdrawal.
    """
    df = load(source).copy()
    comm = df["action_class"] == "Communication"
    eng = df["action_class"] == "Engineering / design"
    withdrawal = df["action_type"].isin(WITHDRAWAL_ACTIONS)
    in_service = df["action_type"].isin(IN_SERVICE_ACTIONS)

    contrasts = {
        "all engineering vs communication": eng,
        "withdrawal (recall) vs communication": withdrawal,
        "in service (repair/replace/modify) vs communication": in_service,
    }

    out = {"source": source, "controls": ADJ_CONTROLS, "contrasts": {}}
    for label, mask in contrasts.items():
        sub = df[mask | comm].copy()
        n_exp = int(mask.sum())
        if n_exp < 20 or len(sub) < 60:
            out["contrasts"][label] = {"n_exposed": n_exp, "n_total": int(len(sub)),
                                       "skipped": "too few exposed records"}
            continue
        res = _adjusted_or(sub, (sub["action_class"] != "Communication").to_numpy(),
                           n_boot=n_boot)
        rec_exp = float(df[mask]["recurred_within_365d"].mean())
        out["contrasts"][label] = {
            "n_exposed": n_exp, "n_comparison": int(comm.sum()),
            "recurrence_exposed": round(rec_exp, 4),
            "recurrence_comparison": round(float(df[comm]["recurred_within_365d"].mean()), 4),
            "adjusted_odds_ratio": res,
            # An interval can exclude 1 in either direction. `protective` is the
            # hypothesised direction (fewer recurrences); `harmful` is the
            # opposite and is just as much a finding.
            "protective": bool(res["hi"] is not None and res["hi"] < 1.0),
            "harmful": bool(res["lo"] is not None and res["lo"] > 1.0),
            "excludes_1": bool((res["hi"] is not None and res["hi"] < 1.0)
                               or (res["lo"] is not None and res["lo"] > 1.0)),
        }

    if verbose:
        print(f"\n{'=' * 84}")
        print(f"  DOES IT SURVIVE WHEN THE DEVICE STAYS IN SERVICE?  ·  source={source}")
        print("=" * 84)
        print("  A recall removes devices from the field, so fewer later reports may")
        print("  mean the devices are gone rather than the action worked. Splitting")
        print("  engineering into withdrawal and in-service actions tests that.\n")
        SHORT = {"all engineering vs communication": "all engineering",
                 "withdrawal (recall) vs communication": "withdrawal (recall)",
                 "in service (repair/replace/modify) vs communication":
                     "in service (repair/replace/modify)"}
        print(f"  {'contrast (vs communication)':<38}{'n':>7}{'recur':>8}"
              f"{'adjusted OR (95% CI)':>24}{'':>4}")
        print("-" * 84)
        for label, v in out["contrasts"].items():
            short = SHORT.get(label, label)[:37]
            if "skipped" in v:
                print(f"  {short:<38}{v['n_exposed']:>7}   -- {v['skipped']}")
                continue
            o = v["adjusted_odds_ratio"]
            ci = f"{o['point']:.3f} [{o['lo']:.3f}, {o['hi']:.3f}]"
            tag = ("fewer" if v["protective"] else
                   "MORE" if v["harmful"] else "n.s.")
            print(f"  {short:<38}{v['n_exposed']:>7}{v['recurrence_exposed']:>8.1%}"
                  f"{ci:>24}{tag:>6}")
        print("-" * 84)
        ins = out["contrasts"].get("in service (repair/replace/modify) vs communication", {})
        wd = out["contrasts"].get("withdrawal (recall) vs communication", {})
        if "skipped" in ins:
            print("  The in-service contrast could not be estimated - too few records.")
            print("  Without it, device withdrawal remains a live alternative")
            print("  explanation and the headline must carry that caveat.")
        elif ins.get("protective"):
            print("  The in-service contrast is protective and excludes 1. Devices that")
            print("  STAYED IN THE FIELD still saw fewer recurrence reports, so")
            print("  mechanical removal cannot explain the association.")
            print("  This is the result to report.")
        elif ins.get("harmful"):
            print("  The in-service contrast runs the OPPOSITE way and excludes 1:")
            print("  in-service engineering actions are followed by MORE recurrence")
            print("  reports than communication actions. Whatever the headline was")
            print("  measuring, it was not in-service action effectiveness.")
        else:
            print("  The in-service contrast does not exclude 1 in either direction.")
            print("  Once recalls are removed the association is not distinguishable")
            print("  from no effect, so the honest reading is that device withdrawal")
            print("  drove the headline, and it should be reported as such.")
        if wd.get("protective") and not ins.get("protective"):
            print("  Recall alone IS protective, which is what removal would look like.")
        print("=" * 84)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"stratified_{source}.json").write_text(json.dumps(
        {"experiment": "withdrawal_vs_in_service", "results": out}, indent=2))
    return out


def ablation(source: str = "maude", seeds: int = 5, verbose: bool = True) -> list[dict]:
    df = load(source)
    tr, te = temporal_split(df)
    groups = {
        "all features": RecurrenceModel.FEATURES,
        "- action_type": [f for f in RecurrenceModel.FEATURES if f != "action_type"],
        "- cause_category": [f for f in RecurrenceModel.FEATURES if f != "cause_category"],
        "- severity": [f for f in RecurrenceModel.FEATURES if f != "severity"],
        "- unit": [f for f in RecurrenceModel.FEATURES if f != "unit"],
        "- area": [f for f in RecurrenceModel.FEATURES if f != "area"],
        "action_type only": ["action_type"],
    }
    if source == "maude":
        groups["+ prior counts (adjusted)"] = RecurrenceModel.ADJUSTED
        groups["prior_365 only"] = ["prior_365"]
        # Kept deliberately, and labelled, because the contrast is the lesson:
        # a covariate that peeks at the future scores near 1.0 and means nothing.
        groups["group_size only  [LEAKY]"] = ["group_size"]

    rows = []
    for name, feats in groups.items():
        aucs = []
        for s in range(seeds):
            m = RecurrenceModel()
            m.FEATURES = feats
            try:
                m.fit(tr)
                aucs.append(m.evaluate(te, tr).metrics["roc_auc"])
            except Exception:
                pass
        rows.append({"configuration": name, "n_features": len(feats),
                     "roc_auc": summarise(aucs)})

    if verbose:
        base = rows[0]["roc_auc"]["mean"]
        print(f"\n{'=' * 80}")
        print(f"  RECURRENCE FEATURE ABLATION  ·  source={source}")
        print("=" * 80)
        print(f"  {'configuration':<28}{'features':>10}{'ROC-AUC':>14}{'Δ':>12}")
        print("-" * 80)
        for r in rows:
            v = r["roc_auc"]
            d = "" if r is rows[0] or v["mean"] is None else f"{v['mean'] - base:+.3f}"
            print(f"  {r['configuration']:<28}{r['n_features']:>10}"
                  f"{(v['mean'] if v['mean'] is not None else float('nan')):>14.3f}{d:>12}")
        print("=" * 80)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"ablation_{source}.json").write_text(json.dumps(
        {"experiment": "recurrence_feature_ablation", "source": source,
         "rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="maude", choices=["synthetic", "maude"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--fast", action="store_true",
                    help="skip the expensive per-bootstrap classifier refit")
    ap.add_argument("--adjusted", action="store_true")
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    a = ap.parse_args()
    if a.stratified:
        stratified(a.source, n_boot=max(a.n_boot, 200))
    elif a.adjusted:
        adjusted(a.source, n_boot=max(a.n_boot, 200))
    elif a.ablation:
        ablation(a.source, seeds=max(a.seeds // 2, 3))
    else:
        repeated(a.source, seeds=a.seeds,
                 n_boot=min(a.n_boot, 60) if a.source == "maude" else a.n_boot,
                 fit_classifier=not a.fast)
