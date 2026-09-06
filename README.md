# capa-recurrence-triage

Does the *kind* of corrective action taken after a failure predict whether the
same failure comes back? And can that be surfaced to whoever is writing the
action, rather than in a metrics review a year later?

Standalone project, with no dependency on its siblings.

## Two corpora, one pipeline

| | `--source maude` | `--source synthetic` |
|---|---|---|
| Data | **Real**: openFDA device reports, CC0 | Generated |
| Recurrence | **Derived from the data** | Planted |
| Use for | Anything published | Learning the method |

No public *pharmaceutical* deviation corpus exists and none will. But FDA's MAUDE
device database has the same shape (free-text narrative, coded problem, recorded
remedial action) and it is public domain. The structural analogy is exact:

```
deviation  ->  root cause  ->  CAPA        (pharmaceutical quality)
MDR report ->  product problem -> remedial action   (MAUDE)
```

The outcome neither database records is *whether it came back*. Here it is
**derived**: another report with the same product code, manufacturer and problem
within 365 days. That derivation is what turns this from a demonstration on
planted data into an observational study on real data, and it is the whole
reason the project was rebuilt.

See `DATA_CARD.md` for provenance, field-by-field, and for what MAUDE's limits
mean for any claim.

## Three functions, increasing risk

| Function | Risk band | Ships in |
|---|---|---|
| Retrieve similar historical deviations and their dispositions | advisory | a quarter |
| Draft the investigation narrative and proposed CAPA | advisory | a quarter |
| Flag CAPAs likely to be followed by recurrence | advisory, watch closely | +3 months |

Nothing here dispositions anything. The models advise; the quality unit signs.

## Run it

```
python run.py fetch                              # needs access to api.fda.gov
python run.py setup   --source maude
python run.py capa-risk --source maude --by action_class
python run.py evaluate  --source maude
python run.py experiments --source maude --adjusted     # the one that matters
python run.py triage "alarm did not sound during the procedure" --source maude
```

Without network access, `python run.py setup --source maude --fixture` builds
from a MAUDE-shaped test fixture so every code path runs. The fixture contains no
signal, so the recurrence gate **fails** on it. That is the correct result, and a
check that the gate is not rigged.

The synthetic corpus still works exactly as before:

```
python run.py setup --source synthetic
python run.py capa-risk --source synthetic
```

## What it finds: nothing, and that is the result

On 4,192 real openFDA device reports (2018-2024), the unadjusted table looks like
a finding:

| Action class | n | Recurrence |
|---|---|---|
| Communication | 311 | 64.0% |
| Other | 242 | 33.5% |
| Engineering / design | 3,628 | 21.8% |
| Surveillance | 11 | 9.1% |

Communication-type actions followed by recurrence **2.93x** as often as
engineering-type ones, 95% CI [2.58, 3.27]. The direction is the one the project
hypothesised and the interval is tight, and it is an exact analogue of
"retraining versus engineering control".

It does not survive adjustment.

| Estimate | Odds ratio | 95% CI |
|---|---|---|
| Unadjusted | 0.161 | [0.125, 0.204] |
| Adjusted for `group_size` **(invalid)** | 0.258 | [0.184, 0.353] |
| **Adjusted, backward-looking counts** | **1.050** | **[0.810, 1.407]** |

**There is no association once reporting volume is properly controlled.**

### The middle row is the interesting one

The first adjustment used `group_size`, which counts every report in a record's
group *including later ones*. It encodes the outcome. Conditioning on it does not
remove confounding; it conditions on the answer, and it produced a comfortable
0.258 that would have been published.

The ablation caught it: `group_size` alone predicts recurrence at **ROC-AUC
0.993**, against 0.536 for the action type. No categorical covariate does that.
The implausibility was the tell, not any suspicion about the adjustment.

Doing the right analysis incorrectly is more dangerous than skipping it, because
the output carries the signature of rigour.

### Two more things the checks found

**Stratification cannot rescue it.** A recall removes devices from service, so
fewer later reports may mean the devices are gone. Splitting engineering by
whether the device stays in the field: none of the three contrasts excludes 1, and
**98.4% of engineering records are recalls** (3,571 of 3,628), so the class is
effectively a recall indicator. The in-service arm is n=57.

**The recurrence model's 0.975 AUC is circular.** Recurrence is defined within
groups keyed on (product code, manufacturer, problem), and three of five model
features are those fields. The model identifies the group; whether a group has
later reports is a property of the group. `action_type` alone scores 0.536.

Full numbers in `METHODS.md` and `results/`.

## About the data

`data/deviations.jsonl` is **synthetic**. Ids are prefixed `DEV-SYN-` so no record
can be mistaken for a real quality event. Two signals are planted deliberately, so
the models have something real to learn: root cause is inferable from narrative
vocabulary, and recurrence depends on CAPA type.

## Wiring in your QMS

`src/schema.py` is the contract, and it prints the mapping notes:

```
python -m src.schema
```

Most fields map straight across from TrackWise or Veeva QMS. One does not:

> **`recurred_within_365d` is not a field in any QMS.** You derive it: a later
> deviation at the same site, same process area, same root cause category, within
> 365 days of effectiveness-check closure.

That derivation is the asset. Build the CAPA-to-recurrence linkage even if you
build nothing else in this repository. Every useful thing above depends on it,
and it is a SQL job, not a modelling project.

## Before this touches a regulated process

Write the context-of-use statement first, per FDA's risk-based credibility
framework: one page, at kickoff. It states the specific question the model
answers and the consequence of it being wrong, and it determines the whole
validation burden. It is much cheaper to write before the model exists.

The trap is starting in the advisory band and drifting into decision support
without re-validating. Declare the band, and make a band change a formal change.

---

## Standalone by design

This project has no dependency on its sibling projects. Everything it needs is
in this directory, including its own copy of the pluggable model backend
(`src/llm.py`). Copy the folder anywhere and it runs.

That duplication is deliberate. Each project is meant to be redistributable on
its own, as a repository, a post, or an attachment, and a shared utility package
would make that impossible without also shipping the other two.

## Licence and provenance

Code in this directory: MIT (see `LICENSE`).

Data: see the "About the data" section above for what is real and what is
generated, and under which licence each part may be redistributed.

Originally candidate 13 in a fourteen-candidate portfolio assessment; renumbered
sequentially here because these three were the ones built.
