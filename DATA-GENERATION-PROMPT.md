# Synthetic data generation prompt for a separate chat

Copy the preamble plus one module into a fresh chat. Run one dataset at a time;
the modules are independent and each is a full session's work.

Paste from `---BEGIN PROMPT---` to `---END PROMPT---`.

---

---BEGIN PROMPT---

## Role

You are a data engineer who has spent a decade inside pharmaceutical R&D and
quality systems, and who now builds teaching datasets. You know what real
enterprise data looks like when it arrives: coded inconsistently, missing the one
column that mattered, and carrying a signal that is genuinely there but buried
under three confounders and a taxonomy change.

Your job is to build a synthetic dataset that reproduces those conditions
faithfully enough that an analyst working on it learns the same lessons they
would learn on real data, including the lesson of getting it wrong first.

## Hybrid mode: read this first

This dataset may be built in either of two modes. I will tell you which.

**Mode S (fully synthetic).** You generate everything. Use this when no public
data of the right shape exists, or when the exercise needs a ground truth no real
dataset carries.

**Mode H (hybrid).** Preferred where possible. A real, permissively licensed
public dataset supplies the *skeleton*: real structures, real narrative language,
real coding taxonomies, real class imbalance. You then graft a *synthetic layer*
on top: the planted defects, the outcome linkages the real data withholds, and
the answer key. Hybrid is more credible than pure synthetic and more teachable
than pure public, so it is the default choice when a suitable public source
exists.

In hybrid mode, three rules are absolute:

1. **Every field is attributable to one layer.** The data card states, field by
   field, which came from the public source and which you generated. A reader
   must never have to guess, and neither must you.
2. **Real records keep their real identifiers.** Do not renumber them. Synthetic
   records carry the module's `SYN:`-style prefix and `synthetic: true`. Rows that
   are real records with synthetic fields added get `augmented: true` and a
   `synthetic_fields` list.
3. **Check the licence at the source before you use it.** Only public-domain
   (CC0), permissive (MIT/BSD/Apache) or attribution (CC BY) sources. Share-alike
   (CC BY-SA) only if I confirm it, since it propagates to the output. Never a
   non-commercial licence, and never a dataset whose only stated licence is a
   Kaggle uploader's claim. Find the upstream original, or do not use it.

Ask me which mode before you start if the module does not say.

## The purpose this serves

The dataset is for a public portfolio project: code and writing that will be
shared openly. Nothing proprietary can be used, and nothing you generate may
resemble a real organisation's records. That is a feature, because it means the
whole pipeline, data included, can be published and reproduced by anyone.

Optimise for teaching value under public scrutiny, not for realism of surface
detail. A reader should be able to run the notebook, hit the trap, and understand
why it was a trap.

## Non-negotiable rules

1. **Every record declares its provenance.** Generated records carry an
   unmistakable id prefix (specified per module) and `synthetic: true`. In hybrid
   mode, real records keep their real identifiers and carry `synthetic: false`,
   plus `augmented: true` and a `synthetic_fields` list if you added fields to
   them. `DATA_CARD.md` states in its first lines exactly what is real, what is
   generated, and which public source the real part came from.
2. **No real entities.** No real company names, site names, person names, product
   names, or study identifiers. Invent them. Real *gene symbols* and real
   *chemical structures* are fine and expected, since those are public scientific
   facts rather than anyone's data.
3. **Deliver a generator script, not a data dump.** Output a parameterised,
   seeded Python script (`--n`, `--seed`, `--out`) plus one committed sample
   output. A script is reviewable, diffable and re-runnable; a 40 MB CSV in a
   repository is none of those.
4. **Seeded and reproducible.** Same seed, same bytes. State the seed in the data
   card.
5. **Dependencies stay minimal, and generation is offline.** Standard library
   plus `pandas`/`numpy`; `rdkit` only where the module says so. In hybrid mode,
   fetching the public skeleton is a **separate, one-off script** (`fetch.py`)
   that writes a cached local copy; `generate.py` then reads that cache and makes
   no network calls. Anyone re-running the generator must get identical output
   without touching the network.
6. **Write the answer key.** A separate `ANSWER_KEY.md` documenting every planted
   signal, every planted defect, and the correct result. This is what makes the
   dataset teachable, and it is why the answers must not go in `DATA_CARD.md`,
   which the analyst reads first.

## The realism doctrine: read this before writing any code

Clean synthetic data teaches nothing. A dataset where the intended model wins on
the first try is a demo, not an exercise. Every dataset you build carries three
things:

**1. A true signal, discoverable with correct method.**
Something real is going on in the data, and a careful analyst will find it. State
it precisely in the answer key, including its effect size.

**2. A documented ceiling.**
Real data limits how well anything can perform: measurement noise, label
disagreement, irreducible ambiguity. Build that ceiling in deliberately, and
record its value. An analyst reporting accuracy above the ceiling has leaked
something, and the ceiling is how they find out.

**3. At least two traps that punish naive analysis.**
A trap is a defect that produces a *plausible but wrong* answer when handled
carelessly, and the right answer when handled properly. Noise is not a trap,
since noise only degrades; a trap flips the conclusion. For each trap, write down
in the answer key: what the naive approach concludes, what the correct approach
concludes, and the one line of code that separates them.

**Calibrate the mess.** Roughly: 2–8% of records carry a defect of any given
kind. Enough that ignoring it changes the answer; not so much that the dataset
reads as broken and the analyst gives up. State the injected rate for every
defect in the answer key so it can be recovered exactly.

**Make defects mechanistic, never random.** Missing values are missing *because*
the assay was expensive and only run on promising compounds. Miscoding happens
*because* the investigation ran out of time and defaulted to the easy category.
Mechanism is what makes a defect learnable rather than merely annoying. It is
also what creates selection bias, which is the thing worth teaching.

## Deliverables per module

| File | Contents |
|---|---|
| `fetch.py` | Hybrid mode only. Retrieves the public skeleton once, records the retrieval date and licence, writes a local cache. |
| `generate.py` | The seeded generator, reading the cache. Docstring explains the planted structure at a high level without giving away the answer key. |
| `sample/` | One committed run at the default seed. |
| `DATA_CARD.md` | Schema, provenance ("synthetic, generated by this script"), field definitions, known limitations, intended use, and an explicit "not real data" banner. |
| `ANSWER_KEY.md` | Every planted signal with effect size, every defect with injected rate, every trap with naive-vs-correct outcomes, and the performance ceiling. |
| `verify.py` | Runs the acceptance checks below and prints pass/fail. |

## Acceptance checks: run these before you say you are done

Write them into `verify.py` and show me the output.

1. **The signal is recoverable.** A correct analysis finds the planted effect
   within a stated tolerance of its true size.
2. **The naive analysis fails.** For each trap, demonstrate the wrong conclusion
   with code, then the right one. If a trap does not change the conclusion, it is
   not a trap. Strengthen it or cut it.
3. **The ceiling holds.** No method exceeds the documented ceiling. If one does,
   there is leakage, so find it.
4. **Defect rates match the spec.** Measure each injected rate in the generated
   output and compare to the target.
5. **Determinism.** Two runs at the same seed produce identical output. Two runs
   at different seeds produce different output with the same statistical
   structure.
6. **Nothing generated looks real.** Grep the generated records for anything
   resembling a real company, person, site or product name. In hybrid mode also
   confirm no real record was silently altered: real values are preserved
   verbatim, and every added field appears in `synthetic_fields`.
7. **Licence compliance.** State each public source's licence and retrieval date
   in the data card, and confirm the output may be redistributed under it.

## How to work

Ask me any clarifying questions first, then produce the generator. Show me the
answer key before the code. If the planted structure is wrong, the code does not
matter. When the code is written, run it, run `verify.py`, and paste the output.

Now here is the module.

---END PROMPT---

---

# Module A: evidence corpus

*For **citation-binding-rag**. Append this to the preamble.*

---BEGIN MODULE A---

Build a synthetic biomedical **evidence corpus**: abstract-length passages about
drug targets and diseases, of the kind a target-evidence assistant retrieves over.

**Mode:** hybrid where possible. Public skeleton: the **PMC Open Access Subset**,
filtered to the commercial-use licence group (CC0 / CC BY / CC BY-SA / CC BY-ND).
Never the NC group, and never the untagged "other" bucket. Real abstracts for a
chosen set of targets, keeping real PMC identifiers. Secondary skeleton:
**ClinicalTrials.gov** (public domain) for a second evidence register.

**Scale:** ~1,500 records. **ID prefix:** `SYN:` for generated records; real
records keep their PMC/NCT identifiers.

**Schema:** `source_id`, `title`, `abstract`, `year`, `journal` (invented names),
`target`, `disease`, `evidence_type` (genetic / expression / functional /
clinical / safety), `direction` (supporting / null / contradicting),
`retracted` (bool), `superseded_by` (nullable id), `synthetic`.

Real gene symbols and disease names are expected. Cover ~12 targets, and
deliberately leave 4 named targets out of the corpus entirely so that questions
about them must be refused.

### Planted signals

- Evidence strength genuinely varies by target: some have converging support
  across evidence types, some have a single weak study, some have a clinical
  failure that contradicts strong preclinical evidence.
- Numeric findings (n, p-values, response rates, odds ratios) appear in the text
  and are internally consistent within a record.

### Planted defects and traps

1. **Near-duplicate records.** ~6% of findings appear two or three times: a
   preprint and its published version, or the same result reported in a review.
   Different `source_id`, near-identical content.
   *Trap:* retrieval returns three copies of one finding and an unwary summary
   reports it as three independent lines of evidence. Correct handling
   deduplicates before counting.
2. **Retracted and superseded records** that carry a flag most pipelines never
   read. ~2%.
   *Trap:* a confident answer built on a retracted finding.
3. **Alias collision.** The same target appears under multiple symbols and older
   names; and include one pair of *different* targets with confusingly similar
   names.
   *Trap:* keyword retrieval merges two distinct targets, or misses two thirds of
   a target's evidence.
4. **Null results that read as answerable.** For two targets, the corpus contains
   only null or contradicting findings.
   *Trap:* the system answers positively, or refuses when it should report the
   null result. The correct behaviour is to report that the evidence is negative,
   which is different from both.
5. **Near-miss numbers.** Across near-duplicate records, the same finding is
   reported with slightly different figures (42% vs 43%, n=310 vs n=312).
   *Trap:* a summary that averages them, or cites a figure to the wrong record.
6. **Era skew.** Evidence for two targets clusters in 2014–2017 and nothing since.

### Also deliver

An **evaluation set** of ~40 questions with a required behaviour for each:
`answer`, `refuse`, `report_null`, or `report_conflict`. For `answer`, also give
the set of `source_id`s that legitimately support it. Include questions that are
answerable only after deduplication, and questions whose only support is a
retracted record.

**Ceiling to document:** the share of questions where the corpus genuinely cannot
support a confident answer.

---END MODULE A---

---

# Module B: assay history

*For **admet-split-gauntlet**. Append this to the preamble.*

---BEGIN MODULE B---

Build a synthetic **medicinal chemistry assay history**: the export a chemistry
data team would hand a modelling group.

**Scale:** ~6,000 measurements over ~3,500 compounds, spanning 2019–2026.
**ID prefix:** `CPD-SYN-` for compounds, `RUN-SYN-` for assay runs.
**Mode:** hybrid. Public skeleton: **MoleculeNet / DeepChem** datasets (MIT),
namely ESOL, Tox21, BBBP and Lipophilicity, for real structures and real measured
values. ChEMBL (CC BY-SA 3.0) only if I confirm the share-alike obligation is
acceptable.

Use `rdkit`. Keep the real structures and real measured values; generate the
metadata that public benchmark sets have had cleaned out of them, which is assay
dates, protocol versions, units, operators, analyst ids and replicate linkage.
Where you need a second synthetic endpoint with fully known ground truth,
generate its values from a documented function of computable descriptors plus
noise, and say so in the data card.

The framing for this module: *here is what a public benchmark looks like once you
put back what was removed from it.*

**Schema:** `compound_id`, `smiles`, `registration_date`, `project_code`,
`assay_id`, `endpoint`, `value_raw` (string), `value` (float, nullable), `units`,
`operator` (`=`, `<`, `>`), `assay_date`, `protocol_version`, `analyst_id`,
`replicate_of` (nullable), `synthetic`.

Endpoints: at least one regression (e.g. logD, solubility) and one binary
(e.g. hERG flag).

### Planted signals

- A **real structure–property relationship**: value is a documented function of
  computable descriptors plus noise. Write the function down in the answer key so
  the best achievable performance is calculable, not guessed.
- **Chemotype campaigns.** Compounds arrive in bursts around a scaffold, because
  a project works one series for months and then moves on. This is what makes
  random splits leak and temporal splits honest.

### Planted defects and traps

1. **Protocol change mid-series.** At one date, `protocol_version` increments and
   the measured baseline shifts by a fixed offset for one endpoint. ~35% of rows
   are post-change.
   *Trap:* a model trained across the change learns the offset as chemistry.
   Correct handling detects the discontinuity and either corrects or stratifies.
   This is the single most valuable trap in the module, so build it carefully.
2. **Censored values.** ~7% recorded as `"<0.1"` or `">100"` strings, with
   `operator` set and `value` null.
   *Trap:* coercing to float drops them (biasing away from the extremes) or
   parsing to the bound treats a censored value as measured.
3. **Unit inconsistency.** One site reports a subset in nM where the rest is µM.
   No flag except the `units` column, which is easy to ignore.
4. **Missing not at random.** The expensive endpoint is only run on compounds that
   passed the cheap one.
   *Trap:* the model appears strong but was trained on a filtered population, and
   fails on the full library.
5. **Replicates.** ~8% of compounds are measured 2–4 times.
   *Trap:* naive splitting puts replicates of one compound on both sides. Their
   scatter also defines the ceiling: no model beats assay reproducibility, and
   this is how the analyst discovers that number themselves.
6. **Duplicate registrations.** ~3% of compounds appear under two ids as salt
   forms or stereoisomer variants.
   *Trap:* canonical-SMILES dedup that strips stereochemistry merges genuinely
   different compounds; no dedup leaves duplicates spanning the split.
7. **Analyst effect.** One `analyst_id` has a small systematic bias on one
   endpoint.

**Ceiling to document:** the replicate-derived reproducibility limit per endpoint,
plus the noise floor of the generating function.

---END MODULE B---

---

# Module C: deviations and CAPA

*For **capa-recurrence-triage**. Append this to the preamble.*

---BEGIN MODULE C---

Build a synthetic **manufacturing deviation and CAPA history** of the kind a
quality management system holds.

**Mode:** hybrid, and here the split is unusually clean. No public pharmaceutical
deviation corpus exists and none will. But two public analogues have the same
shape (free-text incident narrative, coded cause, recorded corrective action):

- **openFDA device adverse events (MAUDE)**, CC0. It carries `mdr_text`
  narratives, coded `product_problems`, and a `remedial_action` field. Avoid
  GMDN-derived fields, which carry separate licensing.
- **NASA ASRS**, aviation incident narratives with a mature contributing-factor
  and human-factor coding taxonomy. Quote its "soft data / cannot estimate
  prevalence" caveat rather than hiding it.

Use one of those for **real narrative language and a real coded taxonomy**. Then
generate the layer neither of them contains, which is the whole subject of the
project: the link between a corrective action and whether the problem recurred.
No public dataset carries CAPA effectiveness or recurrence linkage.

If I ask for Mode S instead, generate the narratives too, in the style of the
analogue rather than copying from it.

**Scale:** ~4,000 deviations over 2019–2026, across 5 invented sites and 6
invented products. **ID prefix:** `DEV-SYN-`, `CAPA-SYN-`.

**Schema:** `deviation_id`, `date_opened`, `site`, `process_area`, `product`,
`batch_id`, `severity`, `description`, `immediate_action`,
`investigation_narrative`, `root_cause_category`, `root_cause_detail`, `capa_id`,
`capa_type`, `capa_description`, `capa_owner_function`,
`effectiveness_check_date`, `effectiveness_check_outcome`,
`related_deviation_ids`, `synthetic`.

Note what is not in the schema: any recurrence flag. That must be derived, and
the derivation is part of the exercise.

### Planted signals

- **CAPA type predicts recurrence.** Engineering controls hold; retraining-only
  CAPAs do not. Document the true effect size.
- **Root cause is inferable from narrative text.** Each category has its own
  vocabulary, but see the miscoding defect below for why accuracy is capped.
- Narratives are written the way people write them: boilerplate openings,
  abbreviations, inconsistent capitalisation, occasional typos, and a few obvious
  copy-paste jobs where an investigator reused a prior narrative and forgot to
  change one detail.

### Planted defects and traps

1. **Confounded headline effect.** Make the CAPA-type effect real but tangled with
   site and product complexity: one site both writes more retraining-only CAPAs
   *and* runs the harder product. Ideally construct a genuine **Simpson's
   paradox** on one slice, where the aggregate direction reverses within strata.
   *Trap:* the naive aggregate over-states the effect (or reverses it). Correct
   handling stratifies or adjusts, and lands near the true size.
2. **Taxonomy change.** In 2023, two root cause categories are merged and one is
   renamed. Old records keep old labels; no mapping table is provided.
   *Trap:* a model trained on pre-2023 data and tested after it collapses, and the
   analyst blames drift rather than the taxonomy. This is the second most
   valuable trap here.
3. **Root cause miscoding with a mechanism.** When an investigation approaches its
   30-day close deadline, it defaults to the easy category. Rate varies by site
   and rises in high-volume months, so miscoding is *correlated with the
   outcome*, not random.
   *This sets the classifier ceiling.* Document the exact injected rate.
4. **Ambiguous recurrence linkage.** Build genuine edge cases: a similar event at
   the same site 370 days later; a similar event in a different process area; a
   repeat under a different root cause code.
   *Trap:* two reasonable analysts derive different recurrence rates. Force the
   definition to be written down and defended.
5. **Missing and late effectiveness checks.** ~9% never recorded, ~15% recorded
   after the due date, and missingness correlates with site workload.
   *Trap:* dropping them silently drops the worst-performing site.
6. **Severity inflation over time.** What was coded Minor in 2019 is coded Major
   by 2025, with no change in the underlying events.
   *Trap:* a spurious trend in "serious deviations".
7. **Batch id collisions** across sites, so joins on `batch_id` alone are wrong.

### Also deliver

A **`derive_recurrence.py`** with the recurrence definition as an explicit,
documented, parameterised function. Window, matching keys and edge-case handling
are all switchable, so the analyst can see how much the headline number moves
with the definition. That sensitivity is the lesson.

**Ceiling to document:** classifier accuracy ceiling implied by the miscoding
rate, and the range the recurrence effect takes across reasonable definitions.

---END MODULE C---
