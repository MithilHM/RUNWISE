# Ranking and comparability rules

Area: comparability and recommendation ranking. This file is the documented answer to PRD 18 deliverable 5
("documented ranking rules"). If the code and this file disagree, the code is
the bug.

Nothing here is a probability, a confidence interval, or a prediction of
improvement. Every number is a deterministic function of stored records, chosen so
a researcher can disagree with one term instead of with an opaque score.

---

## 1. Comparability (FR-4)

`ComparabilityEngine.assess(left, right, ...)` returns one of four statuses.

### Dimension policy

| Dimension | Missing value is | Difference is |
|---|---|---|
| `dataset_identity` | insufficient | blocker |
| `dataset_version` | insufficient | blocker |
| `evaluation_split` | insufficient | blocker |
| `evaluation_protocol` | insufficient | major (caveat) |
| `metric_availability` | insufficient | blocker |
| `metric_direction` | n/a (never reported missing) | blocker |
| `model_family` | not material | major (caveat) |
| `preprocessing` | not material | major (caveat) |
| `training_budget` | not material | major (caveat) above tolerance |
| `random_seed` | minor caveat | minor caveat |
| `model_config` | minor caveat | minor (1 field) / major (2+) |
| `evidence_quality` | not material | minor (caveat) |

`metric_direction` has no missing-value case: a metric with no recorded direction
cannot be compared, and `_check_metric_direction` simply finds no shared metric to
check rather than inventing a finding.

`metric_availability` distinguishes three situations, in order: a named metric with
no value on one side is **missing**; two runs whose metric names do not overlap at
all are a **blocker**; a run that records no metrics at all is **missing**. Name
overlap is checked before value overlap, so two runs that both name
`validation_accuracy` measured the same thing even when one value is blank.

`training_budget` accepts `training_steps`, falling back to `max_training_steps`
when no realised step count was recorded.

### Status precedence

1. `NOT_COMPARABLE` — at least one blocker.
2. `INSUFFICIENT_INFORMATION` — no blocker, but a material field is unrecorded.
3. `COMPARABLE_WITH_CAVEATS` — any major or minor finding.
4. `COMPARABLE` — nothing found.

A blocker outranks missing information on purpose. "These two differ because the
dataset changed" is more actionable than "we cannot tell", and a researcher can
act on it.

### Identity keys

`dataset_id` / `dataset_version_id` are graph node identifiers.
`dataset_name` / `dataset_version` are human labels. They are different
namespaces. A label is never compared against a node id, and `"v1"` is never
assumed to mean `"dsv_churn_v1"`. When one run offers only an id and the other
only a label, the dimension is **missing**, not equal and not different.

### Intentional changes

`ignore_dimensions` and `ignore_config_fields` mark variables the caller means to
move. A finding on an ignored dimension is downgraded to `INFO` and kept in the
record, so the UI can still show it. A single differing configuration field reads
as a controlled ablation (`minor`); two or more read as confounded (`major`).

### Model awareness

Which configuration fields matter is per family, in `ModelFamilyProfile`. New
families register through `register_model_family`. An unregistered family gets no
material fields, and the engine falls back to comparing only keys both runs
recorded — it does not guess which hyperparameters matter for a family it has
never seen (PRD 3: general purpose, not transformer-specific).

---

## 2. Comparability score

An ordering aid in `[0, 1]`, used only to sort candidates.

```
score = max(0, 1 - sum(severity_penalty[reason.severity]))
```

| Severity | Penalty |
|---|---|
| `blocker` | 1.00 |
| `major` | 0.15 |
| `minor` | 0.05 |
| `info` | 0.00 |

A blocked pair always scores exactly `0.0`.

---

## 3. Recommendation ranking (FR-5, PRD 11)

### Pipeline

1. Parse objective -> target metric, family, dataset.
2. Retrieve experiments (graph port).
3. Withhold evidence under an open correction.
4. Select the comparable cohort.
5. Identify gaps.
6. Generate candidates.
7. Score candidates.
8. Enforce the compute budget.
9. Emit proposals with components and caveats.

### The comparable cohort

The anchor is the best run **inside the dominant measurement cohort**, not the best
run overall.

A cohort is the group of runs sharing `dataset_id`, `dataset_version_id`,
`evaluation_split` and `evaluation_protocol`. The largest cohort wins; ties break
on total recorded metric value, then on identifier, so the choice is deterministic.

This matters more than it looks. A run on a newer dataset version often scores
highest, and anchoring on it would treat a data change as a modelling result. The
cohort rule prevents that, and it is what makes the correction demo work: when the
v1 defect is reported, the v1 cohort is withheld and the proposal moves to data
that is still trusted.

### Candidate kinds

| Kind | Generated when |
|---|---|
| `replicate_seed` | fewer than 3 distinct seeds in the cohort |
| `sweep_parameter` | a family field has observed values and untested ladder values |
| `try_dataset_version` | a project dataset version was never used with the family |
| `new_model_family` | another family exists in the project, never compared on this data |
| `evaluate_missing_metric` | some runs in the family did not record the target metric |

At most `max_sweep_fields` (3) fields and `max_untested_values_per_field` (1)
value per field are proposed in one request, so one question cannot return a
twenty-run sweep plan.

### The six ranking factors

Weighted sum. Weights sum to 1.0, so the score stays in `[0, 1]`.

| Factor | Weight | Value reflects |
|---|---|---|
| `relevance` | 0.22 | family match, dataset-version match, objective term overlap |
| `evidence_quality` | 0.20 | cohort size, baseline comparability, withheld records |
| `information_gain` | 0.22 | how underexplored the lever is |
| `cost_efficiency` | 0.14 | position within the candidate pool's cost range |
| `uncertainty_reduction` | 0.10 | recorded uncertainties the proposal would resolve |
| `feasibility` | 0.12 | known cost that fits the budget |

`evidence_quality` takes a flat `0.15` penalty when any record was withheld by a
correction: a proposal built on a smaller evidence base should not rank as highly
as the same proposal built on a complete one.

`information_gain` is highest for seed replication, because with one seed no
observed difference can be separated from variance at all.

Weights are a product decision, not a fitted model. `RankingWeights` validates that
they sum to 1.0, and the evaluation harness should sweep them rather than treat
these as tuned.

Every component is returned on the recommendation as a `ScoreComponent` with its
weight, value, weighted value and a one-line rationale. The UI shows them; a
researcher should be able to say "I disagree with this term".

### Compute estimates

1. Median `compute_cost_gpu_hours` across the comparable cohort.
2. Else the family profile default, flagged as not a measurement.
3. Else `null`, with an uncertainty saying the estimate is unavailable.

An unknown cost is never assumed free. Under a budget, a candidate with no cost
estimate is not selected.

### Budget enforcement

Greedy by score, descending, with a hard ceiling.

Selection is by score, not score-per-hour. A cheap but weakly informative
experiment should not displace an expensive decisive one; cost already enters
through `cost_efficiency`, where the trade-off is visible and adjustable.

The total is never reported above the budget. `budget_exhausted` is true only when
the **budget** blocked a candidate. Candidates dropped for an unknown cost or by
`max_recommendations` are attributed to those causes separately, and the matching
reason appears in `uncertainties`, so a short plan is never blamed on the wrong
constraint.

---

## 4. Trust rules that are not negotiable

* A missing value is `None`. It is never `0`, never the mean, never the previous
  run's value. A metric column that imported blank counts as unrecorded, not as
  zero, everywhere: comparability, gap detection and cohort selection agree.
* Epistemic grades are passed through, never promoted. `disputed`, `superseded`
  and `unresolved` records are withheld from the evidence base; only `observed`,
  `confirmed` and `user_asserted` support a proposal.
* A run that did not complete cannot anchor a proposal. A proposal is built on a
  completed, measured run inside a single cohort.
* A disputed record may be discussed and is listed as withheld. It is never used
  as support for a proposal.
* A proposal is `EvidenceStatus.INFERRED`. Only a researcher approving it can move
  it, and approval never makes it an observation.
* The LLM writes prose. It never selects the target metric, never decides
  comparability, never invents lineage, and never creates a value. Generated text
  is rejected outright if it names an identifier outside the retrieved set or
  states a figure absent from the evidence; otherwise the deterministic text is
  used. A client reporting that it made no completed call is treated as
  unavailable, so `llm_used` cannot be set without a real call.
* With no API key, the deterministic path runs and the response reports
  `mode="deterministic"`, `llm_used=false`.
* An empty result set always carries its reason in `uncertainties`: no gap found,
  budget too small, no usable cohort, or no compute estimate.

---

## 5. Known limits

Stated so nobody over-trusts the numbers:

1. **Weights are unvalidated against real data.** They are documented and
   sweepable. PRD 13 asks for an evaluation against a retrieval-only baseline
   before any improvement is claimed; no such claim is made here.
2. **The LLM prose check is lexical.** It rejects unknown identifiers and figures
   absent from the evidence. A hallucinated *qualitative* claim phrased only in
   supplied words would pass. Catching that needs a claim-level entailment pass,
   which is out of MVP scope.
3. **"Newest" dataset version is a natural sort** of the version string (`v9`
   before `v10`). A non-numeric versioning scheme is compared lexicographically and
   may not order as intended.
4. **Cohort ties break on the summed metric value** without regard to metric
   direction, so for a `minimize` objective a tie on cohort size can resolve
   toward the worse cohort.
5. **`describe_paths` and `get_experiment_lineage` are not called by any service.**
   Path rendering is exercised by tests only; wiring it into Screen C is an
   integration step for the graph owner and the frontend owner.
6. **The LLM prose check cannot match an id with no separator.** `exp_001` and
   `run-0042` are detected because a prefix can be derived; a bare `abc123` has no
   prefix, so a fabricated sibling would pass. Graphs should use a separator, and
   the citation ledger remains the real guarantee for structured data.

---

## 6. Tunables

All in `ComparabilitySettings` and `RankingConfig`, all explicit so the evaluation
can sweep them.

| Setting | Default | Effect |
|---|---|---|
| `training_budget_relative_tolerance` | 0.2 | step difference that still counts as the same budget |
| `material_dimensions` | dataset, version, split, protocol, metric | which absences force `INSUFFICIENT_INFORMATION` |
| `max_sweep_fields` | 3 | parameters swept per request |
| `max_untested_values_per_field` | 1 | values proposed per parameter |
| `min_cluster_for_strong_evidence` | 3 | cohort size scoring full marks on evidence quality |
| `underexplored_field_threshold` | 2 | observations below which a field counts as underexplored |
| `seed_replication_target` | 3 | seeds before variance is considered measured |
| `RankingWeights.*` | see table above | factor weights |