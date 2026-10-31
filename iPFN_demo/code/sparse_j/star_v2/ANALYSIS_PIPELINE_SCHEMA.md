# C.1 frozen analysis, output and timing schemas

This document describes implemented C.1 interfaces, not an authorization to run
Formal Test or Real Model Gate. Synthetic test outputs are never formal results.

## Entry points and immutable inputs

- `star_v2.run_test --cohort test` remains PLAN_ONLY unless explicitly executed.
- Explicit execution additionally requires Test user authorization path/SHA256.
- `star_v2.test_acceptance` validates an actual COMPLETE Run and can write an
  independent acceptance receipt outside the immutable Run.
- `star_v2.run_analysis` requires run directory, Run Manifest SHA256, completion
  SHA256, Source Delivery/Manifest SHA256, execution receipt/path SHA256, and
  independent Run acceptance path/SHA256.
- `star_v2.analysis_acceptance` requires the same input anchors and analysis
  directory/Manifest/report SHA256.

Inputs: v2 initial/final Ledgers, execution_records.jsonl, public factual arrays,
truth NPZ/JSON, prediction and Pair NPZ/JSON, Support arrays/metadata, intervals,
runtime records and failure provenance. No v1 execution.json or legacy IDs.

Recovery uses existing Array/Ledger validators, original request identities and
Bundle constructors. JSON per-effect status lists are restored as tuples; values
are never changed. Canonical order is 00,01,10,11 and Direct,Spillover,Total.
Truth is restored, never computed from predictions. Ordinary predictors are not
called. Private simulator Y/beta/h are not used to compute metrics.

An external independently produced acceptance receipt is mandatory. Loader
revalidates the Run, source modules, Gate association and all persisted hashes.
Pilot, INCOMPLETE, missing/corrupt files, duplicate identities and inconsistent
provenance fail closed. Test production validation retains 200 mechanisms,
3 repeats, 200 nodes, 50 observed nodes, 1,800 Tasks and 10,200 Requests.

## Frozen statistical interfaces

No change to metrics.py, coverage.py, ranking.py or bootstrap.py.
- Pair metrics retain mechanism-equal RMSE/Bias/absolute Bias/ACB and within/between
  variances; incomplete main-cohort metrics remain N/A. Success-subset rows are
  explicitly DESCRIPTIVE_SUCCESS_SUBSET_ONLY.
- Node/group and center-minus-leaf statistics use existing frozen interfaces.
  ATE-only methods do not acquire node predictions.
- Support uses restored masks/reasons/extrapolation flags, not prediction
  finite masks. Support strata are descriptive only.
- Ranking is existing per-effect RANKABLE/NOT_RANKABLE qualification. No new
  composite score, ordinal rank, p-value or multiplicity adjustment is produced.
  Hajek is diagnostic only. Oracle reports remain separate.
- Formal Bootstrap uses the existing sole PCG64 stream:
  SeedSequence([20261030,2026103001,2,0,7,0]); indices int64 [10000,200].
  Seven methods and three effects share draws; duplicate mechanisms retain
  draw_position weighting. Twenty-one method pairs produce 63 effect records,
  including NOT_COMPARABLE entries without success-subset distributions.
- Only paired RMSE differences and nine J coverage distributions receive
  pointwise 95% percentile intervals, quantile method linear.
- J overall denominators are 1,200 per effect, stratified 600 per center arm.
  Failed intervals count as uncovered. Overall .90 target is effect-specific;
  Oracle intervals never enter J denominators. No coverage_valid Bootstrap CI.

Private synthetic seams accept explicit small draw matrices and mark every
artifact SYNTHETIC. Public production entry points cannot use those as Formal
Test acceptance.

## Analysis output inventory

- tables/: 13 primary tidy CSVs plus three effect-specific CSVs:
  pair_ate, node_error, group_ate, center_leaf, runtime, rankability,
  comparisons, comparison_index, j_coverage, capability, oracle_context,
  oracle_pair, support_strata; effect_Direct/Spillover/Total.
- bootstrap/distributions.npz plus JSON array schema and metadata.json.
- raw/: all Attempt statuses/provenance, Pair errors, J and Oracle intervals,
  source execution descriptors; per-method Support queries/counts.
- evaluation/: pair_metrics, node_metrics, center_leaf_metrics, j_coverage,
  rankability, runtime JSON. The directory name avoids the inherited portable
  path guard's forbidden results component.
- figures/: ten families, each SVG, PNG and metadata JSON:
  pair_ate, comparisons, j_coverage, node_error, group_ate, center_leaf,
  support_strata, runtime, oracle_context, oracle_pair.
- input_provenance.json; ANALYSIS_REPORT.md.
- analysis_manifest.json; acceptance_report.json.

The controller report says COMPLETE / PENDING_INDEPENDENT_VALIDATION. Independent
analysis acceptance returns engineering PASS separately; it does not rewrite the
controller report or claim scientific performance PASS.

All outputs are create-only. Partial output remains with ANALYSIS_INCOMPLETE.json
and cannot pass acceptance. Analysis output cannot be inside Runtime, Source
Delivery, Git worktrees or its immutable input Run.

## Table and figure schema

Every tidy CSV has row_id, method_id, effect, group, center_t, support_reason,
metric, value, status, unit, n_requested, n_valid, n_failed_or_missing,
ci_status/lower/upper/level/interpretation, source_selector, source_run_id,
source_snapshot_id, source_manifest_sha256, protocol_sha256, registry_seal,
reference_only and details.

Units: errors/RMSE/Bias/ACB/SD in outcome units, MSE/variances in outcome^2,
coverage in proportions, runtime in seconds per mechanism (sum of its Pair point
times). Blank/JSON null is N/A, never a fake zero.

Comparison index has 21 rows; effect comparison table has 63; J coverage has 9;
rankability has 21. Metric tables are tidy (one metric per row); incomplete
success-subset rows increase their row count explicitly. Node/group/Support
tables preserve center/leaf/all and center_t strata. Figure metadata binds the
actual source CSV hash, selected row IDs, units, denominators, failures,
intervals and input provenance. No figure creates new science.

Bootstrap NPZ contains indices, replicate_ids, draw_positions, method_rmse,
paired_rmse_differences [replicate,21,3], and j_coverage [replicate,3,3].
Non-comparable distributions remain NaN sentinels with reasons in metadata.
Object arrays/pickle are forbidden.

SVG timestamps are omitted and hash salt fixed. PNG metadata is fixed. Repeated
analysis of the same persisted synthetic input is required to produce identical
file bytes. Publication styling is not an acceptance condition.

## Runtime and preload boundaries

Three explicit _get_model calls in order: j, kernel-old, kernel-new. All must
return before DGP or Task materialization. Adapter construction alone is not a
successful load. No warmup, automatic retry or recovery.

run_resources.json:
- setup.adapter_construction and setup.model_loads: measured windows per loader.
- model_setup_seconds: construction plus the three loader windows.
- data_generation and task_materialization, with data_setup_seconds their sum.
- prediction_seconds: existing Pair point-window totals.
- j_interval_seconds: existing separately measured J interval-window totals.
- execution_prediction_seconds: point plus J interval totals.
- persistence_audit_seconds: preflight/evidence, ledger, request bookkeeping,
  failure mapping, persistence and integrity work; this is not pure disk time.
- total_wall_clock_seconds: execute_test API through pre-receipt resource stage.

A create-only sibling <run-directory-name>.execution_receipt.json records full
API wall time through completion-file writes, excluding its own receipt write,
and final_receipt_persistence_seconds. It binds Run Manifest and completion
SHA256. Independent acceptance requires its external SHA256. Module import and
CLI startup are outside this API boundary; not silently counted as prediction.

Per-pair runtime/RSS/GPU windows continue using runtime_monitor unchanged.
Runtime summaries sum three Pair times within mechanism before mean, sample SD
(ddof=1), median and P95 (linear). Failed estimations retain incurred time.
Context point time is not fabricated from a Pair total. Sampled RSS is not a
guaranteed process-lifetime peak. Failed preload creates no formal Task/Ledger;
partial load timing is not promoted into a successful setup table.

## Identification source and pending runtime binding

User explicitly approved on the user-declared date 2026-10-31 that the previously
reviewed POINT mathematics applies unchanged to Test cohort=2. This removes the
missing-mathematical-approval blocker, but is not execution authorization.

Delivery retains the exact approval quotation, unchanged BASIS object/hash,
Candidate identity and scope limitations. No signature, authenticated timestamp
or fictitious message identifier is created. Runtime authorization template is
APPROVED for that decision, but Run ID, corresponding new Gate and Identification
receipt hashes remain null until real identities exist. Final Snapshot/Manifest
are filled only from the new delivery, never from the old Gate.
Production validation rejects the template until every binding/hash is complete.

No new Gate or Formal Test is performed by C.1. Its changed source Snapshot
cannot inherit Stage 8.5-B Gate PASS.
