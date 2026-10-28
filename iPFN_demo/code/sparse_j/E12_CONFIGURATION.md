# E1.2: connected simple Configuration graphs

## Scope and protocol

This extension targets upstream commit `b3c008c3cf178b929681afd6be803a132eee5434`.
Original `Setting` fields, base IDs 0–7, phase6 families 0–6, all RNG channels,
canonical configuration JSON, context IDs and frozen assets remain unchanged.
`ConfigurationSetting` subclasses `Setting`; new entries are appended to
`SETTINGS` (and therefore the existing phase5 `BASE_SETTINGS` registry).
No pretrained parameters or posterior intervals are recalibrated.

| E1.2 family | Base ID | Mean degree | Degree SD | Degree CV |
|---|---|---|---|---|
| 100 | 8 | 4 | 0 | 0 |
| 101 | 9 | 4 | 1 | 0.25 |
| 102 | 10 | 4 | 2 | 0.50 |
| 103 | 11 | 8 | 0 | 0 |
| 104 | 12 | 8 | 2 | 0.25 |
| 105 | 13 | 8 | 4 | 0.50 |

E1.2 uses N=200, shared nonlinear representation, noise SD=1, `dev` split,
split seed 1202601, and family IDs 100–105. Optional controls are the unchanged
legacy families 0 (WS k4), 2 (WS k8), 3 (ER k4), 4 (SBM k4). Controls keep their
original seeds. Comparisons use independent mechanisms from the same DGP class;
they are not paired interventions on identical coefficients/covariates.
The extension manifest is separate from unchanged `experiment.json`.
The graph API also supports the existing phase5 N=50 and N=100 sizes.

## Graph construction

Half the nodes have degree k−s, half have k+s, randomly assigned using the
supplied NumPy generator. Thus the actual mean, population SD and CV are exactly
k, s and s/k. This deliberately uses a bounded two-point degree distribution,
not a power law or an arbitrary requested CV. The API requires even N, integer
k/s, minimum degree >=2 and maximum degree <N. Erdős–Gallai validation rejects
non-graphical sequences explicitly, before any graph is returned.

1. Construct a simple realization with Havel–Hakimi.
2. If disconnected, remove a non-bridge edge from each of two components and
   add two cross edges. Each component has a cycle because minimum degree is
   at least two. This joins components and preserves every node's degree.
3. Attempt 20 double-edge switches per edge with seeded random edge selection
   and orientation. Reject overlapping endpoints, existing edges and proposals
   that disconnect the graph. Every attempt is bounded; no rejection-until-success
   loop, multigraph simplification, or ring backbone is used.

This is `simple_connected_configuration_v1`, a fixed-degree Configuration
variant with a finite connected switch chain, **not an exact uniform draw**.
Finite mixing is not proven; no claim about uniformity or absence of structural
bias is made. Changing the distribution or switch budget requires a new protocol
version/identities. NetworkX and NumPy versions are recorded in the manifest.
See [NetworkX Havel–Hakimi documentation](https://networkx.org/documentation/stable/reference/generated/networkx.generators.degree_seq.havel_hakimi_graph.html).

`graph_statistics.json` records each realized degree vector, histogram, mean,
population SD/variance/CV, extrema, edge count, connectedness, adjacency hash,
mechanism hash, truth hash and canonical configuration. These are producer-side
audit records; prediction still receives exactly the five factual arrays.

## Reproduce

Run from `iPFN_demo/code/sparse_j` in Python 3.12 with `requirements.txt` and
pytest installed. Each output directory must be new.

```sh
python -m pytest -q
python benchmark_e12.py --families 100 --mechanisms 1 --repeats 2 --methods j ht --observed 50 --output runs/e12_smoke
python benchmark_e12.py --families 0 2 3 4 100 101 102 103 104 105 --mechanisms 8 --repeats 12 --observed 20 50 100 --methods j ht hajek f-z2 lin-z2 kernel-old kernel-new --output runs/e12_v1
python make_table.py --input runs/e12_v1 --output results/e12_v1/tables
python make_figures.py --input runs/e12_v1 --output results/e12_v1/figures --observed 50
python report_e12.py --input runs/e12_v1 --output results/e12_v1/family_report
```

The full command has 80 mechanisms, 2,880 factual contexts and 20,160
method/context evaluations. Method names and interfaces are reused unchanged.
The other existing methods remain available through `--methods`; the external
CausalPFN checkpoint is not bundled by upstream.

## Interpretation

RMSE pools squared errors before taking the root. Signed bias comes from
`evaluate.py`. `make_table.py` supplies mean absolute conditional bias and
within-mechanism sample variance (R−1 divisor), equally weighting mechanisms.
These are different quantities: pooled signed bias may hide cancellation.
Incomplete supported predictions remain explicit and are never scored as zero.

Only frozen iPFN-J has an implemented ATE interval in the upstream benchmark;
HT/Hajek/regression/kernel interval coverage is unsupported, not zero.
The nominal 95% intervals are marginal posterior credible intervals. The 90%
empirical coverage target is checked separately for every Configuration family,
observation budget and effect; failed requested intervals count as misses.
Repeats are clustered within mechanisms. Per-family coverage Monte Carlo SEs
use the existing mechanism-cluster estimator; eight independent mechanisms per
family do not establish a population coverage guarantee. No test-set tuning is
performed if the target fails.

`results/e12_v1/` contains the actual run summary, family tables, plots and
compact evaluation records. Full factual arrays, truth and predictions live
under ignored `runs/e12_v1/` and are reproducible with the command above.

## Supplemental coverage evidence

See [results/e12_evidence/README.md](results/e12_evidence/README.md) for the
executed original notebook, mu/ITE task metrics, runtimes, simulation-only
oracle and fixed-mechanism bias diagnosis. A separate targeted replication
uses f103 mechanism IDs 1000–1099 (three repeats, observed=50), preserving all
v1 IDs and frozen estimators. Its direct/spillover/total coverage is
98.00%/94.00%/97.67%. This post-hoc diagnostic does not replace the seven failed
v1 cells or constitute an all-family final test. No calibration was performed.

The subsequent independent calibration/test extension is documented in
[E12_INTERVAL_CALIBRATION.md](E12_INTERVAL_CALIBRATION.md). It freezes 54 factors
using 100 new calibration mechanisms per family, then evaluates 200 different
test mechanisms per family with three repeats. Both frozen J and calibrated J
meet the 90% empirical target in all 54 held-out cells (minimum 93.33%).
These new results coexist with the unchanged original failures. Full coverage,
width costs, baseline comparisons and audit records are in
[results/e12_calibration_v1/README.md](results/e12_calibration_v1/README.md).
