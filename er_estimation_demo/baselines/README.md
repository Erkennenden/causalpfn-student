# Paper-aligned baselines for the unified majority-arm benchmark

All paper-facing baselines consume the **same generated graph episodes** as PFN.
The common arm is

```text
S_i = 1{treated-neighbor count > floor(degree_i/2)}
Z_i = (T_i,S_i) in {00,01,10,11}.
```

The common effects are direct `10-00`, spillover `11-10`, and total `11-00`.

## ITE: main comparison

The main ITE table contains PFN, Localized DR-Lasso, NSI, HyperSCI,
Tnet and causalPFN. The first five report all three majority-arm effects.
causalPFN is the retained covariate-only comparator and reports Direct only.

### Localized DR-Lasso

`localized_walsh.py` contains the radius-one specialization of **Localized
DR-Lasso** from Lu, Chi, Hu and Zhang, *Individualized Causal Effects under
Network Interference with Combinatorial Treatments* (arXiv:2602.19738).
The binary coding is `T_i in {0,1}`, `R_i=2*T_i-1`; nuisance names are `nu_Y` and `nu_R`.
The historical pilot default remains `b_G=2`, with five folds, `lambda_L=1e-4`, `rho=1e-3`,
and 16 conditional count/subset draws per arm. All runtime entry points share
`baselines/localized_config.py`; the older `--localized-nuisance-ridge` spelling
is an alias for `--localized-nuisance-shrinkage`. Cross-fit shrinkage uses only
training-fold means. See `PAPER_ROOTED_BASELINE_NOTES_CN.md` for the symbol map.
This adapter is used for empirical ITE point estimation.
Strict local-support failures remain `NaN` with support diagnostics; locality is
never silently widened just to force a number.

The radius-one distance has ceiling `.25`. The pilot `b_G=2` assigns raw
weights in `[.984375,1]` to every donor and now emits a near-uniform-kernel
diagnostic. Simply reducing bandwidth is not a validated fix: two fixed-graph
validation tasks at `.24` had only 62.5–62.9% supported nodes and highly unstable
estimates. The default remains unchanged for reproducibility; it is not a claim
of optimality. Compare errors on common supported targets and report coverage.
Changing the graph-configuration metric to an exposure metric is a separate
methodological adaptation, not this patch.

The CEPO entry point `python -m causalfm_experiment evaluate` now accepts the
same `--localized-*` flags. Actual values are recorded in `results.json`;
per-task geometry and warnings are in `benchmark_summary.json`. Geometry
summarizes factual-to-factual distances, not counterfactual query support;
consult the existing per-query support diagnostics for the latter.

### NSI (cross-sectional adaptation)

`ns_pcr.py` implements a cross-sectional adaptation of Agarwal, Cen, Shah and
Yu, *Network Synthetic Interventions: A Causal Framework for Panel Data Under
Network Interference*. The original NSI estimator learns synthetic donor
weights from repeated panel outcome trajectories. This benchmark has one
factual observation per node, so the adaptation replaces panel
outcome-trajectory PCR and neighborhood-treatment trajectory donor construction
with factual own-treatment/exposure localization and localized PCR over
the node's own covariate, normalized degree, and exposure. It does **not** claim
the original panel NSI identification, consistency, or asymptotic-normality theory.

For a query `(i,t,e)`, candidate donors satisfy `j != i` and `T_j=t`. Candidate
radii are the deterministic sorted unique set containing zero, every realized
`|E_j-e| <= h_max`, and `h_max`. Exact-exposure donors use equal weights. At a
positive radius `r`, donors within the radius receive truncated Gaussian weights
`exp(-|E_j-e|^2/(2r^2))`; the smallest candidate radius with sufficient kernel
effective sample size and at least one stable numerical PCR component is used.
No outcome information is used to select the radius.

Before fitting, the estimator validates the factual graph invariants: adjacency
must be binary, symmetric, loop-free, and free of isolated nodes; supplied degree
must equal the adjacency row sums; and supplied exposure must equal `(A @ T) /
degree`. This is only an input-integrity check. Donors remain pooled across
degrees and treated-neighbor counts and are localized by exposure proportions;
equal degree or equal treated-neighbor count is never required.

The PCR feature map is deliberately minimal: an intercept, the node's own
covariate, normalized degree, and factual/query exposure. Neighbor covariates,
higher-order powers, and covariate-by-exposure interactions are not used. The
local retained rank is
`min(pcr_rank_max, numerical_rank, n_positive-1)`. Exact count predictions are
then integrated with `estimands.conditional_count_weights`; an arm is supported
only when at least `min_support_mass` of its conditional design probability is
covered. Unsupported effects remain `NaN`. The target node's own factual outcome
is excluded from its donor fit, and simulator counterfactual truth is attached
only after estimation.

### HyperSCI (pair-edge / majority-adapted)

`hypersci.py` wraps the official HyperSCI class from Jing Ma et al., *Learning
Causal Effects on Hypergraphs* (KDD 2022). The table label is `HyperSCI`.
Every undirected graph edge becomes one two-node hyperedge; no additional
multi-person group is invented. Source provenance and exact local adjustments
are recorded in `hypersci_vendor/SOURCE.md`.

Node features are standardized `[X, degree/(N-1)]`. The encoder multiplies
phi(X) by each node's treatment before a single attention HypergraphConv layer
(two heads, width 32, ReLU, dropout .5). Official skip=123 and n_out=0 options
provide two linear outcome heads. Training minimizes standardized factual MSE
plus .01 times the upstream Wasserstein representation-balancing loss.
Adam: learning rate .001, weight decay .01, 500 epochs per graph, CPU.
Only factual A/X/T/Y enter fitting; no DGP coefficients or true effects enter.
This is the prior pilot configuration, not the original paper's default setup.

For each node and own treatment t, sample 4096 neighbor assignments conditional
on the majority arm under the supplied iid Bernoulli design. Recompute
interference under each assignment and average predictions. Low-arm t=0/1
queries share draws. A pair-message cache exactly reproduces the selected
one-layer network at evaluation time; only the conditional expectation is
approximated by Monte Carlo. Effects are 10-00, 11-10 and 11-00, restored to
original Y units. The generic exact scalar-exposure integration helper is
retained for other baselines; it is not used as a substitute for HyperSCI.

Per-node/effect integration standard errors are saved to
`hypersci_integration_diagnostics.csv` alongside each graph's ITE/ATE results.
These quantify conditional-integration Monte Carlo noise only, not statistical
uncertainty of fitted causal effects. HyperSCI replaces the prior ITE row;
no new HyperSCI ATE ranking row is introduced.

## causalPFN (pretrained checkpoint, Direct only)

The display name is exactly `causalPFN`. This is the previous X-only comparator;
its inference algorithm and `checkpoints/causalpfn_v0.pt` are unchanged.
`causalpfn_direct.py` retains the checkpoint loader and wraps
`causalpfn_xonly.estimate_x_only` once per graph. The latter fits its existing
factual gradient-boosted retrieval learner and queries the frozen local model
using covariates X, own treatment and factual outcome. It does not input graph
adjacency, observed exposure, or simulator counterfactual truth.

The node prediction is `mu1(X)-mu0(X)` and the ATE prediction is its mean over
all graph nodes. These are scored against the existing Direct benchmark truth,
as in the previous X-only comparator; this learner does not perform exposure
integration. Spillover and Total are not reported. The exposure-augmented
adapter and its separate table row have been removed.

The vendored model and original license remain under `third_party/causalpfn/`.
This local retrieval path requires scikit-learn but no FAISS, Hugging Face,
additional causalpfn package, or checkpoint download.

## ATE: HT

`runner.py` exposes `HT` for the common four-arm contrasts.  The design-based
reference is Michael P. Leung, *Causal Inference Under Approximate Neighborhood
Interference* (arXiv:1911.07085).  Majority-arm propensities are computed
analytically from node degree and the iid Bernoulli treatment probability.

## ATE: Hajek, Fisher, and Lin (Gao--Ding)

`Haj`, `F`, and `L` follow Gao and Ding, *Causal inference in network experiments:
regression-based analysis and design-based properties* (arXiv:2309.07476), using
the same analytic majority-arm exposure propensities as `HT`.

- `Haj` is the normalized inverse-exposure-probability (Hajek) estimator.
- `F` is the additive inverse-exposure-probability weighted regression adjustment:
  centered covariates share a common slope across exposure states.
- `L` is the fully-interacted version: centered covariates receive exposure-state-specific
  slopes.

All three are evaluated on Direct `10-00`, Spillover `11-10`, and Total `11-00`.
F and L use **unpenalized** weighted least squares, solved directly by SVD
without normal equations or ridge. Redundant covariates are allowed when the
requested intercept contrast is identifiable; otherwise it is unsupported.
The legacy `baseline-ridge` / `ate_ridge` argument is unused by paper-aligned F/L and reg-net.
If either target arm has no realized support in a graph, `Haj`, `F`, and `L` report
`NaN`/unsupported rather than silently changing the estimand.

## ATE: reg-net (Lu--Wang--Zhang)

`reg-net` represents the stronger network-dependent auxiliary-variable
adjustment from Lu, Wang and Zhang, *Adjusting auxiliary variables under
approximate neighborhood interference* (arXiv:2411.19789).  The benchmark uses
`G1=(D_i, E_i, X_i, neighbor-mean X_i)` and forms `G2` by interacting `G1` with
the full four-state joint exposure mapping `00,10,01,11`.

For each Direct `10-00`, Spillover `11-10`, and Total `11-00` contrast, the
`phi0` normalization is computed from the exact iid-Bernoulli design expectation
over own treatment and the treated-neighbor count.  This avoids enumerating all
`2^N` treatment assignments.  The normalized features are then passed to the
paper's network-HAC variance-minimizing linear adjustment. The default now uses
raw I/pi residual influence terms, the original (unprojected) HAC kernel and
zero ridge. If the quadratic has no finite minimum, it reports unsupported;
no PSD projection or automatic regularization is used to manufacture an estimate.
The old stabilized solver is accessible only through an explicit low-level
`mode="stabilized"` call, never from the main benchmark. If either target arm
has no realized support, the estimate is reported as `NaN`/unsupported rather
than changing the estimand.

## ATE: Chen et al. (ICML 2024) Tnet

`tnet.py` implements an adaptation of Chen et al., *Doubly Robust Causal Effect
Estimation under Networked Interference via Targeted Learning* (ICML 2024). It
preserves the paper's GCN feature aggregation, `g1` binary-treatment propensity
head, piecewise-linear `g2` neighborhood-exposure density head, two
treatment-specific outcome heads, degree-2 truncated-power spline perturbation `epsilon(t,z)`, and
targeted loss.  The fitted pointwise potential-outcome surface is integrated
over the benchmark's exact majority-arm exposure distribution.  The paper's DR
guarantee is attached to average dose-response / average-effect targets; the
node-level Tnet effects saved to ITE outputs are predictions without an
individual-effect DR guarantee.

The default training protocol now follows the author's public `src/main.py`:
160 epochs, each with 1 base update and 50 targeted updates; hidden width 64,
20 density intervals, quadratic splines with 9 knots at .1,...,.9 (12 basis
terms), alpha=.5, gamma=1, beta=20/sqrt(n), Adam learning rates 1e-4 / 1e-2,
weight decay 1e-3 for both optimizers, and GCN dropout .05. Predictor dropout
remains .1. Linear weights use the author's Normal(0,.1) initialization.
Observed training outcomes are standardized using their sample mean/SD;
predicted potential outcomes are returned in the original outcome units before
majority integration. Constant outcomes use unit scale. Actual resolved
configuration, optimizer update counts, and scaling are saved in diagnostics.

This remains a majority-arm adaptation, not a byte-for-byte reproduction:
the control denominator uses P(T=0)=1-P(T=1); the existing propensity floor
is retained; dropout is disabled at evaluation; graph computations are device
portable instead of hard-coded CUDA. No oracle outcomes enter fitting.
NSI's existing cross-sectional adaptation is unchanged.

Sources: https://github.com/WeilinChen507/targeted_interference/blob/main/src/main.py
and https://github.com/WeilinChen507/targeted_interference/blob/main/src/experiment.py

## Oracle semiparametric EIF/HAC anchor

`semiparametric.py` plugs simulator-known true outcome nuisances and the known
randomized arm propensity into the majority-arm adaptation of the EIC, then
uses the network-HAC kernel to estimate the long-run variance and standard
error.  The anchor is written separately to
`oracle_semiparametric_bound.csv`; it is a non-ranked theoretical-scale
diagnostic, not a fitted method and not an `Oracle-DR` entry.

The repository may retain older experimental/compatibility helpers internally,
but the unified paper-facing main tables contain only:

```text
ITE: PFN, Localized DR-Lasso, NSI, HyperSCI, Tnet, causalPFN
ATE: PFN, HT, Haj, F, L, reg-net, Tnet, causalPFN
Oracle anchor: true-nuisance EIF + network-HAC (non-ranked)
```
