# Proposed title

Add E1.2 Configuration experiments, independent interval calibration and shift diagnostics

# Draft description — not submitted

Extend the ring-connected WS/ER/SBM benchmark with six reproducible simple,
connected fixed-degree Configuration variants: mean degree 4/8 crossed with
CV 0/.25/.5. Validate graphicality, preserve degrees when joining components
and switching edges, and record realized statistics and fingerprints. This
finite switch sampler does not claim exact uniform Configuration sampling.

Append base IDs 8–13 and register families 100–105 through a separate E1.2
entry point. Reuse the original DGP, truth, method loaders and evaluators.
Existing graphs, identities, RNG channels, checkpoints and method names remain
unchanged. Original J intervals remain available unchanged.

Add a separate j-calibrated interval method with exactly the same mu/ITE/ATE
point outputs as J. Register 100 independent calibration mechanisms and 200
held-out test mechanisms per family at observation budgets 20/50/100.
Calibration uses one prespecified repeat per mechanism and the finite-sample
96th-of-100 score quantile, floored at one to avoid shrinking intervals.
All 54 factors are frozen before test generation. Testing uses three repeats
and reruns seven original estimators on identical observed inputs.

Validation: 58 tests pass; all 16 original notebook code cells execute without
errors. Original mechanism/truth/factual fingerprints and the J/HT smoke
predictions remain unchanged. The new experiment has 600 calibration and 1,200
test mechanisms, 10,800 test contexts and 86,400 method/context records, with
zero prediction or interval exceptions. Audits check every paired point
output, all test graph statistics, factor reconstruction and freeze order.

Results: original J and calibrated J both meet 90% empirical coverage in all
54 held-out cells; both minima are 93.33%. Twelve factors exceed one. The mean
cell-wise width multiplier is 1.01377 and the maximum 1.17570. Eleven cells
have higher observed coverage; point metrics are unchanged. Original J already
passes this independent test, so calibration is a robustness comparison,
not the cause of the overall pass.

Retain the original smaller run's seven failed cells (minimum 86.46%), the
prior targeted replication, and the simulation-only oracle diagnosis of
fixed-mechanism shrinkage. Calibration has a marginal same-distribution
interpretation, not conditional, simultaneous or arbitrary OOD guarantees.
Baseline interval support and incomplete point support remain explicit.
No prior-strength adjustment or test-set retuning was performed.

The original make_table.py and make_figures.py were run. Source records,
coverage/width plots, baseline RMSE/bias/variance tables, task SDs, notebook
evidence and reproduction commands accompany the implementation. Team-wide
interface and designated literature-baseline integration still need coordination.

Complete the same 10,800 held-out contexts with a known-DGP oracle, fitting
observed outcomes only. The original table/figure scripts now compare all nine
methods. Add mu/ITE/ATE task means and task SDs. Original source results and
frozen calibration factors remain unchanged.

Add a preregistered screening of seven conditions: N=50/100/200, doubled main
direct or spillover coefficients, noise SD=2, and deletion of observed (1,1)
states. Across 216 graphs and 1,512 contexts, frozen J's spillover coverage is
75.46% under the stronger spillover mechanism, versus 93.52% for the oracle
with the correct condition prior. Direct-coefficient shifts also yield 89.81%
spillover coverage. These failures are retained; no post-test tuning is used.
This is a small shift screen relative to the reference distribution, not a
certification of training OOD membership or universal 90% coverage.

A tested local interface proposal preserves target/status/node order and
mu[N,2,2], refuses NOT_IDENTIFIED predictions, and blocks unsupported ordinary
plug-in queries. Its direct-observation mask is deliberately conservative;
it does not certify cross-node empirical support or integrate the upstream
identification API. E12_ACCEPTANCE_CHECKLIST.md maps the PDF/group requirements
to evidence, known failures, and remaining E1/E3 coordination.

Awaiting the user's review before any implementation commit, push or PR creation.
