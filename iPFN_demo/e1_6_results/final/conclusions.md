# E1.6 complete-graph validation

- Topology: loopless complete graph K_200 (every degree = 199).
- Evaluation grid: 288 contexts; oracle, frozen iPFN J, one frozen kernel model, and three simple baselines.
- Reused components: Bernoulli(0.5) treatment, continuous-outcome DGP, four-arm truth, observation masks, effect definitions, frozen iPFN J checkpoint, and benchmark adapters.
- Structural result: a single complete-graph assignment realizes at most two of the four (t, s) cells because each node sees every other node.
- Realized support regimes: low-only (00/10): 99, high-only (01/11): 165, threshold split (01/10): 24.
- Empirical contrast support rates (direct / spillover / total): 0.344 / 0.000 / 0.000.
- Frozen iPFN J mean task ITE RMSE (direct / spillover / total): 0.3507 / 0.0249 / 0.3580.
- Frozen iPFN J ATE RMSE across tasks (direct / spillover / total): 0.2841 / 0.0234 / 0.2952.
- Frozen iPFN J nominal 95% ATE interval coverage, including failed requested intervals in the denominator (direct / spillover / total):
  - m=20: 96.88% / 97.92% / 96.88%
  - m=50: 95.83% / 100.00% / 94.79%
  - m=100: 98.96% / 100.00% / 97.92%
- All reported coverage values exceed the registered 90% acceptance threshold.
- As the observed-outcome budget grows from 20 to 100, iPFN-J ATE RMSE falls from 0.3611 to 0.1984 (direct) and from 0.3761 to 0.2065 (total); the per-budget tables report the corresponding bias and sample variance.
- Baseline comparison: iPFN-J beats non-interference OLS on all three supported RMSE targets. It beats the calibrated kernel on direct/total RMSE at m=50 and m=100, while the calibrated kernel is slightly better at m=20 and on spillover RMSE. Unsupported Hajek/Lin-z2 cells are not ranked.
- Population positivity is still true under the randomized design. Empirical zero cells are finite-sample coverage failures, not identification failures.
- The graph is degree-regular, so node differences are not degree effects. Remaining node heterogeneity comes from covariates and the shared DGP coefficients.
- Finite iPFN outputs for contrasts with false empirical support are extrapolations and must not be passed downstream as ordinary covered plug-in estimates.
- Hajek and Lin-z2 correctly leave unsupported targets missing. The non-interference baseline can return finite values, but its zero-spillover assumption is structurally misspecified for this DGP.

See evaluation/, figures/, tables/, aggregate_metrics.csv, task_metrics.csv, node_differences.csv, and rmse_comparison.png for the numeric results.
