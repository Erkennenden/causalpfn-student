# E1.2 independent-mechanism interval calibration

This extension compares frozen `j` with a separately named `j-calibrated`.
It preserves all J point estimates, original intervals, graph/DGP code,
checkpoints, original method names and previously recorded experiment IDs.
No prior, encoder, noise grid or regularization parameter is retrained.

## Registered experiment

The complete specification and source/checkpoint hashes are written to
`runs/e12_calibration_v1/protocol.json` before any new outcomes are generated.

| Item | Fixed choice |
|---|---|
| Configuration families | 100–105; mean degree 4/8 × CV 0/0.25/0.5 |
| Network size | 200 |
| Observation budgets | 20, 50, 100 |
| Calibration mechanisms | 2000–2099 per family; repeat 0 only |
| Test mechanisms | 3000–3199 per family; repeats 0, 1, 2 |
| Previously inspected, excluded | original IDs 0–7 and replication IDs 1000–1099 |
| Confidence level / reporting target | nominal 95% / empirical coverage at least 90% |
| Original comparison methods | j, ht, hajek, f-z2, lin-z2, kernel-old, kernel-new |
| Additional method | j-calibrated |

There are 600 independent calibration mechanisms (1,800 contexts) and 1,200
independent test mechanisms (10,800 contexts). The test has 86,400 method/context
records, including the calibrated copy of J. Repeats and budgets share one
mechanism and must not be counted as independent worlds.

The underlying `LargeNetworkConfig` retains its original `dev` value and seed
construction. Calibration and held-out test roles are explicitly registered
by new, disjoint mechanism IDs in the extension manifest; this does not claim
to modify or use upstream's restricted final-split protocol.

## Calibration and application

For each family, observation budget and effect, form 100 scores, one from each
calibration mechanism's repeat 0:

`score = abs(true_ATE - (lower + upper)/2) / ((upper - lower)/2)`.

Sort the scores and take the `ceil((100+1)*0.95) = 96`th order statistic.
The multiplier is `max(1, score_sorted[95])`. A new interval is centered at
the original interval midpoint, with its half-width multiplied by that value.
The midpoint need not equal J's posterior mean. At multiplier 1, endpoints
are retained exactly; floating-point rounding is not allowed to narrow an
expanded interval. J's mu, ITE, ATE and all point-support masks are copied
without alteration.

The general quantile function returns infinity when the required rank exceeds
the number of calibration mechanisms; it does not incorrectly cap the rank.
The registered 100-mechanism experiment requires finite calibration scores
and a complete calibration set. Calibration failures stop freezing rather
than silently selecting successful mechanisms. Requested test interval
failures remain misses in the existing evaluator.

All 54 multipliers are frozen to a hashed JSON artifact before any test
mechanism is generated. The application function accepts endpoints and factors,
not test truth. Test truth is read only by the evaluator after predictions.
The runner completes f103 first, then the remaining five families regardless
of pilot coverage; it never adjusts factors using the pilot or final test.

This is split-conformal calibration over independent simulated mechanisms
and their latent ATEs. Its marginal interpretation requires exchangeability
within each registered family/budget/effect. It does not promise coverage
conditional on every fixed mechanism, simultaneous coverage of all three
effects or all 54 cells, or validity under arbitrary distribution shift.
Simulation truth is available for calibration; real-data transfer requires
additional justification. The `max(1,q)` floor can make intervals conservative.
See [Angelopoulos and Bates, sections 1.1 and 2.3](https://arxiv.org/html/2107.07511v6).

## Execution and review

Run from this directory using the same Python environment as the original
benchmark. `--workers` controls independent benchmark shards; each worker
uses one BLAS/PyTorch thread. No alternative graph sampler is used for speed.

```sh
python -m pytest -q
python run_e12_calibration.py --output runs/e12_calibration_v1 --workers 4
python report_e12_calibration.py --input runs/e12_calibration_v1 --output results/e12_calibration_v1
python make_table.py --input runs/e12_calibration_v1/test/merged --output results/e12_calibration_v1/tables
python make_figures.py --input runs/e12_calibration_v1/test/merged --output results/e12_calibration_v1/figures --observed 50
```

Completed benchmark shards can be resumed with the same runner command.
The registered specification and inference source hashes must still match.
Interrupted partial shards are retained with an `_incomplete_` suffix and
excluded from merging; resumption never selects mechanisms based on accuracy.
Merged arrays use local hard links where possible, avoiding duplicate disk
space. Source shard arrays remain immutable.

The evidence report checks all 10,800 pairs of J/calibrated outputs for exact
mu/ITE/ATE equality, checks all test graph statistics, reconstructs all 54
factors from calibration scores, and verifies freeze-before-test event order.
It reports every family/budget/effect cell, interval width and multiplier,
original baseline point metrics/support, mechanism-level task RMSE SDs,
repeat-0 coverage and paired whole-mechanism bootstrap bands.

Timing for `j-calibrated` reuses J's measured prediction/interval time and adds
the actual arithmetic application cost; the model is not fitted twice.
Baseline interval coverage remains unsupported where upstream has no interval
implementation. Earlier v1 failures and the prior replication stay in their
own reports and are not replaced by this test.

Review results in `results/e12_calibration_v1/README.md`. Implementation and
results remain uncommitted pending the user's review; no push or PR is made.
