# Interference-CausalPFN: ER topology embedding + standard attention

This package contains the PFN-only experiment pipeline for network interference.

## Current model

- ER graphs only.
- Factual observed inputs and causal queries only; counterfactual truths are simulation supervision/evaluation targets, never model inputs.
- Network/topology-aware input embedding.
- Standard global self-attention; adjacency is used only by the network/topology-aware input embedding.
- Three PFN effect heads: direct, spillover, and total effects, each parameterized as a GMM.
- PFN-only ITE/ATE evaluation. Comparison baselines are intentionally removed from this package.

The DGP fields named `structural_baseline`, `baseline_sd`, and `*_baseline_coefficient` are outcome-model baseline terms, not comparison estimators.

Checkpoints from earlier model structures are intentionally incompatible with
this release. Retrain before using `--test-only`.

## Run on 8 MUSA devices

```bash
bash run_8musa_train.sh
```

The runner prepares the training CSV, launches 8-rank distributed training, loads the best point checkpoint, and writes PFN-only Algorithm-2 evaluation results.

Key result files:

- `algorithm2_summary.json`
- `algorithm2_dataset_metrics.csv`
- `algorithm2_unit_estimates.csv`

## Tests

```bash
python -m pytest -q
```
