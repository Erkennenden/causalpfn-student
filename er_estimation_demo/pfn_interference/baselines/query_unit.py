"""Removed legacy node-level pseudo-ITE adaptation of average-effect estimators.

HT and L are evaluated only at their graph-level ATE target through
``pfn_interference.baselines.runner.run_ate_baselines_on_batch``. TNet is the exception: its fitted pointwise
potential-outcome surface also yields node-level effects, which are reported without an
individual-effect double-robustness guarantee. Localized DR-Lasso remains the dedicated
individualized DR-Lasso baseline.
"""

from __future__ import annotations


def run_query_unit_baseline_benchmark(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "The legacy node-level pseudo-ITE adaptation was removed. "
        "Use pfn_interference.baselines.runner.run_ate_baselines_on_batch for ATE/TNet outputs and "
        "Localized DR-Lasso for the dedicated localized ITE baseline."
    )
