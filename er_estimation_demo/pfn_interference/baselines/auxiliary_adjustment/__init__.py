"""Regression-adjustment baselines under approximate neighborhood interference."""

from .estimators import (
    AdjustmentEstimate,
    NetworkAdjustmentResult,
    estimate_auxiliary_methods,
    estimate_weighted_regression,
    solve_network_adjustment,
)
from .features import (
    ObservedPhi0Result,
    Phi0Result,
    build_g1,
    build_g2,
    build_g2_full,
    build_majority_phi0_g2_exact,
    normalize_phi0,
)

__all__ = [
    "AdjustmentEstimate",
    "NetworkAdjustmentResult",
    "Phi0Result",
    "ObservedPhi0Result",
    "build_g1",
    "build_g2",
    "build_g2_full",
    "build_majority_phi0_g2_exact",
    "normalize_phi0",
    "estimate_weighted_regression",
    "solve_network_adjustment",
    "estimate_auxiliary_methods",
]
