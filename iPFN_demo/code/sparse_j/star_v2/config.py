"""Immutable registered settings from FROZEN Candidate v0.3.2.

Base setting 0 representation/noise are transcribed from the fixed source;
no legacy DGP module is imported. Changes require a separate protocol decision.
"""

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class StarV2Config:
    protocol_id: str = "e1_5_star_v2"
    protocol_version: str = "Candidate v0.3.2"
    protocol_sha256: str = "fa99e65d3082d90a8675354fc4a478051b0fe3b2980d5bf78c59f29ab279209b"
    source_commit: str = "fbc0d9e09789248139ff9a201b8f89bf71ae57cf"
    root_seed: int = 20261030
    protocol_code: int = 2026103001
    pilot_cohort_code: int = 1
    test_cohort_code: int = 2
    N: int = 200
    M: int = 50
    center_node: int = 0
    x_dim: int = 20
    leaf_treatment_probability: float = 0.5
    exposure_threshold: float = 0.5
    exposure_operator: str = "strictly greater than"
    base_setting_id: int = 0
    representation: str = "shared_nonlinear"
    noise_sd: float = 1.0
    representation_seed: int = 2026101604
    arm_order: tuple[str, ...] = ("00", "01", "10", "11")
    effect_order: tuple[str, ...] = ("Direct", "Spillover", "Total")
    target_id: str = "star_v2_mu_iid_bernoulli_0p5_majority_gt0p5_v1"
    pilot_mechanisms: int = 5
    pilot_pairs_per_mechanism: int = 2
    test_mechanisms: int = 200
    test_pairs_per_mechanism: int = 3

    def __post_init__(self):
        # Exact types reject bool/int equivalence, float IDs and mutable orders.
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not type(field.default):
                raise TypeError(f"{field.name} requires {type(field.default).__name__}")
            if value != field.default:
                raise ValueError(f"{field.name} is frozen at {field.default!r}")

    def cohort_limits(self, cohort_code: int) -> tuple[int, int]:
        """Registered (mechanism count, pair repeats); no data are generated."""
        if type(cohort_code) is not int:
            raise TypeError("cohort_code requires a Python int, not bool or float")
        if cohort_code == self.pilot_cohort_code:
            return self.pilot_mechanisms, self.pilot_pairs_per_mechanism
        if cohort_code == self.test_cohort_code:
            return self.test_mechanisms, self.test_pairs_per_mechanism
        raise ValueError("Only registered Pilot=1 and Test=2 cohorts are allowed")


CONFIG = StarV2Config()
