"""Explicit E1.2 registration; no changes to the seven-family phase6 protocol."""
from dataclasses import asdict
import platform

import networkx as nx
import numpy as np
import torch

from pfn_social.configuration_graph import GENERATOR_VERSION, SWITCH_ATTEMPTS_PER_EDGE
from pfn_social.phase3_hierarchical import SETTINGS, CONFIGURATION_BASE_IDS
from pfn_social.phase5_large_network import LargeNetworkConfig, generate_mechanism as parent_mechanism
from pfn_social.phase6_limited_outcomes import (
    N, LimitedOutcomeMechanism, generate_mechanism as legacy_mechanism, mask_rng,
)

PROTOCOL = 'e12_configuration_v1'
SPLIT_SEED = 1202601
FAMILY_BASE_IDS = dict(zip(range(100, 106), CONFIGURATION_BASE_IDS))
# Existing shared nonlinear, noise=1 graph controls: WS k4/k8, ER k4, SBM k4.
CONTROL_FAMILIES = (0, 2, 3, 4)


def configuration(family):
    if family not in FAMILY_BASE_IDS:
        raise ValueError('Unknown E1.2 configuration family')
    return LargeNetworkConfig(FAMILY_BASE_IDS[family], N, family, 'dev', SPLIT_SEED, 'main')


def generate_mechanism(family, mechanism_id):
    if family in CONTROL_FAMILIES:
        return legacy_mechanism(family, mechanism_id)
    config = configuration(family)
    permutation = mask_rng(family, mechanism_id).permutation(N)
    return LimitedOutcomeMechanism(parent_mechanism(config, mechanism_id), permutation)


def protocol_record(families, mechanisms, repeats, observed):
    return dict(name=PROTOCOL, generator=GENERATOR_VERSION,
        upstream_commit='b3c008c3cf178b929681afd6be803a132eee5434',
        switch_attempts_per_edge=SWITCH_ATTEMPTS_PER_EDGE,
        families=list(families), mechanism_ids=list(range(mechanisms)),
        repeat_ids=list(range(repeats)), observed_outcomes=list(observed),
        registry={str(f): dict(base_setting_id=b, setting=asdict(SETTINGS[b]),
                              config=asdict(configuration(f))) for f, b in FAMILY_BASE_IDS.items()},
        controls=list(CONTROL_FAMILIES), target_coverage=.90,
        sampling='Havel-Hakimi, degree-preserving component joins, finite connected switch chain; not exact uniform.',
        comparison='Same DGP class; different families use independent mechanisms, not paired graph interventions.',
        coverage='Frozen J only; no test-set calibration; failures count as misses.',
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      networkx=nx.__version__, torch=torch.__version__))
