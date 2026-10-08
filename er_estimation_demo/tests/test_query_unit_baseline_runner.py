from __future__ import annotations

import pytest

from baselines.query_unit import run_query_unit_baseline_benchmark


def test_legacy_node_level_average_effect_adaptation_is_removed() -> None:
    with pytest.raises(RuntimeError, match="pseudo-ITE adaptation was removed"):
        run_query_unit_baseline_benchmark()
