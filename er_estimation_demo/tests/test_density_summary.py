from __future__ import annotations

from baselines.tnet import TNET_METHOD_NAME
from evaluation.density import select_density_summary_rows


def test_ite_density_keeps_primary_rows_and_causalpfn_direct_only() -> None:
    rows = [
        {"method": "PFN with interference", "effect": "direct", "evaluation_subset": "all_targets"},
        {"method": "PFN with interference", "effect": "direct", "evaluation_subset": "common_support"},
        {"method": "Localized DR-Lasso", "effect": "spillover", "evaluation_subset": "supported_targets"},
        {"method": "Localized DR-Lasso", "effect": "spillover", "evaluation_subset": "common_support"},
        {"method": "GPS", "effect": "total", "evaluation_subset": "all_targets"},
        {"method": TNET_METHOD_NAME, "effect": "total", "evaluation_subset": "all_targets"},
        {"method": "causalPFN", "effect": "direct", "evaluation_subset": "all_targets"},
        {"method": "causalPFN", "effect": "spillover", "evaluation_subset": "all_targets"},
    ]

    selected = select_density_summary_rows(rows, estimand_level="ITE")
    keys = {(row["method"], row["effect"], row["evaluation_subset"]) for row in selected}

    assert ("PFN with interference", "direct", "all_targets") in keys
    assert ("Localized DR-Lasso", "spillover", "supported_targets") in keys
    assert ("GPS", "total", "all_targets") not in keys
    assert (TNET_METHOD_NAME, "total", "all_targets") in keys
    assert ("causalPFN", "direct", "all_targets") in keys
    assert not any(subset == "common_support" for _, _, subset in keys)
    assert not any(method.startswith("causalPFN") and effect != "direct" for method, effect, _ in keys)


def test_ate_density_excludes_oracle_and_keeps_adapted_causalpfn_direct() -> None:
    rows = [
        {"method": "PFN with interference", "effect": "direct"},
        {"method": "HT", "effect": "spillover"},
        {"method": "L", "effect": "total"},
        {"method": "AIPW-Ridge (known propensity)", "effect": "direct"},
        {"method": "Oracle-DR", "effect": "direct"},
        {"method": TNET_METHOD_NAME, "effect": "spillover"},
        {"method": "causalPFN", "effect": "direct"},
        {"method": "causalPFN", "effect": "total"},
    ]

    selected = select_density_summary_rows(rows, estimand_level="ATE")
    keys = {(row["method"], row["effect"]) for row in selected}

    assert ("PFN with interference", "direct") in keys
    assert ("HT", "spillover") in keys
    assert ("L", "total") in keys
    assert ("AIPW-Ridge (known propensity)", "direct") not in keys
    assert (TNET_METHOD_NAME, "spillover") in keys
    assert ("causalPFN", "direct") in keys
    assert not any(method == "Oracle-DR" for method, _ in keys)
    assert not any(method.startswith("causalPFN") and effect != "direct" for method, effect in keys)
