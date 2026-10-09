"""Selection and merge helpers for density-sweep benchmark summaries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from baselines.causalpfn_direct import CAUSALPFN_METHOD_NAME
from baselines.localized_walsh import ITE_METHOD_NAME
from evaluation.unified import MAIN_ATE_METHODS, MAIN_ITE_METHODS

DEFAULT_DENSITY_SETTINGS = (
    ("p002", 0.02),
    ("p005", 0.05),
    ("p010", 0.10),
    ("p015", 0.15),
    ("p020", 0.20),
)


def select_density_summary_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    estimand_level: str,
) -> list[dict[str, object]]:
    """Return only formal primary rows for a density ranking.

    ITE uses all-target rows except Localized DR-Lasso, whose primary row is
    supported_targets. CausalPFN is Direct-only. ATE follows MAIN_ATE_METHODS, which uses the Chen et al. TNet adaptation as
    the formal doubly-robust baseline; the oracle EIF/HAC anchor
    stay outside the formal density ranking.
    """

    level = str(estimand_level).upper()
    if level not in {"ITE", "ATE"}:
        raise ValueError("estimand_level must be 'ITE' or 'ATE'.")

    selected: list[dict[str, object]] = []
    allowed = set(MAIN_ITE_METHODS if level == "ITE" else MAIN_ATE_METHODS)
    for source in rows:
        row = dict(source)
        method = str(row.get("method", ""))
        effect = str(row.get("effect", ""))
        if method not in allowed:
            continue
        if method == CAUSALPFN_METHOD_NAME and effect != "direct":
            continue
        if level == "ITE":
            subset = str(row.get("evaluation_subset", ""))
            expected_subset = "supported_targets" if method == ITE_METHOD_NAME else "all_targets"
            if subset != expected_subset:
                continue
        selected.append(row)
    return selected


def merge_density_summaries(
    root: Path,
    settings: Sequence[tuple[str, float]] = DEFAULT_DENSITY_SETTINGS,
) -> Path:
    """Merge per-density ITE/ATE summaries using formal ranking policy."""

    root = Path(root)
    merged: list[dict[str, object]] = []
    for label, p_er in settings:
        for level, filename in (("ITE", "ite_summary.csv"), ("ATE", "ate_summary.csv")):
            path = root / label / filename
            with path.open(newline="", encoding="utf-8") as handle:
                selected = select_density_summary_rows(
                    csv.DictReader(handle), estimand_level=level
                )
            for row in selected:
                row["estimand_level"] = level
                row["p_er"] = float(p_er)
                merged.append(row)

    if not merged:
        raise ValueError("density summary merge produced no rows.")
    fieldnames = ["p_er", "estimand_level"]
    for row in merged:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output = root / "density_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(merge_density_summaries(args.root))


if __name__ == "__main__":
    main()
