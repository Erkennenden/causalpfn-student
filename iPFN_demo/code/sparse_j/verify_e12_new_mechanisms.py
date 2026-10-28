"""Fixed, targeted replication of f103; no tuning or original protocol change.

Exploratory diagnostic selected after inspecting v1. Reuses its DGP with new
mechanism IDs 1000..1099, three repeats, m=50. Not a final all-family test.
"""
import argparse
from pathlib import Path

from benchmark import run_benchmark
from pfn_social.e12_configuration import generate_mechanism, protocol_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.methods = ['j']
    args.interval_level = .95
    args.cohort = 'e12_targeted_replication_v1'
    protocol = protocol_record([103], 0, 3, [50])
    protocol.update(name=args.cohort, mechanism_ids=list(range(1000, 1100)),
                    purpose='Targeted exploratory replication of the worst v1 family; frozen J; no calibration or parameter selection.',
                    independent_unit='mechanism; three repeats share one graph/coefficients',
                    original_v1_ids_unchanged=True)
    grid = [(103, mid, range(3), (50,)) for mid in range(1000, 1100)]
    run_benchmark(args, case_grid=grid, mechanism_factory=generate_mechanism, protocol=protocol)


if __name__ == '__main__':
    main()
