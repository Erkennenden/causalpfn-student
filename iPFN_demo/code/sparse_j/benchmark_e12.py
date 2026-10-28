"""Run the registered E1.2 Configuration graph extension with frozen estimators."""
import argparse
from pathlib import Path

from benchmark import ALL_METHODS, DEFAULT_METHODS, run_benchmark
from pfn_social.e12_configuration import (
    FAMILY_BASE_IDS, CONTROL_FAMILIES, generate_mechanism, protocol_record,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', type=int, nargs='+',
                        choices=(*CONTROL_FAMILIES, *FAMILY_BASE_IDS),
                        default=list(FAMILY_BASE_IDS))
    parser.add_argument('--mechanisms', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--observed', type=int, nargs='+', choices=(20, 50, 100), default=[50])
    parser.add_argument('--methods', nargs='+', choices=ALL_METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument('--interval-level', type=float, default=.95)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.mechanisms <= 8 or not 1 <= args.repeats <= 12:
        parser.error('E1.2 v1 registers 1..8 mechanisms and 1..12 repeats')
    if len(set(args.families)) != len(args.families) or len(set(args.observed)) != len(args.observed):
        parser.error('Duplicate families or observation budgets')
    args.cohort = 'e12_configuration_v1'
    protocol = protocol_record(args.families, args.mechanisms, args.repeats, args.observed)
    grid = [(f, m, range(args.repeats), tuple(args.observed))
            for f in args.families for m in range(args.mechanisms)]
    run_benchmark(args, case_grid=grid, mechanism_factory=generate_mechanism, protocol=protocol)


if __name__ == '__main__':
    main()
