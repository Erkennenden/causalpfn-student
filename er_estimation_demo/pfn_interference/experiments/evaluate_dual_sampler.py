#!/usr/bin/env python3
"""Compatibility entry: run only the normal (matched-old) benchmark."""


def validate_graph_count(count, *, graph_protocol='resampled'):
    if graph_protocol == 'fixed_er_v1':
        if count <= 0:
            raise ValueError('Fixed-ER episode count must be positive.')
    elif graph_protocol != 'resampled':
        raise ValueError(f'Unknown graph protocol: {graph_protocol}')
    elif count <= 0 or count % 16:
        raise ValueError('Graph count must be a positive multiple of 16 (formal default: 64).')


def main(argv=None):
    from pfn_interference.experiments.evaluate_matched_old_sampler import main as normal_main
    return normal_main(argv)


if __name__ == '__main__':
    main()
