"""Configuration localization for the manuscript's adapted ITE baseline.

Keep the historical b_G=2 pilot for reproducibility, with diagnostics for
its near-uniform weights. A smaller bandwidth is not automatically better:
strict .24 locality was unstable on the fixed ER graph's validation tasks.
"""

LOCALIZED_METHOD_NAME = 'Localized DR-Lasso'
LOCALIZED_PROTOCOL = 'localized_dr_lasso_adapted_v1'
LOCALIZED_DEFAULTS = dict(
    localized_bandwidth=2.0,
    localized_min_neighbors=20,
    localized_mode='kernel',
    localized_lasso_alpha=1e-4,
    localized_nuisance_ridge=1e-3,
    localized_cross_fit_folds=5,
    localized_arm_samples=16,
)


def add_localized_arguments(parser):
    """Use the same settings in training/test-only and checkpoint pfn_interference.metrics."""
    parser.add_argument('--localized-mode', choices=('kernel', 'knn'),
        default=LOCALIZED_DEFAULTS['localized_mode'],
        help='Configuration localization; kernel never automatically switches to kNN.')
    parser.add_argument('--localized-bandwidth', type=float,
        default=LOCALIZED_DEFAULTS['localized_bandwidth'],
        help='b_G for rooted distance d_1 (maximum .25); default 2.0 preserves the historical near-global pilot. Smaller bandwidths require validation of support and stability.')
    parser.add_argument('--localized-min-neighbors', type=int,
        default=LOCALIZED_DEFAULTS['localized_min_neighbors'],
        help='Number of neighbors used only in explicit kNN mode; ignored by kernel mode.')
    parser.add_argument('--localized-lasso-alpha', type=float,
        default=LOCALIZED_DEFAULTS['localized_lasso_alpha'],
        help='lambda_L: L1 penalty for the localized residual coefficient.')
    parser.add_argument('--localized-nuisance-shrinkage', '--localized-nuisance-ridge',
        dest='localized_nuisance_ridge', type=float,
        default=LOCALIZED_DEFAULTS['localized_nuisance_ridge'],
        help='rho: shrinkage toward the training-fold mean; the legacy ridge spelling remains an alias.')
    parser.add_argument('--localized-cross-fit-folds', type=int,
        default=LOCALIZED_DEFAULTS['localized_cross_fit_folds'],
        help='K_cf: number of nuisance cross-fitting folds.')
    parser.add_argument('--localized-arm-samples', type=int,
        default=LOCALIZED_DEFAULTS['localized_arm_samples'],
        help='M: count/subset Monte Carlo draws per node and majority arm.')


def localized_benchmark_kwargs(args):
    """Forward actual CLI values, retaining defaults for older Namespace callers."""
    return {name: getattr(args, name, default) for name, default in LOCALIZED_DEFAULTS.items()}
