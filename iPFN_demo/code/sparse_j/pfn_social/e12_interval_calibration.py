"""Independent-mechanism split-conformal scaling of frozen J ATE intervals.

Calibration consumes simulation truths only from a registered calibration set.
Application consumes interval endpoints and frozen factors, never test truth.
This adds a separately named method; the original J outputs are unchanged.
"""
import math

import numpy as np

METHOD = 'j-calibrated'
INTERVAL_KIND = 'e12_independent_mechanism_split_conformal_95'


def conformal_factor(scores, level=.95):
    """Exact ceil((K+1)*level)-th order statistic; never shrink original CI.

If the required order exceeds K, the mathematical threshold is infinity.
The registered runner uses K=100 and rejects a non-finite artifact.
"""
    scores = np.asarray(scores, dtype=float)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or np.any(scores < 0) or not np.isfinite(level) or not 0 < level < 1):
        raise ValueError('Require finite nonnegative scores and 0<level<1')
    rank = math.ceil((len(scores)+1)*level)
    raw = float(np.sort(scores)[rank-1]) if rank <= len(scores) else float('inf')
    return dict(count=len(scores), order_statistic=rank, raw_quantile=raw,
                factor=max(1.,raw), nominal_level=level)


def interval_scores(truth, lower, upper):
    truth, lower, upper = [np.asarray(x,dtype=float) for x in (truth,lower,upper)]
    if (truth.shape != lower.shape or lower.shape != upper.shape
            or not all(np.isfinite(x).all() for x in (truth,lower,upper))
            or np.any(upper <= lower)):
        raise ValueError('Require matching finite truths and positive-width intervals')
    center = lower+(upper-lower)/2
    return np.abs(truth-center)/((upper-lower)/2)


def apply_factors(lower, upper, factors):
    lower, upper, factors = [np.asarray(x,dtype=float) for x in (lower,upper,factors)]
    if (lower.shape != (3,) or upper.shape != (3,) or factors.shape != (3,)
            or not all(np.isfinite(x).all() for x in (lower,upper,factors))
            or np.any(upper <= lower) or np.any(factors < 1)):
        raise ValueError('Require three valid intervals and finite factors>=1')
    center = lower+(upper-lower)/2
    half = (upper-lower)/2
    # Preserve exact original endpoints at factor=1, avoiding roundoff narrowing.
    lo = np.where(factors==1,lower,np.minimum(lower,center-half*factors))
    hi = np.where(factors==1,upper,np.maximum(upper,center+half*factors))
    return lo,hi
