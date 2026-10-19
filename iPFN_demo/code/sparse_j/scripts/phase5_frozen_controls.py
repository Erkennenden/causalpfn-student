"""Read a frozen Gaussian prior and its feature specification."""
import numpy as np


def load_prior(path):
    with np.load(path,allow_pickle=False) as d:
        spec={k:(d[k].item() if d[k].ndim==0 else d[k].copy()) for k in d.files if k not in ('mean','covariance','noise')}
        prior={k:d[k].copy() for k in ('mean','covariance','noise')}
    return spec,prior

