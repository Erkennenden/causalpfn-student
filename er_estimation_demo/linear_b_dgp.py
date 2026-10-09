"""B mechanism with one shared a~Uniform(4,8) per dataset and four exact CEPO labels."""
from functools import partial
from numbers import Integral
import numpy as np
import torch
from causalfm_dgp_v1.dgp import generate_task as _generate_task
from random_lpe_prior import _task_generator

VERSION = 'linear_b_random_a_four_cepo_v2'
FORMULA = '1-X-3*neighbor_x+(a+0.2*exp(X))*T-0.9*E+epsilon'
A_LOW, A_HIGH = 4., 8.
SCENARIO = 'linear_b_random_a'


def coefficient_metadata(*, seed=12345, stream='train', task_id=0):
    """Replay a dataset-level draw without consuming X/T/noise random streams."""
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError('seed must be an integer')
    if not isinstance(task_id, Integral) or isinstance(task_id, bool) or not 0 <= task_id < 2**63:
        raise ValueError('task_id must be a nonnegative signed-64-bit integer')
    if not isinstance(stream, str) or not stream:
        raise ValueError('stream must be nonempty')
    rng = _task_generator(int(seed), stream, int(task_id), 'outcome/linear_b/a')
    a = A_LOW + (A_HIGH-A_LOW) * float(torch.rand((), generator=rng, dtype=torch.float64))
    return dict(a=a, a_seed=rng.initial_seed())


def outcome_b(inputs, *, a):
    x, neighbor_x, treatment, exposure = np.moveaxis(np.asarray(inputs), -1, 0)
    return 1 - x - 3 * neighbor_x + (a + .2 * np.exp(x)) * treatment - .9 * exposure


def generate_task(adjacency, *, task_id=0, seed=12345, stream='train',
                  noise_sd=4., treatment_prob=.5):
    coefficient = coefficient_metadata(seed=seed, stream=stream, task_id=task_id)
    task = _generate_task(adjacency, task_id=task_id, seed=seed, stream=stream,
        noise_sd=noise_sd, treatment_prob=treatment_prob,
        outcome=partial(outcome_b, a=coefficient['a']))
    task.metadata.update(version=VERSION, outcome_kind=SCENARIO,
        outcome_family='linear_in_treatment_and_exposure', outcome_seed=coefficient['a_seed'],
        **coefficient, a_distribution=dict(name='uniform', low=A_LOW, high=A_HIGH),
        formula=FORMULA, noise_law=f'iid Normal(0,{noise_sd**2:g})',
        cepo_order=['mu00', 'mu01', 'mu10', 'mu11'])
    return task
