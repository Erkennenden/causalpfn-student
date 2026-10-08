"""Fixed A-group outcome on the unchanged v2 graph/X/T/query random streams.

Y(t,e) = 1 - X - 3 lambda mean_neighbor(X) + (6 + .2 exp(X)) t - .9 lambda e + epsilon,
epsilon iid N(0, 16). The exponential is raw, without reference normalization.
"""
import math
from numbers import Integral

import torch

from random_lpe_prior import PriorConfig, _assemble_task, _draw_graph, _task_generator

VERSION = 'a_group_fixed_v1'
RANDOM_GAMMA_VERSION = 'a_group_random_gamma_v1'
RANDOM_COEFF_VERSION = 'a_group_random_coeff_v1'
NOISE_SD = 4.0
OUTCOME_SPEC = dict(version=VERSION, formula='1-X-3*neighbor_x+(6+0.2*exp(X))*T-0.9*E+epsilon',
                    noise_sd=NOISE_SD, noise_law='iid Normal(0,16)',
                    neighbor_coefficient=-3.0, spillover_coefficient=-0.9,
                    nointerference_prob=0.0, function_normalization=False)


def outcome_spec(interference_lambda=1.0):
    if not math.isfinite(interference_lambda) or interference_lambda < 0:
        raise ValueError('interference_lambda must be nonnegative and finite.')
    return {**OUTCOME_SPEC, 'interference_lambda': float(interference_lambda),
            'formula': '1-X-3*lambda*neighbor_x+(6+0.2*exp(X))*T-0.9*lambda*E+epsilon',
            'neighbor_coefficient': -3.0*interference_lambda,
            'spillover_coefficient': -0.9*interference_lambda}


def outcome_spec_for_config(config):
    if config.dgp_version == RANDOM_COEFF_VERSION:
        beta, gamma = config.neighbor_beta, config.spillover_gamma
        distribution = lambda value, low, high: ({'family': 'uniform', 'low': low, 'high': high}
            if value is None else {'family': 'point_mass', 'value': value})
        return {**OUTCOME_SPEC, 'version': RANDOM_COEFF_VERSION,
            'formula': '1-X+beta_task*neighbor_x+(6+0.2*exp(X))*T+gamma_task*E+epsilon',
            'neighbor_coefficient': beta, 'spillover_coefficient': gamma,
            'neighbor_distribution': distribution(beta, config.neighbor_beta_min, config.neighbor_beta_max),
            'spillover_distribution': distribution(gamma, config.spillover_gamma_min, config.spillover_gamma_max),
            'coefficient_scope': 'independent beta and gamma draws per task; shared across units',
            'coefficients_observed_by_model': False, 'gamma_observed_by_model': False}
    if config.dgp_version != RANDOM_GAMMA_VERSION:
        return outcome_spec(config.interference_lambda)
    fixed = config.spillover_gamma
    distribution = ({'family': 'uniform', 'low': config.spillover_gamma_min,
                     'high': config.spillover_gamma_max} if fixed is None else
                    {'family': 'point_mass', 'value': fixed})
    return {**OUTCOME_SPEC, 'version': RANDOM_GAMMA_VERSION,
            'formula': '1-X-3*neighbor_x+(6+0.2*exp(X))*T+gamma_task*E+epsilon',
            'neighbor_coefficient': -3.0, 'spillover_coefficient': fixed,
            'spillover_distribution': distribution,
            'gamma_scope': 'one coefficient per data task, shared across all units',
            'gamma_observed_by_model': False}


def outcome_description(spec):
    """Plain-text mechanism label from the recorded coefficients/distribution."""
    distribution = spec.get('spillover_distribution', {})
    if distribution.get('family') == 'uniform':
        gamma = f'gamma~Uniform({distribution["low"]:g},{distribution["high"]:g})'
    else:
        gamma = f'gamma={spec["spillover_coefficient"]:g}'
    neighbor = spec.get('neighbor_distribution', {})
    beta = (f'b_n~Uniform({neighbor["low"]:g},{neighbor["high"]:g})'
        if neighbor.get('family') == 'uniform' else f'b_n={spec["neighbor_coefficient"]:g}')
    return f'{beta}, {gamma}, noise SD={spec["noise_sd"]:g}'


def gamma_kwargs(config):
    if config.dgp_version == RANDOM_COEFF_VERSION:
        if config.neighbor_beta is not None:
            return {'neighbor_beta': config.neighbor_beta, 'spillover_gamma': config.spillover_gamma}
        return {'neighbor_range': (config.neighbor_beta_min, config.neighbor_beta_max),
                'spillover_range': (config.spillover_gamma_min, config.spillover_gamma_max)}
    if config.dgp_version != RANDOM_GAMMA_VERSION:
        return {}
    if config.spillover_gamma is not None:
        return {'spillover_gamma': config.spillover_gamma}
    return {'spillover_range': (config.spillover_gamma_min, config.spillover_gamma_max)}


def generate_tasks(config: PriorConfig, task_ids, *, stream='train', seed=12345,
                   device='cpu', balanced=False, fixed_adjacency=None, interference_lambda=1.0,
                   spillover_gamma=None, spillover_range=None, neighbor_beta=None, neighbor_range=None):
    """Only the outcome mechanism changes; balanced selects graph families only."""
    spec = outcome_spec(interference_lambda)
    neighbor_coefficient = spec['neighbor_coefficient']
    spillover_coefficient = spec['spillover_coefficient']
    if neighbor_beta is not None or neighbor_range is not None:
        if interference_lambda != 1.0:
            raise ValueError('Neighbor coefficients cannot be combined with interference_lambda != 1.')
        if neighbor_beta is not None and neighbor_range is not None:
            raise ValueError('Specify either neighbor_beta or neighbor_range.')
        if neighbor_beta is not None and not math.isfinite(neighbor_beta):
            raise ValueError('neighbor_beta must be finite.')
        if neighbor_range is not None:
            low, high = neighbor_range
            if not all(math.isfinite(v) for v in (low, high)) or not low < high:
                raise ValueError('neighbor_range must have finite increasing bounds.')
    if spillover_gamma is not None or spillover_range is not None:
        if interference_lambda != 1.0:
            raise ValueError('Random/fixed gamma cannot be combined with interference_lambda != 1.')
        if spillover_gamma is not None and spillover_range is not None:
            raise ValueError('Specify either spillover_gamma or spillover_range.')
        if spillover_gamma is not None and not math.isfinite(spillover_gamma):
            raise ValueError('spillover_gamma must be finite.')
        if spillover_range is not None:
            low, high = spillover_range
            if not all(math.isfinite(v) for v in (low, high)) or not low < high:
                raise ValueError('spillover_range must have finite increasing bounds.')
    ids = list(task_ids)
    if not ids or any(not isinstance(i, Integral) or isinstance(i, bool) or i < 0 or i >= 2**63 for i in ids):
        raise ValueError('task_ids must be nonempty nonnegative signed-64-bit integers.')
    if len(set(ids)) != len(ids):
        raise ValueError('task_ids must not contain duplicates.')
    if not isinstance(stream, str) or not stream:
        raise ValueError('stream must be a nonempty string.')
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError('seed must be an integer.')
    tasks = []
    for task_id in ids:
        rng = lambda channel: _task_generator(seed, stream, task_id, channel)
        if neighbor_range is not None:
            low, high = neighbor_range
            neighbor_coefficient = float(low+(high-low)*torch.rand((), generator=rng('outcome/beta')))
        elif neighbor_beta is not None:
            neighbor_coefficient = float(neighbor_beta)
        if spillover_range is not None:
            low, high = spillover_range
            spillover_coefficient = float(low+(high-low)*torch.rand((), generator=rng('outcome/gamma')))
        elif spillover_gamma is not None:
            spillover_coefficient = float(spillover_gamma)
        if fixed_adjacency is None:
            graph, graph_type, target_degree = _draw_graph(config, task_id, rng('graph'), balanced=balanced)
        else:
            from train_local_network_interference import ER_GRAPH
            graph, graph_type = fixed_adjacency, ER_GRAPH
            target_degree = (config.n_units - 1) * config.er_edge_probability
        x = torch.randn(config.n_units, generator=rng('data/normal'))
        neighbor_x = torch.mv(graph, x)/graph.sum(-1)
        exp_x = torch.exp(x)
        task = _assemble_task(
            x=x, adjacency=graph,
            treatment=(torch.rand(config.n_units, generator=rng('treatment')) < config.treatment_prob).float(),
            structural_baseline=1-x-(-neighbor_coefficient)*neighbor_x,
            tau=6+.2*exp_x, gamma=torch.full_like(x, spillover_coefficient),
            noise=NOISE_SD*torch.randn(config.n_units, generator=rng('noise/realization')),
            query_generator=rng('query'))
        task.update(prior_baseline_function=x, prior_neighbor_function=neighbor_x,
                    prior_direct_function=exp_x, prior_function_family=torch.tensor([0, 0, 4]),
                    prior_function_center=torch.zeros(3, dtype=torch.float64),
                    prior_function_scale=torch.ones(3, dtype=torch.float64))
        scalars = dict(prior_baseline_intercept=1., prior_baseline_amplitude=-1.,
            prior_neighbor_coefficient=neighbor_coefficient, prior_tau_level=6., prior_tau_amplitude=.2,
            prior_gamma=spillover_coefficient, prior_noise_sd=NOISE_SD, prior_target_mean_degree=target_degree,
            prior_effective_degree_min=config.effective_degree_bounds()[0],
            prior_effective_degree_max=config.effective_degree_bounds()[1])
        task.update({k: torch.tensor(v, dtype=torch.float32) for k, v in scalars.items()})
        integers = dict(task_id=task_id, regime=0 if neighbor_coefficient == spillover_coefficient == 0 else 3,
                        graph_type=graph_type, star_center=-1,
                        prior_interference_active=int(spillover_coefficient != 0),
                        prior_neighbor_active=int(neighbor_coefficient != 0),
                        task_seed=rng('identity').initial_seed())
        task.update({k: torch.tensor(v, dtype=torch.long) for k, v in integers.items()})
        if any(not bool(torch.isfinite(v).all()) for v in task.values()):
            raise FloatingPointError(f'Nonfinite A-group tensor: stream={stream!r}, task_id={task_id}.')
        tasks.append(task)
    return {k: torch.stack([task[k] for task in tasks]).to(device) for k in tasks[0]}
