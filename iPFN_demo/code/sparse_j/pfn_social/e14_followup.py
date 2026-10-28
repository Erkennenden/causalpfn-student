"""Independent replication and coordinate-only, degree-matched spatial graphs."""
from dataclasses import replace
import json
import networkx as nx
import numpy as np
from pfn_social.contracts import arm_mean_exposure
from pfn_social.e14_rgg import RGGConfig, SCENARIOS, draw_positions, generate_rgg, radius_graph
from pfn_social.phase3_hierarchical import _components, _hash_arrays
from pfn_social.phase5_large_network import rng_for
from pfn_social.phase6_limited_outcomes import LimitedOutcomeMechanism

REPLICATION = (replace(SCENARIOS[2], split_seed=2026102714),)
MATCHED = (RGGConfig('uniform', .18, 13, split_seed=2026102724),
           RGGConfig('clustered', .18, 14, split_seed=2026102724))
TARGET_DEGREE = 18
PAIRED_DGP_CONFIG = RGGConfig('uniform', .18, 90, split_seed=2026102724)


def matched_radius_graph(positions, target_degree=TARGET_DEGREE):
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    edges = n*target_degree/2
    if not edges.is_integer() or not 0 < edges < n*(n-1)/2:
        raise ValueError('Integer, non-extreme edge count required')
    edges = int(edges)
    i,j = np.triu_indices(n, 1)
    distances = np.linalg.norm(positions[i]-positions[j], axis=1)
    selected = np.partition(distances, (edges-1, edges))
    lo,hi = selected[edges-1],selected[edges]
    if not lo < hi:
        raise ValueError('Distance tie at cutoff')
    radius = float(lo+(hi-lo)/2)
    a = radius_graph(positions, radius)
    if a.sum() != 2*edges:
        raise ValueError('Incorrect edge count')
    return a,radius


def generate_matched(config, mechanism_id):
    rng = rng_for(config, mechanism_id, 'graph')
    for attempt in range(1, config.max_attempts+1):
        positions = draw_positions(config, rng)
        a,radius = matched_radius_graph(positions)
        if np.all(a.sum(1)>0):
            break
    else:
        raise RuntimeError('No isolate-free point cloud found')
    template,_,_ = generate_rgg(PAIRED_DGP_CONFIG, mechanism_id)
    parent = template.parent
    h,beta = parent.h,parent.coefficients
    components = _components(a,h,beta)
    b,tau,gamma,eta = components.T
    degree = a.sum(1)
    hbar = a@h/degree[:,None]
    truth = np.empty((config.n,4))
    for own,arm in ((0,0),(0,1),(1,0),(1,1)):
        exposure = np.array([arm_mean_exposure(int(d),.5,arm) for d in degree])
        truth[:,2*own+arm] = b+own*tau+exposure*(gamma+own*eta+hbar@(beta[19:22]+own*beta[22:25]))
    identity = np.frombuffer(json.dumps(dict(spatial=config.canonical_json(),target_degree=TARGET_DEGREE,
        selected_radius=radius,dgp=PAIRED_DGP_CONFIG.canonical_json()),sort_keys=True).encode(),dtype=np.uint8)
    # Keep the shared DGP namespace 90 for treatment/noise; runner labels families 13/14.
    parent = replace(parent,adjacency=a,components=components,truth=truth,
        mechanism_sha256=_hash_arrays(a,parent.x,beta,positions,identity),truth_sha256=_hash_arrays(truth))
    graph = nx.from_numpy_array(a)
    info = dict(family=config.setting_id,mechanism=mechanism_id,distribution=config.distribution,
        radius=radius,n=config.n,attempts=attempt,edges=graph.number_of_edges(),density=nx.density(graph),
        mean_degree=float(degree.mean()),min_degree=int(degree.min()),max_degree=int(degree.max()),
        components=nx.number_connected_components(graph),clustering=nx.average_clustering(graph),
        mechanism_sha256=parent.mechanism_sha256,truth_sha256=parent.truth_sha256,
        target_degree=TARGET_DEGREE,pair_id=mechanism_id,dgp_family=90)
    return LimitedOutcomeMechanism(parent,template.observation_permutation),positions,info
