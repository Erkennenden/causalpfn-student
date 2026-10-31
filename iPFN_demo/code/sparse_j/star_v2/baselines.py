"""Four ordinary joint-pair baselines; dispatch only, no numerical core."""
from types import MappingProxyType

from .contracts import ContractError
from .baselines_state_ht import estimate_state_mean, estimate_pair_ht
from .baselines_regression import estimate_f_z2, estimate_lin_z2

_ESTIMATORS = MappingProxyType({
    "state-mean": estimate_state_mean,
    "ht": estimate_pair_ht,
    "f-z2": estimate_f_z2,
    "lin-z2": estimate_lin_z2,
})
METHOD_IDS = tuple(_ESTIMATORS)


def estimate_pair(pair_task, request, method_id):
    """Return the original PairEstimateBundle from the selected implementation.

    PairTaskSpec and EstimationRequest retain all validation and ownership in
    existing contracts. Methods share no mutable execution state. This boundary
    accepts no privileged data or extra options; it does not retry, aggregate,
    discard failed results, or infer Identification.
    """
    if type(method_id) is not str or method_id not in _ESTIMATORS:
        raise ContractError("METHOD", "Unknown Stage-3 baseline method_id",
                            {"method_id": method_id, "allowed": METHOD_IDS})
    return _ESTIMATORS[method_id](request, pair_task)
