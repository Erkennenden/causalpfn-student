# Leung ANI baseline

This folder adapts the official replication functions `make_Zs` and `network_SE`
from <https://github.com/mpleung/ANI/blob/main/inference_module.py>.

`estimator.py` preserves the official statistical calculations while accepting a
NumPy adjacency matrix, which is the representation used by the PFN demo.
