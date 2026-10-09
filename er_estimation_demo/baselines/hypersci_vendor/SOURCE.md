# HyperSCI source provenance

Jing Ma et al., Learning Causal Effects on Hypergraphs, KDD 2022.
Repository: https://github.com/jma712/HyperSCI
Pinned commit: ba33ef07b16e2110faabd127bb510aed63e29ea1

model.py contains the HyperSCI class from Model.py, with imports narrowed,
utils made package-relative and the module device explicitly set to CPU.
The model class body is unchanged. Only the supported one-layer, n_out=0,
attention, skip=123 configuration is instantiated by our adapter.
utils.py contains pdist, wasserstein and get_hyperedge_attr from utils.py.
Small-sample guards: retain a two-dimensional group of size one instead of
squeeze(); cap dropout probability at .99 when n_treated*n_control <= 10.
For the N=1000 pilot these guards do not change the original computation.
The original Wasserstein estimator (including gradient behavior) is retained.
This is a pair-edge / majority-estimand adaptation, not a replication of the
paper's datasets, default hyperparameters, or evaluation protocol.

Original Model.py SHA256: a5fc0efcaa45443faa11295554481bf478244a39b313fe5667923aa7423cb874
Original utils.py SHA256: e1c4b753b093d0596ae968a8cd63dd9ad74698d7dcd67298e054201bfacb6af3
PyG version used by the pilot and this release: 2.6.1.
