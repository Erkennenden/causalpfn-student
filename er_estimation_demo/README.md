# er_estimation_demo

See [README_CN.md](README_CN.md) for the current experiment and server commands.

Fixed ER(1000, 0.02, seed=12345), conditioned on no isolated nodes; a 10-layer Transformer; one treatment intercept a~Uniform(4,8) per dataset; four marginal CEPO GMMs; all baselines; ten held-out datasets on the same saved graph.

The DGP and graph changed. Start fresh training rather than reusing an earlier checkpoint.
