# E1.5 Star v2 presentation-only supplement

Run ID: e1_5_star_v2_test_20261031_01
Snapshot ID: 99c8b9c9a1e75a9fac4d86d491aaf4505a47cdc5518e11525fc3092e1f73113d

Inputs are the Gate 1.3 independently accepted verification copy. No formal metric, bootstrap distribution, model prediction or DGP was recomputed.

main_results.csv has 21 method-effect rows carrying 126 original metric values. main_results.tex contains three effect panels and a 9-row J interval summary. Absolute Bias differs from ACB; variance has squared outcome units. No ordinal ranking or rank shading is added.

J width is conditional on valid context posterior intervals. Width CIs are NOT_REQUESTED. Coverage CIs are the existing pointwise 95% mechanism-block Bootstrap CIs. The 90% target applies only to overall coverage point estimates, not center strata or CI bounds.

effects_true_vs_predicted.csv copies all 10800 saved method-context-effect records. Each supplies center, leaf mean and center-minus-leaf coordinates (32400 displayed points). The 200 mechanisms are independent; nested contexts are not independent mechanisms. No sampling, fitted regression or new confidence interval is used. Oracle and Hajek are not ranked.

Original make_figures.py styling and make_table.py names/latex_escape are reused unchanged. Their main/read_results/ate_table/build_figure/render_table entrypoints are not called: they require a different schema or would recompute statistics/add rankings. These are adapted views, not direct outputs of the legacy template runners.
