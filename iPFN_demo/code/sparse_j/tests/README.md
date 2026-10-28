# Validation

Run `python -m pytest -q` from the sparse_j directory (pytest is a development
dependency, separate from the unchanged runtime requirements).

The tests cover the six new configurations at all registered phase5 sizes and
three seeds, deterministic replay, degree-preserving component joins, dense and
cycle boundaries, malformed/non-graphical inputs, registry identities, factual
storage isolation, actual frozen J/HT inference, and error/coverage aggregation.

`legacy_graph_hashes.json` contains adjacency fingerprints for original phase6
families 0–6 at mechanism 0. Before source changes, full mechanism/truth/factual
fingerprints were captured from upstream commit
`b3c008c3cf178b929681afd6be803a132eee5434`. After changes they were checked for
exact equality in the same Python 3.12 / NumPy 2.1.3 environment; the full hashes
are retained in `results/e12_v1/legacy_hashes.json`. Only integer graph hashes are
used as portable regression assertions; floating-point QR/BLAS-dependent hashes
are audit evidence for the recorded environment.

The original smoke command was also run before and after the refactor. Every
saved factual/truth/prediction array (including posterior intervals) and the
context evaluation CSV were equal. This checkout did not contain a pre-existing
test suite in the upstream archive.

Latest full run: 58 tests passed (73.95 seconds). Added acceptance tests cover
target/status/node ordering, [t,s] axes, no predictor call for NOT_IDENTIFIED,
explicit support masks and plug-in rejection, plus an independent enumeration
of perturbed DGP truth. Earlier result folders retain the test counts at their
own generation times. Statistical shift-screen failures are reported in the
results; passing software tests does not imply adequate coverage under shifts.
