# Baseline alignment — 2026-09-24

TNet: author-style 160 epochs, 1 base update + 50 targeted updates per epoch,
hidden width 64, density grid 20, spline basis 12, alpha .5, beta 20/sqrt(n),
Adam weight decay .001, GCN dropout .05, Normal(0,.1) weight initialization.
Normalize factual training Y and restore original units for predictions.
Preserve actual-arm propensity denominator, propensity floor, deterministic
inference and majority-arm estimands. This remains a benchmark adaptation.

F/L: unpenalized weighted least squares using SVD; return unsupported if the
requested contrast is not identifiable. No statistical ridge penalty.

reg-net: raw inverse-probability influence terms and unprojected HAC kernel
in the paper's variance objective. No ridge and no PSD projection in the
benchmark. If the empirical quadratic is unbounded, return unsupported (N/A).
The previous stabilized solver is available only as an explicit low-level mode.

NSI retains the documented single-cross-section adaptation. HyperSCI retains
the existing adaptation to the ordinary graph and majority exposure arms.
Localized DR-Lasso retains b_G=2 and its diagnostics; because d_1 <= .25 this
is a near-global bandwidth, not a new validation-selected local bandwidth.
These adaptations must not be described as exact reproductions of the papers.

PFN architecture, CEPO loss, training, DGP, and effect definitions are unchanged.
The reevaluation uses the existing PFN checkpoint, one seed=2026 test task,
and the same graph. Baselines are refitted; no PFN retraining is needed.
Previous result files are retained separately and are never relabeled as new.

Local validation: 412 tests passed, one existing PyTorch JIT deprecation warning.
No server/MUSA experiment has been started from this update.
