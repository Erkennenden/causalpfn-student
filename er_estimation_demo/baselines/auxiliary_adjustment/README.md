# Auxiliary-variable adjustment baselines

Implemented methods:

- HT and Hajek;
- Fisher and Lin weighted regression;
- F-phi0(G1) and F-phi0(G2);
- ND-F, ND-L, ND-phi0(G1), reg-net;
- ND-G1 and ND-G2 as the paper's unnormalized diagnostics.

The implementation follows Sections 2--5 and equation (1) of *Adjusting auxiliary
variables under approximate neighborhood interference*. The exact ten-unit treatment
design is enumerated to calculate exposure probabilities and `phi0` expectations, so
there is no Monte Carlo error in `gamma_i` for this demo.

No public author GitHub repository for this paper was located during the revision;
these files therefore document and implement the published equations rather than
claiming source-code identity with an unavailable repository.
