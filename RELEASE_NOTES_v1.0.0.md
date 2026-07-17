# pyGDIS v1.0.0

This is the first publication-aligned release of pyGDIS and accompanies the manuscript **“Generalized Dynamical Instability Score (GDIS): A Universal Framework for Quantifying Instability and Critical Transitions in Nonlinear Dynamical Systems.”**

## Highlights

- Frozen reference implementation of the manuscript GDIS formulation.
- Reusable modules for descriptors, scaling, transition localization, potential construction, scoring, validation, sensitivity analysis, and plotting.
- Canonical Lorenz, Rössler, Chen, and logistic systems isolated as benchmarks.
- Complete scripts for reproducing the manuscript benchmark, transition-weight sensitivity analysis, and publication figures.
- Documentation synchronized with the strict bound `0 <= GDIS < 1` and the reference transition weight `lambda_t = 0.18`.
- Expanded tests and GitHub project templates.

## Compatibility note

The preferred configuration keyword is now `transition_weight`. The v0.1.0 keyword `transition_gain` remains accepted as an alias when passed to `GDIS(...)`.
