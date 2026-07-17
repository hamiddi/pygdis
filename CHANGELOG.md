# Changelog

All notable changes to pyGDIS are documented here.

## [1.0.0]

- Added a complete long-format CSV dataset and data-only analysis example.
- Added CSV input documentation and generated result/figure workflow. - 2026-07-16

### Added
- Publication-aligned reference implementation of the five-descriptor GDIS framework.
- Explicit separation of the unweighted transition-localization term from the configurable transition weight `lambda_t`.
- Result metadata recording the critical-point source, reference weights, and transition weight.
- Transition-weight sensitivity utility that reuses computed descriptors.
- Formal property tests for boundedness, monotonicity, and baseline preservation.
- Manuscript-reproduction workflows for benchmark tables, sensitivity analysis, attractor galleries, and publication figures.
- Architecture figure, expanded mathematical documentation, security policy, and GitHub contribution templates.

### Changed
- Package version advanced from `0.1.0` to `1.0.0`.
- Documentation synchronized with the manuscript equations and strict range `0 <= GDIS < 1`.
- Canonical systems are exposed as optional benchmarks rather than embedded in the core score implementation.
- `transition_weight` replaces `transition_gain` as the preferred public configuration name; the old keyword remains a compatibility alias.

### Preserved
- Reference descriptor exponents: `alpha_j=0.42`, `alpha_s=0.33`, `alpha_a=0.25`.
- Hill parameters, saturation gains, complexity and temporal modulation defaults, and benchmark transition weight used by the manuscript.

## [0.1.0]

- Initial public prototype.
