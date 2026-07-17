# pyGDIS API

## Core estimator

### `gdis.GDIS`

```python
from gdis import GDIS

result = GDIS().fit_transform(
    trajectories,
    parameters,
    jacobian_function=None,
    critical_value=None,
)
```

`trajectories` is a sequence of arrays with shape `(time, state_dimension)`. `parameters` must contain unique values and is sorted internally.

If `jacobian_function` is omitted, a data-only local-divergence proxy is used. If `critical_value` is omitted, the transition-energy maximum is used as a data-driven reference and recorded in `result.metadata`.

### `gdis.GDISConfig`

Controls the descriptor exponents, channel saturation gains, Hill parameters, modulation strengths, smoothing, and transition localization.

Preferred transition setting:

```python
from gdis import GDISConfig

config = GDISConfig(transition_weight=0.18)
```

`transition_gain` remains accepted as a deprecated constructor alias for compatibility with v0.1.0.

## Results

### `gdis.GDISResult`

```python
result.gdis
result.potential
result.sustained_instability
result.transition_instability
result.components
result.metadata
result.to_dataframe()
result.plot()
```

`result.components["transition_base"]` stores the unweighted transition-localization term.

## Sensitivity analysis

### `gdis.transition_weight_sensitivity`

```python
from gdis import transition_weight_sensitivity

table = transition_weight_sensitivity(
    result,
    weights=(0.0, 0.18, 0.25, 0.50, 0.75, 1.0),
)
```

This operation reuses the descriptor and sustained-instability calculations.

## Validation

From `gdis.validation`:

- `best_threshold`
- `roc_auc`
- `safe_correlation`
- `validate_against_reference`

Threshold optimization is an evaluation procedure, not part of the GDIS definition.

## Plotting

From `gdis.plotting`:

- `plot_gdis`
- `plot_gdis_vs_reference`
- `plot_components`

## Benchmarks

Canonical systems are available through `gdis.benchmarks` and are intentionally separate from the core estimator.
