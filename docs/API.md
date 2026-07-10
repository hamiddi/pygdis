# pyGDIS API

## `gdis.GDIS`

```python
result = model.fit_transform(
    trajectories,
    parameters,
    jacobian_function=None,
    critical_value=None,
)
```

## `gdis.GDISConfig`

Controls channel exponents, saturation gains, Hill parameters, smoothing, and transition settings.

## `gdis.GDISResult`

```python
result.gdis
result.potential
result.sustained_instability
result.transition_instability
result.components
result.to_dataframe()
result.plot()
```

## Validation

- `best_threshold`
- `roc_auc`
- `safe_correlation`
- `validate_against_reference`

## Plotting

- `plot_gdis`
- `plot_gdis_vs_reference`
- `plot_components`
