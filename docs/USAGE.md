# Usage

## Input requirements

GDIS expects:

1. an ordered sequence of trajectories;
2. one control-parameter value for each trajectory;
3. optionally, a Jacobian function;
4. optionally, a known critical parameter value.

Each trajectory should have shape:

```text
(number_of_time_points, number_of_state_variables)
```

## Equation-aware mode

```python
from gdis import GDIS

result = GDIS().fit_transform(
    trajectories=trajectories,
    parameters=parameters,
    jacobian_function=jacobian,
    critical_value=critical_parameter,
)
```

The Jacobian callable has the form:

```python
def jacobian(state, parameter):
    return jacobian_matrix
```

## Data-only mode

```python
result = GDIS().fit_transform(
    trajectories=trajectories,
    parameters=parameters,
)
```

## Accessing outputs

```python
result.gdis
result.potential
result.sustained_instability
result.transition_instability
result.components
result.to_dataframe()
```

## Plotting

```python
from gdis.plotting import plot_gdis, plot_components

plot_gdis(result, output_path="gdis.png")
plot_components(result, output_path="components.png")
```

## Reading long-format CSV data

```python
import pandas as pd
from gdis.datasets import trajectories_from_long_dataframe

frame = pd.read_csv("data.csv")
trajectories, parameters = trajectories_from_long_dataframe(
    frame,
    parameter_column="rho",
    state_columns=["x", "y", "z"],
    time_column="time",
)
```

## Analyzing long-format CSV data

pyGDIS analyzes an **ordered family of trajectories**, not a single isolated
row or trajectory. A convenient CSV representation uses one row per time point:

```text
parameter,time,x1,x2
0.00,0.00,0.12,0.98
0.00,0.10,0.18,0.95
...
0.05,0.00,0.13,1.02
```

Required information:

- a control or operating parameter that identifies each trajectory;
- an optional time/sample-order column;
- one or more state-variable columns.

The repository includes `examples/synthetic_parameter_sweep.csv` and a
complete executable analysis in `examples/csv_data_example.py`.

```bash
cd examples
python csv_data_example.py
```

Equivalent Python code:

```python
import pandas as pd
from gdis import GDIS
from gdis.datasets import trajectories_from_long_dataframe

frame = pd.read_csv("examples/synthetic_parameter_sweep.csv")
trajectories, parameters = trajectories_from_long_dataframe(
    frame,
    parameter_column="parameter",
    time_column="time",
    state_columns=["x1", "x2"],
)

# Data-only mode: no governing equations or analytical Jacobian are required.
result = GDIS().fit_transform(trajectories, parameters)
results_table = result.to_dataframe()
results_table.to_csv("gdis_results.csv", index=False)
```

When `critical_value` is omitted, pyGDIS estimates the localization center from
the maximum transition-energy value and records
`data_driven_transition_energy_peak` in `result.metadata`. A known critical
value can instead be supplied explicitly:

```python
result = GDIS().fit_transform(
    trajectories,
    parameters,
    critical_value=0.55,
)
```

### Interpreting the result table

The main columns are:

| Column | Interpretation |
|---|---|
| `parameter` | Control/operating parameter for the trajectory |
| `gdis` | Final bounded instability score |
| `potential` | Generalized instability potential |
| `sustained_instability` | Persistent nonlinear-instability component |
| `transition_instability` | Localized transition contribution |
| `jacobian_raw` | Local-sensitivity estimate; a trajectory proxy in data-only mode |
| `stretching_raw` | Average trajectory motion |
| `expansion_raw` | Geometric spread of the trajectory |
| `entropy_raw` | Informational complexity |
| `temporal_raw` | Persistence of nonlinear activity |
| `transition_energy` | Rate of descriptor reorganization across the parameter sweep |

GDIS values should be interpreted **relative to the analyzed parameter family**,
because robust descriptor normalization uses cross-parameter context. The score
is most informative when trajectories are sampled consistently and the control
parameter spans stable, transitional, and unstable regimes.

