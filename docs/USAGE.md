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
