from __future__ import annotations

import numpy as np


def trajectories_from_long_dataframe(dataframe, parameter_column, state_columns, time_column=None):
    missing = [c for c in [parameter_column, *state_columns] if c not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    trajectories, parameters = [], []
    for parameter, group in dataframe.groupby(parameter_column, sort=True):
        if time_column is not None:
            if time_column not in group.columns:
                raise ValueError(f"Missing time column: {time_column}")
            group = group.sort_values(time_column)
        trajectories.append(group[list(state_columns)].to_numpy(dtype=float))
        parameters.append(float(parameter))
    return trajectories, np.asarray(parameters, dtype=float)
