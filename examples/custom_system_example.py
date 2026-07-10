import numpy as np
from gdis import GDIS


def jacobian_function(state, parameter):
    x, y = state
    return np.array([[parameter - 3.0 * x * x, -1.0], [1.0, -0.2]])

parameters = np.linspace(0.0, 2.0, 20)
trajectories = []
for parameter in parameters:
    t = np.linspace(0.0, 50.0, 2000)
    trajectories.append(np.column_stack([np.sin((1.0 + parameter) * t), np.cos((1.0 + 0.5 * parameter) * t)]))

result = GDIS().fit_transform(trajectories, parameters, jacobian_function=jacobian_function)
print(result.to_dataframe().head())
