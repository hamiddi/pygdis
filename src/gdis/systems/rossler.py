import numpy as np
from .base import ODESystem


class RosslerSystem(ODESystem):
    def __init__(self):
        a, b = 0.2, 0.2

        def rhs(t, state, c):
            x, y, z = state
            return np.array([-y - z, x + a * y, b + z * (x - c)])

        def jacobian(state, c):
            x, y, z = state
            return np.array([[0.0, -1.0, -1.0], [1.0, a, 0.0], [z, 0.0, x - c]])

        super().__init__(
            name="Rossler",
            parameter_values=np.linspace(2.0, 12.0, 81),
            critical_value=4.0,
            initial_state=np.array([1.0, 1.0, 1.0]),
            rhs_function=rhs,
            jacobian_function=jacobian,
        )
