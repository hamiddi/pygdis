import numpy as np
from .base import ODESystem


class ChenSystem(ODESystem):
    def __init__(self):
        a, b = 35.0, 3.0

        def rhs(t, state, c):
            x, y, z = state
            return np.array([a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z])

        def jacobian(state, c):
            x, y, z = state
            return np.array([[-a, a, 0.0], [c - a - z, c, -x], [y, x, -b]])

        super().__init__(
            name="Chen",
            parameter_values=np.linspace(5.0, 30.0, 81),
            critical_value=20.0,
            initial_state=np.array([0.1, 0.0, 0.0]),
            rhs_function=rhs,
            jacobian_function=jacobian,
        )
