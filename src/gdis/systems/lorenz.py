import numpy as np
from .base import ODESystem


class LorenzSystem(ODESystem):
    def __init__(self):
        sigma = 10.0
        beta = 8.0 / 3.0

        def rhs(t, state, rho):
            x, y, z = state
            return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

        def jacobian(state, rho):
            x, y, z = state
            return np.array([[-sigma, sigma, 0.0], [rho - z, -1.0, -x], [y, x, -beta]])

        super().__init__(
            name="Lorenz",
            parameter_values=np.linspace(0.0, 60.0, 81),
            critical_value=24.74,
            initial_state=np.array([1.0, 1.0, 1.0]),
            rhs_function=rhs,
            jacobian_function=jacobian,
        )
