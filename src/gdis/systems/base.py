from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class ODESystem:
    name: str
    parameter_values: np.ndarray
    critical_value: float
    initial_state: np.ndarray
    rhs_function: Callable
    jacobian_function: Callable
    t0: float = 0.0
    t1: float = 80.0
    dt: float = 0.03
    transient_fraction: float = 0.50

    def simulate(self, parameter: float) -> np.ndarray:
        time = np.arange(self.t0, self.t1, self.dt)
        solution = solve_ivp(
            fun=lambda t, state: self.rhs_function(t, state, parameter),
            t_span=(self.t0, self.t1),
            y0=self.initial_state,
            t_eval=time,
            method="DOP853",
            rtol=1e-8,
            atol=1e-10,
        )
        if not solution.success:
            raise RuntimeError(f"{self.name} integration failed: {solution.message}")
        trajectory = solution.y.T
        start = int(len(trajectory) * self.transient_fraction)
        return trajectory[start:]

    def jacobian(self, state, parameter):
        return self.jacobian_function(state, parameter)

    def generate_sweep(self):
        trajectories = [self.simulate(float(p)) for p in self.parameter_values]
        return trajectories, self.parameter_values.copy()
