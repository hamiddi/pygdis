from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .components import (
    JacobianFunction,
    attractor_expansion,
    entropy_complexity,
    jacobian_instability,
    stretching_rate,
    temporal_persistence,
)
from .potential import instability_potential, potential_to_gdis
from .result import GDISResult
from .scaling import EPS, hill_saturation, robust_scale, saturate_unit, smooth_series
from .transition import critical_window, transition_energy, transition_instability


@dataclass
class GDISConfig:
    alpha_j: float = 0.42
    alpha_s: float = 0.33
    alpha_a: float = 0.25
    k_j: float = 3.0
    k_s: float = 2.6
    k_a: float = 2.4
    hill_gamma: float = 0.72
    hill_c: float = 0.22
    complexity_gain: float = 0.055
    temporal_gain: float = 0.055
    temporal_threshold: float = 0.60
    transition_gain: float = 0.18
    transition_width_fraction: float = 0.09
    critical_value: Optional[float] = None
    smoothing_window: int = 9
    smoothing_polynomial_order: int = 3
    max_sustained: float = 0.985


class GDIS:
    """Generalized Dynamical Instability Score calculator."""

    def __init__(self, config: Optional[GDISConfig] = None, **overrides):
        self.config = config or GDISConfig()
        for key, value in overrides.items():
            if not hasattr(self.config, key):
                raise TypeError(f"Unknown GDIS configuration option: {key}")
            setattr(self.config, key, value)

    def fit_transform(
        self,
        trajectories: Sequence[np.ndarray],
        parameters: Sequence[float],
        jacobian_function: Optional[JacobianFunction] = None,
        critical_value: Optional[float] = None,
    ) -> GDISResult:
        if len(trajectories) != len(parameters):
            raise ValueError("The number of trajectories must equal the number of parameters.")
        if len(trajectories) < 3:
            raise ValueError("At least three parameter-ordered trajectories are required.")

        parameters_array = np.asarray(parameters, dtype=float)
        order = np.argsort(parameters_array)
        parameters_array = parameters_array[order]
        sorted_trajectories = [np.asarray(trajectories[i], dtype=float) for i in order]

        raw_j, raw_s, raw_a, raw_h, raw_t = [], [], [], [], []
        temporal_mean, temporal_fraction = [], []

        for trajectory, parameter in zip(sorted_trajectories, parameters_array):
            if trajectory.ndim == 1:
                trajectory = trajectory[:, None]
            if trajectory.ndim != 2 or len(trajectory) < 4:
                raise ValueError("Each trajectory must have shape (time, state_dimension) and at least four rows.")

            raw_j.append(jacobian_instability(trajectory, float(parameter), jacobian_function))
            raw_s.append(stretching_rate(trajectory))
            raw_a.append(attractor_expansion(trajectory))
            raw_h.append(entropy_complexity(trajectory))
            mean_value, fraction, combined = temporal_persistence(trajectory, self.config.temporal_threshold)
            temporal_mean.append(mean_value)
            temporal_fraction.append(fraction)
            raw_t.append(combined)

        raw_j = np.asarray(raw_j)
        raw_s = np.asarray(raw_s)
        raw_a = np.asarray(raw_a)
        raw_h = np.asarray(raw_h)
        raw_t = np.asarray(raw_t)

        scaled_j = robust_scale(np.log1p(raw_j))
        scaled_s = robust_scale(np.log1p(raw_s))
        scaled_a = robust_scale(np.log1p(raw_a))

        saturated_j = np.clip(smooth_series(saturate_unit(scaled_j, self.config.k_j), self.config.smoothing_window, self.config.smoothing_polynomial_order), 0.0, 1.0)
        saturated_s = np.clip(smooth_series(saturate_unit(scaled_s, self.config.k_s), self.config.smoothing_window, self.config.smoothing_polynomial_order), 0.0, 1.0)
        saturated_a = np.clip(smooth_series(saturate_unit(scaled_a, self.config.k_a), self.config.smoothing_window, self.config.smoothing_polynomial_order), 0.0, 1.0)

        alpha_sum = self.config.alpha_j + self.config.alpha_s + self.config.alpha_a
        core = ((saturated_j + EPS) ** self.config.alpha_j * (saturated_s + EPS) ** self.config.alpha_s * (saturated_a + EPS) ** self.config.alpha_a) ** (1.0 / alpha_sum)

        scaled_h = robust_scale(raw_h)
        scaled_t = robust_scale(raw_t)
        complexity_factor = 1.0 + self.config.complexity_gain * scaled_h
        temporal_factor = 1.0 + self.config.temporal_gain * scaled_t
        sustained_input = core * complexity_factor * temporal_factor

        sustained = hill_saturation(sustained_input, self.config.hill_gamma, self.config.hill_c)
        sustained = np.clip(smooth_series(sustained, self.config.smoothing_window, self.config.smoothing_polynomial_order), 0.0, self.config.max_sustained)

        energy = transition_energy(parameters_array, scaled_j, scaled_s, scaled_a)
        resolved_critical = critical_value if critical_value is not None else self.config.critical_value
        if resolved_critical is None:
            resolved_critical = float(parameters_array[np.argmax(energy)])
        parameter_range = float(np.max(parameters_array) - np.min(parameters_array))
        width = max(self.config.transition_width_fraction * parameter_range, EPS)
        window = critical_window(parameters_array, resolved_critical, width)
        transition = transition_instability(energy, window, self.config.transition_gain)

        potential = instability_potential(sustained, transition)
        gdis = np.clip(smooth_series(potential_to_gdis(potential), self.config.smoothing_window, self.config.smoothing_polynomial_order), 0.0, 1.0)

        components = {
            "jacobian_raw": raw_j,
            "stretching_raw": raw_s,
            "expansion_raw": raw_a,
            "entropy_raw": raw_h,
            "temporal_raw": raw_t,
            "temporal_mean": np.asarray(temporal_mean),
            "temporal_persistence": np.asarray(temporal_fraction),
            "jacobian_scaled": scaled_j,
            "stretching_scaled": scaled_s,
            "expansion_scaled": scaled_a,
            "jacobian_saturated": saturated_j,
            "stretching_saturated": saturated_s,
            "expansion_saturated": saturated_a,
            "core": core,
            "complexity_factor": complexity_factor,
            "temporal_factor": temporal_factor,
            "transition_energy": energy,
            "critical_window": window,
        }

        return GDISResult(parameters_array, gdis, potential, sustained, transition, components)
