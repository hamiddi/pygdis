"""Reference implementation of the Generalized Dynamical Instability Score."""
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
from .transition import critical_window, transition_base, transition_energy


@dataclass
class GDISConfig:
    """Configuration for the manuscript reference formulation.

    The descriptor exponents are fixed reference parameters. ``transition_weight``
    is the coefficient :math:`\\lambda_t` used in the generalized instability
    potential. The deprecated alias ``transition_gain`` is accepted by
    :class:`GDIS` for compatibility with v0.1.0.
    """

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
    transition_weight: float = 0.18
    transition_width_fraction: float = 0.09
    critical_value: Optional[float] = None
    smoothing_window: int = 9
    smoothing_polynomial_order: int = 3
    max_sustained: float = 0.985

    def validate(self) -> None:
        weights = (self.alpha_j, self.alpha_s, self.alpha_a)
        if any(value <= 0 for value in weights):
            raise ValueError("alpha_j, alpha_s, and alpha_a must be positive.")
        if self.transition_weight < 0:
            raise ValueError("transition_weight must be nonnegative.")
        if not 0 < self.max_sustained < 1:
            raise ValueError("max_sustained must lie strictly between 0 and 1.")
        if self.hill_gamma <= 0 or self.hill_c <= 0:
            raise ValueError("hill_gamma and hill_c must be positive.")
        if self.transition_width_fraction <= 0:
            raise ValueError("transition_width_fraction must be positive.")


class GDIS:
    """Compute GDIS over an ordered family of trajectories.

    Notes
    -----
    GDIS is defined across a parameter-ordered trajectory family rather than a
    single isolated trajectory because robust normalization and transition
    energy require cross-parameter context.
    """

    def __init__(self, config: Optional[GDISConfig] = None, **overrides):
        self.config = config or GDISConfig()
        if "transition_gain" in overrides:
            if "transition_weight" in overrides:
                raise TypeError("Use only transition_weight; transition_gain is a deprecated alias.")
            overrides["transition_weight"] = overrides.pop("transition_gain")
        for key, value in overrides.items():
            if not hasattr(self.config, key):
                raise TypeError(f"Unknown GDIS configuration option: {key}")
            setattr(self.config, key, value)
        self.config.validate()

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
        if not np.all(np.isfinite(parameters_array)):
            raise ValueError("parameters must contain only finite values.")
        order = np.argsort(parameters_array)
        parameters_array = parameters_array[order]
        if np.any(np.diff(parameters_array) <= 0):
            raise ValueError("parameters must be unique.")
        sorted_trajectories = [np.asarray(trajectories[i], dtype=float) for i in order]

        raw_j, raw_s, raw_a, raw_h, raw_t = [], [], [], [], []
        temporal_mean, temporal_fraction = [], []
        for trajectory, parameter in zip(sorted_trajectories, parameters_array):
            if trajectory.ndim == 1:
                trajectory = trajectory[:, None]
            if trajectory.ndim != 2 or len(trajectory) < 4:
                raise ValueError("Each trajectory must have shape (time, state_dimension) and at least four rows.")
            if not np.all(np.isfinite(trajectory)):
                raise ValueError("Trajectories must contain only finite values.")
            raw_j.append(jacobian_instability(trajectory, float(parameter), jacobian_function))
            raw_s.append(stretching_rate(trajectory))
            raw_a.append(attractor_expansion(trajectory))
            raw_h.append(entropy_complexity(trajectory))
            mean_value, fraction, combined = temporal_persistence(trajectory, self.config.temporal_threshold)
            temporal_mean.append(mean_value)
            temporal_fraction.append(fraction)
            raw_t.append(combined)

        raw_j, raw_s, raw_a, raw_h, raw_t = map(np.asarray, (raw_j, raw_s, raw_a, raw_h, raw_t))
        scaled_j = robust_scale(np.log1p(raw_j))
        scaled_s = robust_scale(np.log1p(raw_s))
        scaled_a = robust_scale(np.log1p(raw_a))
        smooth = lambda x: smooth_series(x, self.config.smoothing_window, self.config.smoothing_polynomial_order)
        saturated_j = np.clip(smooth(saturate_unit(scaled_j, self.config.k_j)), 0.0, 1.0)
        saturated_s = np.clip(smooth(saturate_unit(scaled_s, self.config.k_s)), 0.0, 1.0)
        saturated_a = np.clip(smooth(saturate_unit(scaled_a, self.config.k_a)), 0.0, 1.0)

        alpha_sum = self.config.alpha_j + self.config.alpha_s + self.config.alpha_a
        core = (
            (saturated_j + EPS) ** self.config.alpha_j
            * (saturated_s + EPS) ** self.config.alpha_s
            * (saturated_a + EPS) ** self.config.alpha_a
        ) ** (1.0 / alpha_sum)

        scaled_h = robust_scale(raw_h)
        scaled_t = robust_scale(raw_t)
        complexity_factor = 1.0 + self.config.complexity_gain * scaled_h
        temporal_factor = 1.0 + self.config.temporal_gain * scaled_t
        sustained_input = core * complexity_factor * temporal_factor
        sustained = hill_saturation(sustained_input, self.config.hill_gamma, self.config.hill_c)
        sustained = np.clip(smooth(sustained), 0.0, self.config.max_sustained)

        energy = transition_energy(parameters_array, scaled_j, scaled_s, scaled_a)
        resolved_critical = critical_value if critical_value is not None else self.config.critical_value
        critical_source = "provided"
        if resolved_critical is None:
            resolved_critical = float(parameters_array[np.argmax(energy)])
            critical_source = "data_driven_transition_energy_peak"
        parameter_range = float(np.ptp(parameters_array))
        width = max(self.config.transition_width_fraction * parameter_range, EPS)
        window = critical_window(parameters_array, resolved_critical, width)
        base = transition_base(energy, window)
        transition = self.config.transition_weight * base

        potential = instability_potential(
            sustained,
            transition_base=base,
            transition_weight=self.config.transition_weight,
        )
        gdis = np.clip(smooth(potential_to_gdis(potential)), 0.0, 1.0 - EPS)

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
            "transition_base": base,
        }
        metadata = {
            "critical_value": float(resolved_critical),
            "critical_value_source": critical_source,
            "transition_weight": float(self.config.transition_weight),
            "reference_weights": {
                "alpha_j": self.config.alpha_j,
                "alpha_s": self.config.alpha_s,
                "alpha_a": self.config.alpha_a,
            },
        }
        return GDISResult(parameters_array, gdis, potential, sustained, transition, components, metadata)
