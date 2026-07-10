from __future__ import annotations

import numpy as np

from .scaling import EPS, robust_scale, smooth_series


def transition_energy(parameters, jacobian_channel, stretching_channel, expansion_channel):
    parameters = np.asarray(parameters, dtype=float)
    d_j = robust_scale(np.abs(np.gradient(jacobian_channel, parameters)))
    d_s = robust_scale(np.abs(np.gradient(stretching_channel, parameters)))
    d_a = robust_scale(np.abs(np.gradient(expansion_channel, parameters)))
    energy = np.sqrt(d_j ** 2 + d_s ** 2 + d_a ** 2)
    return np.clip(smooth_series(robust_scale(energy)), 0.0, 1.0)


def critical_window(parameters, critical_value, width):
    parameters = np.asarray(parameters, dtype=float)
    width = max(float(width), EPS)
    return np.exp(-0.5 * ((parameters - critical_value) / width) ** 2)


def transition_instability(energy, window, gain):
    values = gain * np.asarray(energy, dtype=float) * np.asarray(window, dtype=float)
    return np.clip(smooth_series(values), 0.0, None)
