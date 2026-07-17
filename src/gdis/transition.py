"""Transition-energy and transition-localization utilities."""
from __future__ import annotations

import numpy as np

from .scaling import EPS, robust_scale, smooth_series


def transition_energy(parameters, jacobian_channel, stretching_channel, expansion_channel):
    """Return the normalized magnitude of descriptor variation over parameter space."""
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 1 or len(parameters) < 3:
        raise ValueError("parameters must be a one-dimensional sequence with at least three values.")
    if np.any(np.diff(parameters) <= 0):
        raise ValueError("parameters must be strictly increasing.")
    d_j = robust_scale(np.abs(np.gradient(jacobian_channel, parameters)))
    d_s = robust_scale(np.abs(np.gradient(stretching_channel, parameters)))
    d_a = robust_scale(np.abs(np.gradient(expansion_channel, parameters)))
    energy = np.sqrt(d_j**2 + d_s**2 + d_a**2)
    return np.clip(smooth_series(robust_scale(energy)), 0.0, 1.0)


def critical_window(parameters, critical_value, width):
    """Gaussian transition-localization window centered at ``critical_value``."""
    parameters = np.asarray(parameters, dtype=float)
    width = max(float(width), EPS)
    return np.exp(-0.5 * ((parameters - critical_value) / width) ** 2)


def transition_base(energy, window):
    """Construct the nonnegative transition term before applying ``lambda_t``."""
    values = np.asarray(energy, dtype=float) * np.asarray(window, dtype=float)
    return np.clip(smooth_series(values), 0.0, None)


def transition_instability(energy, window, gain):
    """Backward-compatible weighted transition term."""
    if gain < 0:
        raise ValueError("gain must be nonnegative.")
    return gain * transition_base(energy, window)
