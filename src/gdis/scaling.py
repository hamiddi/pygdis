from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

EPS = 1e-12


def finite_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    fill = np.nanmedian(values[finite])
    return np.nan_to_num(
        values,
        nan=fill,
        posinf=np.nanmax(values[finite]),
        neginf=np.nanmin(values[finite]),
    )


def robust_scale(values: np.ndarray, low_percentile: float = 5.0, high_percentile: float = 95.0) -> np.ndarray:
    values = finite_array(values)
    low = np.percentile(values, low_percentile)
    high = np.percentile(values, high_percentile)
    if abs(high - low) < EPS:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def smooth_series(values: np.ndarray, window_length: int = 9, polynomial_order: int = 3) -> np.ndarray:
    values = finite_array(values)
    if len(values) < 5:
        return values
    window = min(window_length, len(values))
    if window % 2 == 0:
        window -= 1
    if window <= polynomial_order:
        return values
    try:
        return savgol_filter(values, window_length=window, polyorder=polynomial_order, mode="interp")
    except Exception:
        return values


def saturate_unit(values: np.ndarray, gain: float) -> np.ndarray:
    values = np.clip(finite_array(values), 0.0, 1.0)
    return 1.0 - np.exp(-gain * values)


def hill_saturation(values: np.ndarray, gamma: float, c: float) -> np.ndarray:
    values = np.clip(finite_array(values), 0.0, None)
    powered = np.power(values + EPS, gamma)
    return powered / (powered + c + EPS)


def bounded_from_potential(potential: np.ndarray) -> np.ndarray:
    potential = np.maximum(finite_array(potential), 0.0)
    return 1.0 - np.exp(-potential)
