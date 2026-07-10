from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy

from .scaling import EPS, robust_scale

JacobianFunction = Callable[[np.ndarray, float], np.ndarray]


def jacobian_instability(trajectory: np.ndarray, parameter: float, jacobian_function: Optional[JacobianFunction], stride: int = 10) -> float:
    """Average positive largest real Jacobian eigenvalue or data-only proxy."""
    trajectory = np.asarray(trajectory, dtype=float)
    if jacobian_function is None:
        return local_divergence_proxy(trajectory)
    values = []
    for state in trajectory[::stride]:
        jacobian = np.asarray(jacobian_function(state, parameter), dtype=float)
        eigenvalues = np.linalg.eigvals(jacobian)
        values.append(max(0.0, float(np.max(np.real(eigenvalues)))))
    return float(np.mean(values)) if values else 0.0


def local_divergence_proxy(trajectory: np.ndarray) -> float:
    trajectory = np.asarray(trajectory, dtype=float)
    if len(trajectory) < 4:
        return 0.0
    increments = np.linalg.norm(np.diff(trajectory, axis=0), axis=1) + EPS
    growth = np.log(increments[1:] / increments[:-1] + EPS)
    return float(np.mean(np.maximum(growth, 0.0)))


def stretching_rate(trajectory: np.ndarray) -> float:
    trajectory = np.asarray(trajectory, dtype=float)
    if len(trajectory) < 2:
        return 0.0
    derivative = np.gradient(trajectory, axis=0)
    return float(np.mean(np.linalg.norm(derivative, axis=1)))


def attractor_expansion(trajectory: np.ndarray) -> float:
    trajectory = np.asarray(trajectory, dtype=float)
    return 0.0 if len(trajectory) == 0 else float(np.linalg.norm(np.std(trajectory, axis=0)))


def spectral_entropy_1d(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 16 or np.std(signal) < EPS:
        return 0.0
    _, power = welch(signal, nperseg=min(512, len(signal)))
    power = np.maximum(power, EPS)
    power /= np.sum(power)
    return float(scipy_entropy(power) / (np.log(len(power)) + EPS))


def permutation_entropy_1d(signal: np.ndarray, order: int = 4, delay: int = 2) -> float:
    signal = np.asarray(signal, dtype=float)
    if len(signal) < order * delay:
        return 0.0
    patterns = {}
    for index in range(len(signal) - delay * (order - 1)):
        window = signal[index:index + delay * order:delay]
        pattern = tuple(np.argsort(window))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    counts = np.asarray(list(patterns.values()), dtype=float)
    if len(counts) == 0:
        return 0.0
    probabilities = counts / np.sum(counts)
    return float(scipy_entropy(probabilities) / (np.log(float(math.factorial(order))) + EPS))


def entropy_complexity(trajectory: np.ndarray) -> float:
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim == 1:
        trajectory = trajectory[:, None]
    values = []
    for column in range(trajectory.shape[1]):
        signal = trajectory[:, column]
        values.append(0.5 * spectral_entropy_1d(signal) + 0.5 * permutation_entropy_1d(signal))
    return float(np.mean(values)) if values else 0.0


def temporal_instability_signal(trajectory: np.ndarray) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=float)
    if len(trajectory) < 3:
        return np.zeros(len(trajectory), dtype=float)
    first = np.gradient(trajectory, axis=0)
    second = np.gradient(first, axis=0)
    speed = np.linalg.norm(first, axis=1)
    acceleration = np.linalg.norm(second, axis=1)
    radial = np.linalg.norm(trajectory - np.mean(trajectory, axis=0), axis=1)
    signal = 0.40 * robust_scale(speed) + 0.30 * robust_scale(acceleration) + 0.30 * robust_scale(radial)
    return np.clip(signal, 0.0, 1.0)


def temporal_persistence(trajectory: np.ndarray, threshold: float = 0.60) -> tuple[float, float, float]:
    signal = temporal_instability_signal(trajectory)
    if len(signal) == 0:
        return 0.0, 0.0, 0.0
    mean_value = float(np.mean(signal))
    persistence = float(np.mean(signal > threshold))
    return mean_value, persistence, 0.5 * mean_value + 0.5 * persistence
