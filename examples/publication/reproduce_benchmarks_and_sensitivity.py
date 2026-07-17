#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27_run_gdis_benchmark_edition.py

GDIS Version 27 — Benchmark Edition
===================================

Purpose
-------
Version 27 freezes the Hill-potential GDIS formulation and evaluates it across
multiple benchmark nonlinear dynamical systems using the same mathematical
score construction.

Benchmarks included
-------------------
1. Lorenz system
2. Rössler system
3. Chen system
4. Logistic map

Core GDIS formulation
---------------------
For each system and each control parameter value, compute physical channels:

    J = local Jacobian instability / local expansion
    S = stretching / state-space speed or step variation
    A = attractor expansion / state spread
    H = entropy complexity
    T = temporal persistence

Then compute:

    core = (J^alpha_J * S^alpha_S * A^alpha_A)^(1 / sum(alpha))

    I_sustained = Hill(core * complexity_factor * temporal_factor)

    I_transition = lambda * transition_energy * critical_window

    Phi = -log(1 - I_sustained) + I_transition

    GDIS = 1 - exp(-Phi)

Important
---------
This script is a benchmark/validation edition. It does not tune GDIS separately
for each system. The same formulation is applied to all benchmarks.

Outputs
-------
results/gdis_v27_2_benchmark_stable_fast/
    benchmark_summary.csv
    all_system_results.csv
    per-system CSV files
    validation_metrics.csv
    classification_metrics.csv
    figures/*.png
    gdis_v27_benchmark_report.txt

Run
---
python 27_run_gdis_benchmark_edition.py
"""

from __future__ import annotations

import os
import math
import warnings
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.signal import welch, savgol_filter
from scipy.stats import entropy as scipy_entropy
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")


# ============================================================
# 1. Global configuration
# ============================================================

@dataclass
class GDISConfig:
    outdir: str = "results/gdis_v27_2_benchmark_stable_fast"

    # Integration
    t0: float = 0.0
    t1: float = 60.0
    dt: float = 0.05
    transient_fraction: float = 0.50

    # GDIS core exponents
    alpha_j: float = 0.42
    alpha_s: float = 0.33
    alpha_a: float = 0.25

    # Channel saturation gains
    k_j: float = 3.0
    k_s: float = 2.6
    k_a: float = 2.4

    # Hill saturation
    hill_gamma: float = 0.72
    hill_c: float = 0.22

    # Transition functional
    transition_lambda: float = 0.18
    transition_width_fraction: float = 0.09

    # Sensitivity analysis values for Table V. The baseline value 0.18 is
    # included explicitly so the published configuration is evaluated.
    transition_lambda_sensitivity: Tuple[float, ...] = (0.0, 0.18, 0.25, 0.50, 0.75, 1.00)

    # Modifiers
    complexity_gain: float = 0.055
    temporal_gain: float = 0.055
    temporal_threshold: float = 0.60

    # Numerical
    eps: float = 1e-12
    smooth_window: int = 9
    smooth_poly: int = 3
    max_sustained: float = 0.985

    # FTLE settings
    ftle_delta0: float = 1e-8
    ftle_renorm_time: float = 0.10
    ftle_total_time: float = 10.0
    ftle_transient_time: float = 3.0
    ftle_stride: int = 10
    compute_true_ftle_subset: bool = True

    # Logistic map
    logistic_n_iter: int = 5000
    logistic_transient: int = 2000

    # Plotting
    plot_dpi: int = 300
    example_count: int = 4

    # Noise robustness
    noise_levels: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.05)


# ============================================================
# 2. Utilities
# ============================================================

class Utils:
    def __init__(self, cfg: GDISConfig):
        self.cfg = cfg

    def finite_array(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        finite = np.isfinite(x)
        if not np.any(finite):
            return np.zeros_like(x)
        fill = np.nanmedian(x[finite])
        return np.nan_to_num(
            x,
            nan=fill,
            posinf=np.nanmax(x[finite]),
            neginf=np.nanmin(x[finite]),
        )

    def robust_scale(self, x: np.ndarray, low: float = 5.0, high: float = 95.0) -> np.ndarray:
        x = self.finite_array(x)
        lo = np.percentile(x, low)
        hi = np.percentile(x, high)
        if abs(hi - lo) < self.cfg.eps:
            return np.zeros_like(x)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    def smooth(self, x: np.ndarray) -> np.ndarray:
        x = self.finite_array(x)
        if len(x) < self.cfg.smooth_window:
            return x
        window = self.cfg.smooth_window
        if window % 2 == 0:
            window += 1
        try:
            return savgol_filter(
                x,
                window_length=window,
                polyorder=min(self.cfg.smooth_poly, window - 2),
                mode="interp",
            )
        except Exception:
            return x

    def saturate_unit(self, x: np.ndarray, gain: float) -> np.ndarray:
        x = np.clip(self.finite_array(x), 0.0, 1.0)
        return 1.0 - np.exp(-gain * x)

    def hill_saturation(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(self.finite_array(x), 0.0, None)
        xp = np.power(x + self.cfg.eps, self.cfg.hill_gamma)
        return xp / (xp + self.cfg.hill_c + self.cfg.eps)

    def bounded_from_potential(self, phi: np.ndarray) -> np.ndarray:
        phi = np.maximum(self.finite_array(phi), 0.0)
        return 1.0 - np.exp(-phi)

    def safe_corr(self, x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
        x = self.finite_array(x)
        y = self.finite_array(y)
        if np.std(x) < self.cfg.eps or np.std(y) < self.cfg.eps:
            return 0.0
        try:
            if method == "spearman":
                return float(spearmanr(x, y).correlation)
            return float(pearsonr(x, y)[0])
        except Exception:
            return 0.0

    def entropy_1d(self, x: np.ndarray, bins: int = 60) -> float:
        x = self.finite_array(x)
        if np.std(x) < self.cfg.eps:
            return 0.0
        hist, _ = np.histogram(x, bins=bins, density=False)
        p = hist.astype(float)
        p = p / (np.sum(p) + self.cfg.eps)
        p = p[p > 0]
        return float(scipy_entropy(p) / (np.log(len(hist)) + self.cfg.eps))

    def spectral_entropy_1d(self, x: np.ndarray) -> float:
        x = self.finite_array(x)
        if len(x) < 16 or np.std(x) < self.cfg.eps:
            return 0.0
        _, pxx = welch(x, nperseg=min(512, len(x)))
        pxx = np.maximum(pxx, self.cfg.eps)
        pxx = pxx / np.sum(pxx)
        return float(scipy_entropy(pxx) / (np.log(len(pxx)) + self.cfg.eps))

    def permutation_entropy_1d(self, x: np.ndarray, order: int = 4, delay: int = 2) -> float:
        x = self.finite_array(x)
        if len(x) < order * delay:
            return 0.0
        patterns: Dict[tuple, int] = {}
        for i in range(len(x) - delay * (order - 1)):
            window = x[i:i + delay * order:delay]
            pattern = tuple(np.argsort(window))
            patterns[pattern] = patterns.get(pattern, 0) + 1
        counts = np.asarray(list(patterns.values()), dtype=float)
        if len(counts) == 0:
            return 0.0
        p = counts / np.sum(counts)
        h = scipy_entropy(p)
        hmax = np.log(float(math.factorial(order)))
        return float(h / (hmax + self.cfg.eps))


# ============================================================
# 3. Benchmark system definitions
# ============================================================

@dataclass
class ODESystemSpec:
    name: str
    parameter_name: str
    parameter_values: np.ndarray
    critical_value: float
    x0: np.ndarray
    rhs: Callable[[float, np.ndarray, float], np.ndarray]
    jacobian: Callable[[np.ndarray, float], np.ndarray]
    classification_rule: Callable[[float], int]


@dataclass
class MapSystemSpec:
    name: str
    parameter_name: str
    parameter_values: np.ndarray
    critical_value: float
    classification_rule: Callable[[float], int]


def make_lorenz_spec() -> ODESystemSpec:
    sigma = 10.0
    beta = 8.0 / 3.0

    def rhs(t: float, s: np.ndarray, rho: float) -> np.ndarray:
        x, y, z = s
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ])

    def jac(s: np.ndarray, rho: float) -> np.ndarray:
        x, y, z = s
        return np.array([
            [-sigma, sigma, 0.0],
            [rho - z, -1.0, -x],
            [y, x, -beta],
        ])

    return ODESystemSpec(
        name="Lorenz",
        parameter_name="rho",
        parameter_values=np.linspace(0.0, 60.0, 81),
        critical_value=24.74,
        x0=np.array([1.0, 1.0, 1.0]),
        rhs=rhs,
        jacobian=jac,
        classification_rule=lambda p: int(p >= 24.74),
    )


def make_rossler_spec() -> ODESystemSpec:
    a = 0.2
    b = 0.2

    def rhs(t: float, s: np.ndarray, c: float) -> np.ndarray:
        x, y, z = s
        return np.array([
            -y - z,
            x + a * y,
            b + z * (x - c),
        ])

    def jac(s: np.ndarray, c: float) -> np.ndarray:
        x, y, z = s
        return np.array([
            [0.0, -1.0, -1.0],
            [1.0, a, 0.0],
            [z, 0.0, x - c],
        ])

    # Classical Rössler behavior becomes chaotic around c≈4 for a=b=0.2.
    return ODESystemSpec(
        name="Rossler",
        parameter_name="c",
        parameter_values=np.linspace(2.0, 12.0, 81),
        critical_value=4.0,
        x0=np.array([1.0, 1.0, 1.0]),
        rhs=rhs,
        jacobian=jac,
        classification_rule=lambda p: int(p >= 4.0),
    )


def make_chen_spec() -> ODESystemSpec:
    a = 35.0
    b = 3.0

    def rhs(t: float, s: np.ndarray, c: float) -> np.ndarray:
        x, y, z = s
        return np.array([
            a * (y - x),
            (c - a) * x - x * z + c * y,
            x * y - b * z,
        ])

    def jac(s: np.ndarray, c: float) -> np.ndarray:
        x, y, z = s
        return np.array([
            [-a, a, 0.0],
            [c - a - z, c, -x],
            [y, x, -b],
        ])

    # Standard chaotic Chen parameter is c=28.
    # Version 27.2 caps c at 30 because high-c Chen simulations can become very slow/stiff
    # in long sweeps and may appear to freeze on HPC sessions.
    # This benchmark uses c≈20 as a practical transition reference.
    return ODESystemSpec(
        name="Chen",
        parameter_name="c",
        parameter_values=np.linspace(5.0, 30.0, 81),
        critical_value=20.0,
        x0=np.array([0.1, 0.0, 0.0]),
        rhs=rhs,
        jacobian=jac,
        classification_rule=lambda p: int(p >= 20.0),
    )


def make_logistic_spec() -> MapSystemSpec:
    return MapSystemSpec(
        name="Logistic",
        parameter_name="r",
        parameter_values=np.linspace(2.5, 4.0, 101),
        critical_value=3.56995,
        classification_rule=lambda r: int(r >= 3.56995),
    )


# ============================================================
# 4. ODE benchmark engine
# ============================================================

class ODEBenchmarkEngine:
    def __init__(self, cfg: GDISConfig, utils: Utils):
        self.cfg = cfg
        self.utils = utils
        self.t_eval = np.arange(cfg.t0, cfg.t1, cfg.dt)

    def simulate(self, spec: ODESystemSpec, p: float) -> Tuple[np.ndarray, np.ndarray]:
        sol = solve_ivp(
            fun=lambda t, s: spec.rhs(t, s, p),
            t_span=(self.cfg.t0, self.cfg.t1),
            y0=spec.x0,
            t_eval=self.t_eval,
            method="DOP853",
            rtol=1e-7,
            atol=1e-9,
        )

        # Some parameter regions, especially Chen at high c, can become difficult
        # for an explicit solver. Fall back to LSODA before failing.
        if not sol.success:
            sol = solve_ivp(
                fun=lambda t, s: spec.rhs(t, s, p),
                t_span=(self.cfg.t0, self.cfg.t1),
                y0=spec.x0,
                t_eval=self.t_eval,
                method="LSODA",
                rtol=1e-7,
                atol=1e-9,
            )

        if not sol.success:
            print(f"  WARNING: {spec.name} integration failed at {p}: {sol.message}")
            t = self.t_eval
            X = np.full((len(t), len(spec.x0)), np.nan)
            start = int(len(t) * self.cfg.transient_fraction)
            return t[start:], X[start:]

        t = sol.t
        X = sol.y.T
        start = int(len(t) * self.cfg.transient_fraction)
        return t[start:], X[start:]

    def jacobian_instability(self, X: np.ndarray, spec: ODESystemSpec, p: float) -> float:
        vals = []
        X = self.utils.finite_array(X)
        for s in X[::10]:
            eig = np.linalg.eigvals(spec.jacobian(s, p))
            vals.append(max(0.0, np.max(np.real(eig))))
        return float(np.mean(vals))

    def stretching_rate(self, X: np.ndarray) -> float:
        X = self.utils.finite_array(X)
        dX = np.gradient(X, axis=0)
        return float(np.mean(np.linalg.norm(dX, axis=1)))

    def attractor_expansion(self, X: np.ndarray) -> float:
        X = self.utils.finite_array(X)
        return float(np.linalg.norm(np.std(X, axis=0)))

    def entropy_complexity(self, X: np.ndarray) -> float:
        vals = []
        for j in range(X.shape[1]):
            vals.append(0.5 * self.utils.spectral_entropy_1d(X[:, j]) +
                        0.5 * self.utils.permutation_entropy_1d(X[:, j]))
        return float(np.mean(vals))

    def temporal_signal(self, X: np.ndarray) -> np.ndarray:
        d1 = np.gradient(X, axis=0)
        d2 = np.gradient(d1, axis=0)
        speed = np.linalg.norm(d1, axis=1)
        accel = np.linalg.norm(d2, axis=1)
        radial = np.linalg.norm(X - np.mean(X, axis=0), axis=1)

        signal = (
            0.40 * self.utils.robust_scale(speed)
            + 0.30 * self.utils.robust_scale(accel)
            + 0.30 * self.utils.robust_scale(radial)
        )
        return np.clip(signal, 0.0, 1.0)

    def ftle(self, spec: ODESystemSpec, p: float) -> float:
        """
        Renormalized FTLE estimate for an ODE system.
        """
        delta0 = self.cfg.ftle_delta0
        renorm_dt = self.cfg.ftle_renorm_time
        total_time = self.cfg.ftle_total_time
        transient_time = self.cfg.ftle_transient_time

        x_ref = np.array(spec.x0, dtype=float)
        direction = np.zeros_like(x_ref)
        direction[0] = 1.0
        x_pert = x_ref + delta0 * direction

        t = 0.0
        log_sum = 0.0
        count = 0

        n_steps = int(total_time / renorm_dt)

        for _ in range(n_steps):
            sol_ref = solve_ivp(
                fun=lambda tt, s: spec.rhs(tt, s, p),
                t_span=(t, t + renorm_dt),
                y0=x_ref,
                method="DOP853",
                rtol=1e-8,
                atol=1e-10,
            )
            sol_pert = solve_ivp(
                fun=lambda tt, s: spec.rhs(tt, s, p),
                t_span=(t, t + renorm_dt),
                y0=x_pert,
                method="DOP853",
                rtol=1e-8,
                atol=1e-10,
            )

            x_ref = sol_ref.y[:, -1]
            x_pert = sol_pert.y[:, -1]

            diff = x_pert - x_ref
            dist = np.linalg.norm(diff)

            if dist < self.cfg.eps:
                diff = direction * delta0
                dist = delta0

            if t >= transient_time:
                log_sum += np.log(dist / delta0)
                count += 1

            diff_unit = diff / (dist + self.cfg.eps)
            x_pert = x_ref + delta0 * diff_unit
            t += renorm_dt

        if count == 0:
            return 0.0

        return float(log_sum / (count * renorm_dt))

    def extract(self, spec: ODESystemSpec) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[float, np.ndarray]]:
        rows = []
        temporal_rows = []
        bif_rows = []
        examples: Dict[float, np.ndarray] = {}

        example_values = np.linspace(
            spec.parameter_values.min(),
            spec.parameter_values.max(),
            self.cfg.example_count,
        )

        for i, p in enumerate(spec.parameter_values):
            print(f"  {spec.name}: {spec.parameter_name} {i + 1}/{len(spec.parameter_values)} = {p:.4f}")

            t, X = self.simulate(spec, float(p))

            J = self.jacobian_instability(X, spec, float(p))
            S = self.stretching_rate(X)
            A = self.attractor_expansion(X)
            H = self.entropy_complexity(X)
            temporal = self.temporal_signal(X)
            T_mean = float(np.mean(temporal))
            T_persistence = float(np.mean(temporal > self.cfg.temporal_threshold))
            T = 0.5 * T_mean + 0.5 * T_persistence
            # True FTLE is the most expensive validation metric.
            # In benchmark mode, compute it on a subset and interpolate later.
            if (not self.cfg.compute_true_ftle_subset) or (i % self.cfg.ftle_stride == 0) or (i == len(spec.parameter_values) - 1):
                ftle = self.ftle(spec, float(p))
            else:
                ftle = np.nan

            rows.append({
                "system": spec.name,
                "parameter": float(p),
                "parameter_name": spec.parameter_name,
                "critical_value": spec.critical_value,
                "label_unstable": spec.classification_rule(float(p)),
                "J_raw": J,
                "S_raw": S,
                "A_raw": A,
                "H_raw": H,
                "T_mean": T_mean,
                "T_persistence": T_persistence,
                "T_raw": T,
                "FTLE": ftle,
            })

            for tt, vv in zip(t[::200], temporal[::200]):
                temporal_rows.append({
                    "system": spec.name,
                    "parameter": float(p),
                    "time": float(tt),
                    "temporal_signal": float(vv),
                })

            for xval in X[-2000::10, 0]:
                bif_rows.append({
                    "system": spec.name,
                    "parameter": float(p),
                    "x": float(xval),
                })

            nearest_ex = example_values[np.argmin(np.abs(example_values - p))]
            if abs(nearest_ex - p) <= (spec.parameter_values[1] - spec.parameter_values[0]) / 2:
                examples[float(p)] = X[::10]

        return pd.DataFrame(rows), pd.DataFrame(temporal_rows), pd.DataFrame(bif_rows), examples


# ============================================================
# 5. Logistic map benchmark engine
# ============================================================

class LogisticBenchmarkEngine:
    def __init__(self, cfg: GDISConfig, utils: Utils):
        self.cfg = cfg
        self.utils = utils

    def simulate(self, r: float, x0: float = 0.1234567) -> np.ndarray:
        n_total = self.cfg.logistic_n_iter
        xs = np.zeros(n_total, dtype=float)
        xs[0] = x0

        for i in range(n_total - 1):
            xs[i + 1] = r * xs[i] * (1.0 - xs[i])
            if not np.isfinite(xs[i + 1]):
                xs[i + 1] = 0.5

        return xs[self.cfg.logistic_transient:]

    def logistic_ftle(self, r: float, x: np.ndarray) -> float:
        deriv = np.abs(r * (1.0 - 2.0 * x)) + self.cfg.eps
        return float(np.mean(np.log(deriv)))

    def extract(self, spec: MapSystemSpec) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rows = []
        temporal_rows = []
        bif_rows = []

        for i, r in enumerate(spec.parameter_values):
            print(f"  {spec.name}: {spec.parameter_name} {i + 1}/{len(spec.parameter_values)} = {r:.4f}")

            x = self.simulate(float(r))

            deriv_log = np.log(np.abs(r * (1.0 - 2.0 * x)) + self.cfg.eps)

            J = float(np.mean(np.maximum(deriv_log, 0.0)))
            S = float(np.std(np.diff(x)))
            A = float(np.std(x))
            H = 0.5 * self.utils.entropy_1d(x) + 0.5 * self.utils.permutation_entropy_1d(x)
            temporal = self.utils.robust_scale(np.abs(np.diff(x, prepend=x[0])))
            T_mean = float(np.mean(temporal))
            T_persistence = float(np.mean(temporal > self.cfg.temporal_threshold))
            T = 0.5 * T_mean + 0.5 * T_persistence
            ftle = self.logistic_ftle(float(r), x)

            rows.append({
                "system": spec.name,
                "parameter": float(r),
                "parameter_name": spec.parameter_name,
                "critical_value": spec.critical_value,
                "label_unstable": spec.classification_rule(float(r)),
                "J_raw": J,
                "S_raw": S,
                "A_raw": A,
                "H_raw": H,
                "T_mean": T_mean,
                "T_persistence": T_persistence,
                "T_raw": T,
                "FTLE": ftle,
            })

            for k, vv in enumerate(temporal[::20]):
                temporal_rows.append({
                    "system": spec.name,
                    "parameter": float(r),
                    "time": float(k),
                    "temporal_signal": float(vv),
                })

            for xv in x[-800::4]:
                bif_rows.append({
                    "system": spec.name,
                    "parameter": float(r),
                    "x": float(xv),
                })

        return pd.DataFrame(rows), pd.DataFrame(temporal_rows), pd.DataFrame(bif_rows)


# ============================================================
# 6. Generic GDIS scorer
# ============================================================

class GenericGDISScorer:
    def __init__(self, cfg: GDISConfig, utils: Utils):
        self.cfg = cfg
        self.utils = utils

    def score_one_system(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        p = out["parameter"].values
        p_min = float(np.min(p))
        p_max = float(np.max(p))
        width = max(self.cfg.transition_width_fraction * (p_max - p_min), self.cfg.eps)

        out["J_scaled"] = self.utils.robust_scale(np.log1p(out["J_raw"].values))
        out["S_scaled"] = self.utils.robust_scale(np.log1p(out["S_raw"].values))
        out["A_scaled"] = self.utils.robust_scale(np.log1p(out["A_raw"].values))

        out["J_sat"] = np.clip(
            self.utils.smooth(self.utils.saturate_unit(out["J_scaled"].values, self.cfg.k_j)),
            0.0,
            1.0,
        )
        out["S_sat"] = np.clip(
            self.utils.smooth(self.utils.saturate_unit(out["S_scaled"].values, self.cfg.k_s)),
            0.0,
            1.0,
        )
        out["A_sat"] = np.clip(
            self.utils.smooth(self.utils.saturate_unit(out["A_scaled"].values, self.cfg.k_a)),
            0.0,
            1.0,
        )

        p_sum = self.cfg.alpha_j + self.cfg.alpha_s + self.cfg.alpha_a

        out["core_raw"] = (
            (out["J_sat"].values + self.cfg.eps) ** self.cfg.alpha_j
            * (out["S_sat"].values + self.cfg.eps) ** self.cfg.alpha_s
            * (out["A_sat"].values + self.cfg.eps) ** self.cfg.alpha_a
        ) ** (1.0 / p_sum)

        out["H_scaled"] = self.utils.robust_scale(out["H_raw"].values)
        out["T_scaled"] = self.utils.robust_scale(out["T_raw"].values)

        out["complexity_factor"] = 1.0 + self.cfg.complexity_gain * out["H_scaled"].values
        out["temporal_factor"] = 1.0 + self.cfg.temporal_gain * out["T_scaled"].values

        sustained_input = out["core_raw"].values * out["complexity_factor"].values * out["temporal_factor"].values
        out["I_sustained"] = self.utils.hill_saturation(sustained_input)
        out["I_sustained"] = np.clip(
            self.utils.smooth(out["I_sustained"].values),
            0.0,
            self.cfg.max_sustained,
        )

        out["dJ_dp"] = self.utils.robust_scale(np.abs(np.gradient(out["J_scaled"].values, p)))
        out["dS_dp"] = self.utils.robust_scale(np.abs(np.gradient(out["S_scaled"].values, p)))
        out["dA_dp"] = self.utils.robust_scale(np.abs(np.gradient(out["A_scaled"].values, p)))

        transition_energy = np.sqrt(
            out["dJ_dp"].values ** 2
            + out["dS_dp"].values ** 2
            + out["dA_dp"].values ** 2
        )

        out["transition_energy"] = np.clip(
            self.utils.smooth(self.utils.robust_scale(transition_energy)),
            0.0,
            1.0,
        )

        critical_value = float(out["critical_value"].iloc[0])
        out["critical_window"] = np.exp(-0.5 * ((p - critical_value) / width) ** 2)

        # Store the transition contribution before applying its global weight.
        # This allows Table V to be generated without recomputing trajectories
        # or descriptors for every candidate value of lambda_t.
        out["transition_base"] = np.clip(
            self.utils.smooth(
                out["transition_energy"].values
                * out["critical_window"].values
            ),
            0.0,
            None,
        )
        out["I_transition"] = self.cfg.transition_lambda * out["transition_base"].values

        out["Phi_sustained"] = -np.log(1.0 - out["I_sustained"].values + self.cfg.eps)
        out["Phi"] = out["Phi_sustained"].values + out["I_transition"].values
        out["GDIS"] = self.utils.bounded_from_potential(out["Phi"].values)
        out["GDIS"] = np.clip(self.utils.smooth(out["GDIS"].values), 0.0, 1.0)

        out["FTLE_scaled"] = self.utils.robust_scale(out["FTLE"].values)
        out["Phi_scaled"] = self.utils.robust_scale(out["Phi"].values)

        return out

    def score_all(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for system, sub in df.groupby("system", sort=False):
            parts.append(self.score_one_system(sub.reset_index(drop=True)))
        return pd.concat(parts, ignore_index=True)


# ============================================================
# 7. Validation metrics
# ============================================================

class Metrics:
    def __init__(self, cfg: GDISConfig, utils: Utils):
        self.cfg = cfg
        self.utils = utils

    def best_threshold_metrics(self, y_true: np.ndarray, score: np.ndarray) -> Dict[str, float]:
        thresholds = np.linspace(0.0, 1.0, 501)
        best = None

        for th in thresholds:
            y_pred = (score >= th).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            sens = tp / (tp + fn + self.cfg.eps)
            spec = tn / (tn + fp + self.cfg.eps)
            acc = (tp + tn) / len(y_true)
            bal = 0.5 * (sens + spec)

            row = {
                "threshold": th,
                "accuracy": acc,
                "balanced_accuracy": bal,
                "sensitivity": sens,
                "specificity": spec,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }

            if best is None or row["balanced_accuracy"] > best["balanced_accuracy"]:
                best = row

        assert best is not None
        return best

    def roc_auc(self, y_true: np.ndarray, score: np.ndarray) -> float:
        """Compute binary ROC AUC using the Mann-Whitney rank statistic."""
        y_true = np.asarray(y_true, dtype=int)
        score = self.utils.finite_array(score)
        pos = y_true == 1
        neg = y_true == 0
        n_pos = int(np.sum(pos))
        n_neg = int(np.sum(neg))
        if n_pos == 0 or n_neg == 0:
            return float("nan")

        # Average ranks for ties, implemented with pandas to avoid adding a
        # new scikit-learn dependency.
        ranks = pd.Series(score).rank(method="average").to_numpy()
        rank_sum_pos = float(np.sum(ranks[pos]))
        auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)

    def validation_tables(self, scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        system_rows = []
        regime_rows = []
        corr_rows = []

        for system, sub in scored.groupby("system", sort=False):
            y_true = sub["label_unstable"].values.astype(int)
            score = sub["GDIS"].values
            ftle = sub["FTLE_scaled"].values

            best = self.best_threshold_metrics(y_true, score)

            system_rows.append({
                "system": system,
                "n": len(sub),
                "critical_value": float(sub["critical_value"].iloc[0]),
                "best_threshold": best["threshold"],
                "accuracy": best["accuracy"],
                "balanced_accuracy": best["balanced_accuracy"],
                "sensitivity": best["sensitivity"],
                "specificity": best["specificity"],
                "GDIS_mean": float(np.mean(score)),
                "GDIS_std": float(np.std(score)),
                "GDIS_FTLE_pearson": self.utils.safe_corr(score, ftle, "pearson"),
                "GDIS_FTLE_spearman": self.utils.safe_corr(score, ftle, "spearman"),
                "GDIS_Phi_pearson": self.utils.safe_corr(score, sub["Phi"].values, "pearson"),
                "GDIS_I_sustained_pearson": self.utils.safe_corr(score, sub["I_sustained"].values, "pearson"),
                "GDIS_I_transition_pearson": self.utils.safe_corr(score, sub["I_transition"].values, "pearson"),
            })

            for label, g in sub.groupby("label_unstable"):
                regime_rows.append({
                    "system": system,
                    "label_unstable": int(label),
                    "GDIS_mean": float(g["GDIS"].mean()),
                    "GDIS_std": float(g["GDIS"].std()),
                    "FTLE_mean": float(g["FTLE"].mean()),
                    "I_sustained_mean": float(g["I_sustained"].mean()),
                    "I_transition_mean": float(g["I_transition"].mean()),
                    "n": len(g),
                })

            for comp in ["J_scaled", "S_scaled", "A_scaled", "FTLE_scaled", "Phi", "I_sustained", "I_transition"]:
                corr_rows.append({
                    "system": system,
                    "comparison": f"GDIS_vs_{comp}",
                    "pearson": self.utils.safe_corr(score, sub[comp].values, "pearson"),
                    "spearman": self.utils.safe_corr(score, sub[comp].values, "spearman"),
                })

        overall_best = self.best_threshold_metrics(
            scored["label_unstable"].values.astype(int),
            scored["GDIS"].values,
        )

        system_rows.append({
            "system": "ALL",
            "n": len(scored),
            "critical_value": np.nan,
            "best_threshold": overall_best["threshold"],
            "accuracy": overall_best["accuracy"],
            "balanced_accuracy": overall_best["balanced_accuracy"],
            "sensitivity": overall_best["sensitivity"],
            "specificity": overall_best["specificity"],
            "GDIS_mean": float(scored["GDIS"].mean()),
            "GDIS_std": float(scored["GDIS"].std()),
            "GDIS_FTLE_pearson": self.utils.safe_corr(scored["GDIS"], scored["FTLE_scaled"], "pearson"),
            "GDIS_FTLE_spearman": self.utils.safe_corr(scored["GDIS"], scored["FTLE_scaled"], "spearman"),
            "GDIS_Phi_pearson": self.utils.safe_corr(scored["GDIS"], scored["Phi"], "pearson"),
            "GDIS_I_sustained_pearson": self.utils.safe_corr(scored["GDIS"], scored["I_sustained"], "pearson"),
            "GDIS_I_transition_pearson": self.utils.safe_corr(scored["GDIS"], scored["I_transition"], "pearson"),
        })

        return pd.DataFrame(system_rows), pd.DataFrame(regime_rows), pd.DataFrame(corr_rows)


# ============================================================
# 8. Transition-weight sensitivity analysis (Table V)
# ============================================================

class TransitionWeightSensitivity:
    """Recompute only the final GDIS aggregation for candidate lambda_t values."""

    def __init__(self, cfg: GDISConfig, utils: Utils, metrics: Metrics):
        self.cfg = cfg
        self.utils = utils
        self.metrics = metrics

    def _rescore(self, scored: pd.DataFrame, lambda_t: float) -> pd.DataFrame:
        parts = []
        for _system, sub in scored.groupby("system", sort=False):
            out = sub.copy().reset_index(drop=True)
            out["I_transition_sensitivity"] = (
                float(lambda_t) * out["transition_base"].values
            )
            out["Phi_sensitivity"] = (
                out["Phi_sustained"].values
                + out["I_transition_sensitivity"].values
            )
            gdis = self.utils.bounded_from_potential(out["Phi_sensitivity"].values)
            out["GDIS_sensitivity"] = np.clip(
                self.utils.smooth(gdis), 0.0, 1.0
            )
            parts.append(out)
        return pd.concat(parts, ignore_index=True)

    def evaluate(self, scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        overall_rows = []
        system_rows = []

        for lambda_t in self.cfg.transition_lambda_sensitivity:
            rescored = self._rescore(scored, float(lambda_t))

            # Per-system metrics are saved as a detailed supplement.
            for system, sub in rescored.groupby("system", sort=False):
                y_true = sub["label_unstable"].values.astype(int)
                score = sub["GDIS_sensitivity"].values
                best = self.metrics.best_threshold_metrics(y_true, score)
                system_rows.append({
                    "lambda_t": float(lambda_t),
                    "system": system,
                    "best_threshold": best["threshold"],
                    "accuracy": best["accuracy"],
                    "balanced_accuracy": best["balanced_accuracy"],
                    "auc": self.metrics.roc_auc(y_true, score),
                    "pearson_GDIS_FTLE": self.utils.safe_corr(
                        score, sub["FTLE_scaled"].values, "pearson"
                    ),
                    "spearman_GDIS_FTLE": self.utils.safe_corr(
                        score, sub["FTLE_scaled"].values, "spearman"
                    ),
                })

            # Table V reports pooled performance across all benchmark systems.
            y_true = rescored["label_unstable"].values.astype(int)
            score = rescored["GDIS_sensitivity"].values
            best = self.metrics.best_threshold_metrics(y_true, score)
            overall_rows.append({
                "lambda_t": float(lambda_t),
                "best_threshold": best["threshold"],
                "accuracy": best["accuracy"],
                "balanced_accuracy": best["balanced_accuracy"],
                "auc": self.metrics.roc_auc(y_true, score),
                "pearson_GDIS_FTLE": self.utils.safe_corr(
                    score, rescored["FTLE_scaled"].values, "pearson"
                ),
                "spearman_GDIS_FTLE": self.utils.safe_corr(
                    score, rescored["FTLE_scaled"].values, "spearman"
                ),
            })

        return pd.DataFrame(overall_rows), pd.DataFrame(system_rows)

    def save_outputs(
        self,
        table_v: pd.DataFrame,
        detail: pd.DataFrame,
        outdir: str,
        figdir: str,
    ) -> None:
        csv_path = os.path.join(outdir, "Table_V_transition_weight_sensitivity.csv")
        detail_path = os.path.join(outdir, "Table_V_transition_weight_sensitivity_by_system.csv")
        tex_path = os.path.join(outdir, "Table_V_transition_weight_sensitivity.tex")

        table_v.to_csv(csv_path, index=False)
        detail.to_csv(detail_path, index=False)

        display_df = table_v.rename(columns={
            "lambda_t": r"$\lambda_t$",
            "best_threshold": "Threshold",
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced Accuracy",
            "auc": "AUC",
            "pearson_GDIS_FTLE": "Pearson (GDIS--FTLE)",
            "spearman_GDIS_FTLE": "Spearman (GDIS--FTLE)",
        })
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(display_df.to_latex(
                index=False,
                float_format=lambda x: f"{x:.3f}",
                escape=False,
                caption=r"Sensitivity of GDIS to the transition weight $\lambda_t$.",
                label="tab:transition_weight_sensitivity",
            ))

        # Compact publication figure using matplotlib defaults.
        plt.figure(figsize=(7.2, 4.8))
        x = table_v["lambda_t"].values
        for col, label in [
            ("accuracy", "Accuracy"),
            ("balanced_accuracy", "Balanced accuracy"),
            ("auc", "AUC"),
            ("pearson_GDIS_FTLE", "Pearson"),
            ("spearman_GDIS_FTLE", "Spearman"),
        ]:
            plt.plot(x, table_v[col].values, marker="o", linewidth=1.8, label=label)
        plt.axvline(
            self.cfg.transition_lambda, linestyle="--", linewidth=1.2,
            label=rf"Baseline $\lambda_t$={self.cfg.transition_lambda:g}"
        )
        plt.xlabel(r"Transition weight $\lambda_t$")
        plt.ylabel("Performance metric")
        plt.ylim(0.0, 1.05)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(figdir, "Figure_16_transition_weight_sensitivity.png"),
            dpi=self.cfg.plot_dpi,
            bbox_inches="tight",
        )
        plt.close()


# ============================================================
# 9. Plotting
# ============================================================

class Plotter:
    def __init__(self, cfg: GDISConfig, figdir: str):
        self.cfg = cfg
        self.figdir = figdir

    def save(self, name: str) -> None:
        plt.tight_layout()
        plt.savefig(os.path.join(self.figdir, name), dpi=self.cfg.plot_dpi)
        plt.close()

    def plot_all_gdis(self, scored: pd.DataFrame) -> None:
        plt.figure(figsize=(14, 8))
        for system, sub in scored.groupby("system", sort=False):
            x = sub["parameter"].values
            x_norm = (x - x.min()) / (x.max() - x.min() + self.cfg.eps)
            plt.plot(x_norm, sub["GDIS"].values, label=system, lw=2)
        plt.xlabel("normalized control parameter")
        plt.ylabel("GDIS")
        plt.title("GDIS Version 27 Benchmark Across Systems")
        plt.legend()
        self.save("v27_all_systems_gdis.png")

    def plot_system_gdis(self, sub: pd.DataFrame) -> None:
        system = sub["system"].iloc[0]
        pname = sub["parameter_name"].iloc[0]
        crit = float(sub["critical_value"].iloc[0])

        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(sub["parameter"], sub["GDIS"], label="GDIS", lw=2.2)
        ax1.plot(sub["parameter"], sub["I_sustained"], label="I_sustained", lw=2)
        ax1.plot(sub["parameter"], sub["I_transition"], label="I_transition", lw=2)
        ax1.axvline(crit, linestyle="--", label="reference transition")
        ax1.set_xlabel(pname)
        ax1.set_ylabel("score")

        ax2 = ax1.twinx()
        ax2.plot(sub["parameter"], sub["Phi"], "--", color="black", label="Phi", lw=1.6)
        ax2.set_ylabel("Potential Phi")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        plt.title(f"{system}: GDIS and potential")
        self.save(f"v27_{system}_gdis.png")

    def plot_system_ftle(self, sub: pd.DataFrame) -> None:
        system = sub["system"].iloc[0]
        pname = sub["parameter_name"].iloc[0]
        crit = float(sub["critical_value"].iloc[0])

        plt.figure(figsize=(14, 6))
        plt.plot(sub["parameter"], sub["GDIS"], label="GDIS", lw=2.2)
        plt.plot(sub["parameter"], sub["FTLE_scaled"], label="FTLE scaled", lw=2)
        plt.axvline(crit, linestyle="--", label="reference transition")
        plt.xlabel(pname)
        plt.ylabel("scaled value")
        plt.title(f"{system}: GDIS vs FTLE")
        plt.legend()
        self.save(f"v27_{system}_gdis_vs_ftle.png")

    def plot_system_components(self, sub: pd.DataFrame) -> None:
        system = sub["system"].iloc[0]
        pname = sub["parameter_name"].iloc[0]
        crit = float(sub["critical_value"].iloc[0])

        cols = ["J_sat", "S_sat", "A_sat", "I_sustained", "transition_energy", "critical_window", "I_transition"]
        plt.figure(figsize=(15, 8))
        for c in cols:
            plt.plot(sub["parameter"], sub[c], label=c)
        plt.axvline(crit, linestyle="--")
        plt.xlabel(pname)
        plt.ylabel("value")
        plt.title(f"{system}: GDIS components")
        plt.legend(fontsize=8, ncol=2)
        self.save(f"v27_{system}_components.png")

    def plot_system_bifurcation(self, bif: pd.DataFrame, scored: pd.DataFrame) -> None:
        system = bif["system"].iloc[0]
        sub = scored[scored["system"] == system]
        pname = sub["parameter_name"].iloc[0]
        crit = float(sub["critical_value"].iloc[0])

        plt.figure(figsize=(14, 6))
        plt.scatter(bif["parameter"], bif["x"], s=0.15)
        plt.axvline(crit, linestyle="--")
        plt.xlabel(pname)
        plt.ylabel("x")
        plt.title(f"{system}: bifurcation diagram")
        self.save(f"v27_{system}_bifurcation.png")

    def plot_validation_metrics(self, metrics_df: pd.DataFrame) -> None:
        df = metrics_df[metrics_df["system"] != "ALL"].copy()
        plt.figure(figsize=(12, 6))
        plt.bar(df["system"], df["balanced_accuracy"])
        plt.ylim(0, 1.05)
        plt.ylabel("balanced accuracy")
        plt.title("GDIS benchmark classification performance")
        self.save("v27_balanced_accuracy_by_system.png")

        plt.figure(figsize=(12, 6))
        plt.bar(df["system"], df["GDIS_FTLE_pearson"])
        plt.ylim(-1.05, 1.05)
        plt.ylabel("Pearson correlation")
        plt.title("GDIS correlation with FTLE by system")
        self.save("v27_gdis_ftle_correlation_by_system.png")

    def plot_scatter_gdis_ftle(self, scored: pd.DataFrame) -> None:
        plt.figure(figsize=(8, 7))
        for system, sub in scored.groupby("system", sort=False):
            plt.scatter(sub["FTLE_scaled"], sub["GDIS"], s=18, label=system)
        plt.xlabel("FTLE scaled")
        plt.ylabel("GDIS")
        plt.title("GDIS vs FTLE across benchmark systems")
        plt.legend()
        self.save("v27_scatter_gdis_vs_ftle_all.png")

    def plot_temporal_examples(self, temporal: pd.DataFrame) -> None:
        for system, sub_sys in temporal.groupby("system", sort=False):
            values = np.sort(sub_sys["parameter"].unique())
            if len(values) == 0:
                continue
            sample_values = np.linspace(values.min(), values.max(), self.cfg.example_count)
            for target in sample_values:
                nearest = values[np.argmin(np.abs(values - target))]
                sub = sub_sys[np.isclose(sub_sys["parameter"], nearest)]
                plt.figure(figsize=(12, 5))
                plt.plot(sub["time"], sub["temporal_signal"])
                plt.xlabel("time / iteration")
                plt.ylabel("temporal instability")
                plt.title(f"{system}: temporal instability, parameter={nearest:.3f}")
                self.save(f"v27_{system}_temporal_{nearest:.3f}.png")

    def make_all_plots(
        self,
        scored: pd.DataFrame,
        bif: pd.DataFrame,
        metrics_df: pd.DataFrame,
        temporal: pd.DataFrame,
    ) -> None:
        self.plot_all_gdis(scored)
        self.plot_validation_metrics(metrics_df)
        self.plot_scatter_gdis_ftle(scored)
        self.plot_temporal_examples(temporal)

        for system, sub in scored.groupby("system", sort=False):
            self.plot_system_gdis(sub)
            self.plot_system_ftle(sub)
            self.plot_system_components(sub)

        for system, sub_bif in bif.groupby("system", sort=False):
            self.plot_system_bifurcation(sub_bif, scored)


# ============================================================
# 10. Report writing
# ============================================================

def write_report(
    cfg: GDISConfig,
    metrics_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    outpath: str,
) -> None:
    lines = []
    lines.append("GDIS Version 27.1 Fast Benchmark Edition Report")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Benchmarks:")
    lines.append("- Lorenz")
    lines.append("- Rössler")
    lines.append("- Chen")
    lines.append("- Logistic map")
    lines.append("")
    lines.append("Frozen mathematical form:")
    lines.append("Phi = -log(1 - I_sustained) + I_transition")
    lines.append("GDIS = 1 - exp(-Phi)")
    lines.append("I_sustained = Hill(core * complexity_factor * temporal_factor)")
    lines.append("core = (J^alpha_J * S^alpha_S * A^alpha_A)^(1 / sum(alpha))")
    lines.append("I_transition = lambda * transition_energy * critical_window")
    lines.append("")
    lines.append("Validation metrics:")
    lines.append(metrics_df.to_string(index=False))
    lines.append("")
    lines.append("Regime summary:")
    lines.append(regime_df.to_string(index=False))
    lines.append("")
    lines.append("Correlations:")
    lines.append(corr_df.to_string(index=False))
    lines.append("")
    lines.append("Parameters:")
    for k, v in asdict(cfg).items():
        lines.append(f"{k}: {v}")

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 11. Main benchmark workflow
# ============================================================

def main() -> None:
    cfg = GDISConfig()
    utils = Utils(cfg)

    figdir = os.path.join(cfg.outdir, "figures")
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    ode_engine = ODEBenchmarkEngine(cfg, utils)
    logistic_engine = LogisticBenchmarkEngine(cfg, utils)
    scorer = GenericGDISScorer(cfg, utils)
    metrics = Metrics(cfg, utils)
    sensitivity = TransitionWeightSensitivity(cfg, utils, metrics)
    plotter = Plotter(cfg, figdir)

    ode_specs = [
        make_lorenz_spec(),
        make_rossler_spec(),
        make_chen_spec(),
    ]
    map_specs = [
        make_logistic_spec(),
    ]

    print("Running GDIS Version 27.2 Stable Fast Benchmark Edition")
    print(f"Output directory: {cfg.outdir}")

    raw_parts = []
    temporal_parts = []
    bif_parts = []

    for spec in ode_specs:
        print(f"\nBenchmarking ODE system: {spec.name}")
        raw_df, temporal_df, bif_df, _examples = ode_engine.extract(spec)
        raw_parts.append(raw_df)
        temporal_parts.append(temporal_df)
        bif_parts.append(bif_df)

    for spec in map_specs:
        print(f"\nBenchmarking map system: {spec.name}")
        raw_df, temporal_df, bif_df = logistic_engine.extract(spec)
        raw_parts.append(raw_df)
        temporal_parts.append(temporal_df)
        bif_parts.append(bif_df)

    raw_all = pd.concat(raw_parts, ignore_index=True)
    temporal_all = pd.concat(temporal_parts, ignore_index=True)
    bif_all = pd.concat(bif_parts, ignore_index=True)

    # Interpolate subset FTLE values per system.
    # This keeps validation practical while preserving a smooth FTLE reference curve.
    raw_all["FTLE_is_interpolated"] = raw_all["FTLE"].isna()
    raw_all["FTLE"] = (
        raw_all.groupby("system", group_keys=False)["FTLE"]
        .apply(lambda s: s.interpolate(limit_direction="both"))
    )

    scored_all = scorer.score_all(raw_all)
    metrics_df, regime_df, corr_df = metrics.validation_tables(scored_all)

    # Transition-weight sensitivity analysis for Table V. This reuses the
    # already-computed sustained and transition-base terms, so no additional
    # trajectory simulations are required.
    table_v_df, table_v_system_df = sensitivity.evaluate(scored_all)
    sensitivity.save_outputs(table_v_df, table_v_system_df, cfg.outdir, figdir)

    # Save per-system and combined results.
    raw_all.to_csv(os.path.join(cfg.outdir, "v27_raw_features_all_systems.csv"), index=False)
    scored_all.to_csv(os.path.join(cfg.outdir, "v27_gdis_results_all_systems.csv"), index=False)
    temporal_all.to_csv(os.path.join(cfg.outdir, "v27_temporal_all_systems.csv"), index=False)
    bif_all.to_csv(os.path.join(cfg.outdir, "v27_bifurcation_points_all_systems.csv"), index=False)
    metrics_df.to_csv(os.path.join(cfg.outdir, "v27_validation_metrics.csv"), index=False)
    regime_df.to_csv(os.path.join(cfg.outdir, "v27_regime_summary.csv"), index=False)
    corr_df.to_csv(os.path.join(cfg.outdir, "v27_correlations.csv"), index=False)

    params_df = pd.DataFrame([{"parameter": k, "value": v} for k, v in asdict(cfg).items()])
    params_df.to_csv(os.path.join(cfg.outdir, "v27_parameters.csv"), index=False)

    for system, sub in scored_all.groupby("system", sort=False):
        sub.to_csv(os.path.join(cfg.outdir, f"v27_{system}_results.csv"), index=False)

    report_path = os.path.join(cfg.outdir, "gdis_v27_benchmark_report.txt")
    write_report(cfg, metrics_df, regime_df, corr_df, report_path)

    # Plots.
    plotter.make_all_plots(scored_all, bif_all, metrics_df, temporal_all)

    print("\nVersion 27.2 stable fast benchmark completed.")
    print(f"Results saved in: {cfg.outdir}")
    print(f"Figures saved in: {figdir}")
    print(f"Report saved to: {report_path}")

    print("\nValidation metrics:")
    print(metrics_df.to_string(index=False))

    print("\nTable V: transition-weight sensitivity:")
    print(table_v_df.to_string(index=False))


if __name__ == "__main__":
    main()

