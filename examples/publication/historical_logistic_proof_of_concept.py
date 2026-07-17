#!/usr/bin/env python3
"""
01_run_logistic_gdis.py

First proof-of-concept implementation of GDIS using the logistic map.

System:
    x_{n+1} = r x_n (1 - x_n)

Outputs:
    results/logistic/logistic_gdis_results.csv
    results/logistic/logistic_bifurcation_points.csv
    results/logistic/figures/bifurcation_diagram.png
    results/logistic/figures/lle_vs_r.png
    results/logistic/figures/gdis_vs_r.png
    results/logistic/figures/components_vs_r.png
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "results/logistic"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

R_MIN = 2.5
R_MAX = 4.0
R_STEP = 0.001

N_TOTAL = 10000
N_TRANSIENT = 2000
X0 = 0.2

EPS = 1e-12


# ============================================================
# Logistic map simulation
# ============================================================

def simulate_logistic(r, x0=X0, n_total=N_TOTAL):
    x = np.zeros(n_total)
    x[0] = x0

    for n in range(n_total - 1):
        x[n + 1] = r * x[n] * (1.0 - x[n])

        # Safety check: logistic map should remain in [0,1] for r <= 4
        if not np.isfinite(x[n + 1]):
            x[n + 1:] = np.nan
            break

    return x

def logistic_lle(r, x):
    """
    Compute the largest Lyapunov exponent for the logistic map.

    For the logistic map:

        f'(x) = r(1 - 2x)

    The Lyapunov exponent is:

        lambda = mean(log |f'(x_n)|)

    Parameters
    ----------
    r : float
        Logistic map control parameter.
    x : numpy.ndarray
        Trajectory after transient removal.

    Returns
    -------
    lle : float
        Largest Lyapunov exponent estimate.
    """
    deriv = np.abs(r * (1.0 - 2.0 * x))
    deriv = np.maximum(deriv, EPS)
    lle = np.mean(np.log(deriv))
    return lle


# ============================================================
# GDIS component estimators
# ============================================================

def permutation_entropy(x, order=3):
    """
    Compute normalized permutation entropy.

    This measures temporal complexity based on ordinal patterns.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional time series.
    order : int
        Embedding order for ordinal patterns.

    Returns
    -------
    pe_norm : float
        Normalized permutation entropy in [0, 1].
    """
    patterns = {}

    for i in range(len(x) - order + 1):
        pattern = tuple(np.argsort(x[i:i + order]))
        patterns[pattern] = patterns.get(pattern, 0) + 1

    counts = np.array(list(patterns.values()), dtype=float)
    probs = counts / (counts.sum() + EPS)

    pe = -np.sum(probs * np.log(probs + EPS))

    # FIXED:
    # Older code used np.math.factorial(order), which fails in NumPy 2.x.
    # The standard Python math module should be used instead.
    pe_max = np.log(math.factorial(order))

    pe_norm = pe / (pe_max + EPS)
    return pe_norm


def recurrence_metrics(x, threshold_quantile=0.10):
    """
    Compute simple recurrence metrics for a one-dimensional trajectory.

    This is a proof-of-concept recurrence implementation.
    For publication, this should later be replaced with a more complete
    RQA implementation.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional time series.
    threshold_quantile : float
        Quantile used to define recurrence threshold.

    Returns
    -------
    rr : float
        Recurrence rate.
    det : float
        Determinism approximation.
    lam : float
        Laminarity approximation.
    entr : float
        Recurrence entropy approximation.
    """
    x = np.asarray(x).reshape(-1, 1)

    # Pairwise distance matrix
    dist = np.abs(x - x.T)

    # Recurrence threshold
    eps_r = np.quantile(dist, threshold_quantile)

    # Recurrence matrix
    R = (dist <= eps_r).astype(int)

    rr = R.mean()
    n = R.shape[0]

    # ----------------------------
    # Diagonal line statistics
    # ----------------------------
    diag_lengths = []

    for k in range(-n + 1, n):
        diag = np.diagonal(R, offset=k)
        count = 0

        for val in diag:
            if val == 1:
                count += 1
            else:
                if count >= 2:
                    diag_lengths.append(count)
                count = 0

        if count >= 2:
            diag_lengths.append(count)

    if len(diag_lengths) == 0:
        det = 0.0
        entr = 0.0
    else:
        diag_lengths = np.array(diag_lengths)
        det = diag_lengths.sum() / (R.sum() + EPS)

        vals, counts = np.unique(diag_lengths, return_counts=True)
        probs = counts / (counts.sum() + EPS)

        entr = -np.sum(probs * np.log(probs + EPS))
        entr = entr / (np.log(len(probs) + EPS) + EPS)

    # ----------------------------
    # Vertical line statistics
    # ----------------------------
    vert_lengths = []

    for j in range(n):
        col = R[:, j]
        count = 0

        for val in col:
            if val == 1:
                count += 1
            else:
                if count >= 2:
                    vert_lengths.append(count)
                count = 0

        if count >= 2:
            vert_lengths.append(count)

    if len(vert_lengths) == 0:
        lam = 0.0
    else:
        vert_lengths = np.array(vert_lengths)
        lam = vert_lengths.sum() / (R.sum() + EPS)

    return rr, det, lam, entr


def predictability_collapse_index(x, horizon=10):
    """
    Compute a simple Predictability Collapse Index proxy.

    This proof-of-concept version compares x(t+h) with x(t).
    Later, this should be replaced by a proper local prediction model.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional trajectory.
    horizon : int
        Maximum forecast horizon.

    Returns
    -------
    pci : float
        Forecast-error growth proxy.
    """
    errors = []

    for h in range(1, horizon + 1):
        pred = x[:-h]
        true = x[h:]
        err = np.mean(np.abs(true - pred))
        errors.append(err)

    e1 = errors[0]
    eH = errors[-1]

    pci = (eH - e1) / (horizon - 1 + EPS)
    pci = max(0.0, pci)

    return pci


def geometric_convergence_score(x, n_blocks=10):
    """
    Compute a simple Attractor Convergence Stability proxy.

    The trajectory is split into blocks. Each block is represented by
    an occupancy histogram. The score is the average change between
    consecutive histograms.

    Larger values indicate weaker geometric convergence.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional trajectory.
    n_blocks : int
        Number of trajectory blocks.

    Returns
    -------
    geo_score : float
        Geometric convergence instability proxy.
    """
    histograms = []
    bins = np.linspace(0.0, 1.0, 51)
    block_size = len(x) // n_blocks

    for k in range(n_blocks):
        block = x[k * block_size:(k + 1) * block_size]

        hist, _ = np.histogram(block, bins=bins, density=False)
        hist = hist.astype(float)
        hist = hist / (hist.sum() + EPS)

        histograms.append(hist)

    distances = []

    for k in range(len(histograms) - 1):
        d = np.mean(np.abs(histograms[k + 1] - histograms[k]))
        distances.append(d)

    geo_score = np.mean(distances)
    return geo_score


def resilience_proxy(x):
    """
    Placeholder resilience score for the logistic map.

    In the full GDIS framework, this should be replaced by Dynamic
    Recovery Instability, where the system is perturbed and recovery
    time is measured.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional trajectory.

    Returns
    -------
    score : float
        Simple variance-based proxy.
    """
    return np.var(x)


# ============================================================
# Normalization
# ============================================================

def minmax_normalize(values):
    """
    Normalize values to [0, 1].

    Parameters
    ----------
    values : array-like
        Raw component values.

    Returns
    -------
    normalized : numpy.ndarray
        Min-max normalized values.
    """
    values = np.asarray(values, dtype=float)
    return (values - values.min()) / (values.max() - values.min() + EPS)


# ============================================================
# Main experiment
# ============================================================

def main():
    #r_values = np.arange(R_MIN, R_MAX + R_STEP, R_STEP)
    r_values = np.arange(R_MIN, R_MAX + 0.5 * R_STEP, R_STEP)
    r_values = r_values[r_values <= R_MAX]

    rows = []
    bifurcation_rows = []

    print("Running logistic map GDIS proof of concept...")

    for idx, r in enumerate(r_values):
        # Simulate full trajectory
        x_full = simulate_logistic(r)

        # Remove transient
        x = x_full[N_TRANSIENT:]
        if np.any(~np.isfinite(x)):
            print(f"Skipping invalid trajectory at r={r}")
            continue

        # Compute divergence component raw input
        lle = logistic_lle(r, x)

        # Compute entropy component raw input
        pe = permutation_entropy(x, order=3)

        # Compute recurrence metrics.
        # Downsampling is used to reduce memory/time cost of the recurrence matrix.
        rr, det, lam, rentr = recurrence_metrics(x[::10])

        # Compute predictability proxy
        pci = predictability_collapse_index(x, horizon=10)

        # Compute geometry proxy
        geo = geometric_convergence_score(x)

        # Compute resilience proxy
        resil = resilience_proxy(x)

        # Recurrence instability combines disrupted recurrence structure.
        recurrence_instability = (
            (1.0 - det)
            + (1.0 - lam)
            + rentr
            + (1.0 - rr)
        ) / 4.0

        rows.append({
            "r": r,
            "LLE_raw": lle,
            "PE_raw": pe,
            "RR_raw": rr,
            "DET_raw": det,
            "LAM_raw": lam,
            "RENTR_raw": rentr,
            "PCI_raw": pci,
            "GEOMETRY_raw": geo,
            "RESILIENCE_raw": resil,
            "RECURRENCE_raw": recurrence_instability,
        })

        # Store last 200 values for bifurcation diagram
        for val in x[-200:]:
            bifurcation_rows.append({
                "r": r,
                "x": val
            })

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(r_values)} r-values")

    # Convert raw results to DataFrame
    df = pd.DataFrame(rows)

    # Normalize all six GDIS components
    df["I_D"] = minmax_normalize(np.maximum(0.0, df["LLE_raw"].values))
    df["I_P"] = minmax_normalize(df["PCI_raw"].values)
    df["I_R"] = minmax_normalize(df["RECURRENCE_raw"].values)
    df["I_E"] = minmax_normalize(df["PE_raw"].values)
    df["I_G"] = minmax_normalize(df["GEOMETRY_raw"].values)
    df["I_U"] = minmax_normalize(df["RESILIENCE_raw"].values)

    component_cols = ["I_D", "I_P", "I_R", "I_E", "I_G", "I_U"]

    # Baseline GDIS: equal-weight average
    df["GDIS_equal"] = df[component_cols].mean(axis=1)

    # Save numerical results
    csv_path = os.path.join(OUTPUT_DIR, "logistic_gdis_results.csv")
    df.to_csv(csv_path, index=False)

    # Save bifurcation data
    bif_df = pd.DataFrame(bifurcation_rows)
    bif_csv_path = os.path.join(OUTPUT_DIR, "logistic_bifurcation_points.csv")
    bif_df.to_csv(bif_csv_path, index=False)

    print(f"Saved results to {csv_path}")
    print(f"Saved bifurcation points to {bif_csv_path}")

    # ========================================================
    # Generate plots
    # ========================================================

    plt.figure(figsize=(10, 6))
    plt.scatter(bif_df["r"], bif_df["x"], s=0.05)
    plt.xlabel("r")
    plt.ylabel("x")
    plt.title("Logistic Map Bifurcation Diagram")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "bifurcation_diagram.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df["r"], df["LLE_raw"])
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("r")
    plt.ylabel("Largest Lyapunov Exponent")
    plt.title("LLE vs r")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "lle_vs_r.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df["r"], df["GDIS_equal"])
    plt.xlabel("r")
    plt.ylabel("GDIS")
    plt.title("Preliminary Equal-Weight GDIS vs r")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "gdis_vs_r.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for col in component_cols:
        plt.plot(df["r"], df[col], label=col)

    plt.xlabel("r")
    plt.ylabel("Normalized component value")
    plt.title("GDIS Components vs r")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "components_vs_r.png"), dpi=300)
    plt.close()

    print("Finished.")
    print(f"Figures saved in {FIG_DIR}")


if __name__ == "__main__":
    main()
