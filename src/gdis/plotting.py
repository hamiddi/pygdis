from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(figure, output_path, dpi):
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)


def plot_gdis(result, output_path=None, dpi=300):
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(result.parameters, result.gdis, label="GDIS", linewidth=2.4)
    axis.plot(result.parameters, result.sustained_instability, label="Sustained instability", linewidth=2.0)
    axis.plot(result.parameters, result.transition_instability, label="Transition instability", linewidth=2.0, linestyle="--")
    axis.set_xlabel("Control parameter")
    axis.set_ylabel("Score")
    axis.set_ylim(-0.03, 1.03)
    axis.set_title("Generalized Dynamical Instability Score", fontweight="bold")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save(figure, output_path, dpi)
    return figure, axis


def plot_gdis_vs_reference(result, reference, reference_label="Reference", output_path=None, dpi=300):
    reference = np.asarray(reference, dtype=float)
    if len(reference) != len(result.gdis):
        raise ValueError("Reference length must equal the number of GDIS values.")
    rmin, rmax = np.min(reference), np.max(reference)
    scaled = np.zeros_like(reference) if abs(rmax - rmin) < 1e-12 else (reference - rmin) / (rmax - rmin)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(result.parameters, result.gdis, label="GDIS", linewidth=2.4)
    axis.plot(result.parameters, scaled, label=reference_label, linewidth=2.0, linestyle="--")
    axis.set_xlabel("Control parameter")
    axis.set_ylabel("Normalized value")
    axis.set_ylim(-0.03, 1.03)
    axis.set_title(f"GDIS versus {reference_label}", fontweight="bold")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save(figure, output_path, dpi)
    return figure, axis


def plot_components(result, output_path=None, dpi=300):
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    figure.subplots_adjust(hspace=0.35)
    axes[0].plot(result.parameters, result.components["jacobian_saturated"], label="Jacobian")
    axes[0].plot(result.parameters, result.components["stretching_saturated"], label="Stretching")
    axes[0].plot(result.parameters, result.components["expansion_saturated"], label="Expansion")
    axes[0].set_ylabel("Saturated channel")
    axes[0].set_title("Core physical channels", fontweight="bold")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(result.parameters, result.sustained_instability, label="Sustained")
    axes[1].plot(result.parameters, result.transition_instability, label="Transition", linestyle="--")
    axes[1].plot(result.parameters, result.gdis, label="GDIS", linewidth=2.4)
    axes[1].set_xlabel("Control parameter")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Instability decomposition", fontweight="bold")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    _save(figure, output_path, dpi)
    return figure, axes
