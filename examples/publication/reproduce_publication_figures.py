#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28_2_run_gdis_publication_validation_with_attractors.py

GDIS Version 28.2 — Publication-Quality Validation with Attractor Gallery
=========================================================================

Purpose
-------
Version 28 does NOT change the GDIS mathematics. It takes the benchmark outputs
from Version 27.2 and generates a publication-ready validation package:

1. Clean benchmark metrics.
2. Per-system classification metrics.
3. GDIS–FTLE correlations.
4. Early-warning analysis around the critical parameter.
5. Publication-quality figures:
   - 2x2 benchmark panel: GDIS vs FTLE
   - 2x2 early-warning panel centered at the critical point
   - GDIS vs FTLE scatter panels with regression lines
   - Benchmark metric heatmap
   - Balanced accuracy bar plot
   - GDIS–FTLE correlation bar plot
   - Per-system component panels
   - Per-system bifurcation diagrams with GDIS overlay
   - Per-system temporal examples
   - A concise summary table figure

Input
-----
By default, this script reads Version 27.2 outputs:

    results/gdis_v27_2_benchmark_stable_fast/
        v27_gdis_results_all_systems.csv
        v27_temporal_all_systems.csv
        v27_bifurcation_points_all_systems.csv

Output
------
    results/gdis_v28_publication/
        v28_validation_metrics.csv
        v28_regime_summary.csv
        v28_correlations.csv
        v28_early_warning_metrics.csv
        gdis_v28_publication_report.txt
        figures/*.png
        figures/*.pdf

Run
---
python 28_2_run_gdis_publication_validation_with_attractors.py

Notes
-----
- This script is intentionally focused on presentation and validation.
- It avoids re-running expensive simulations.
- It assumes Version 27.2 completed successfully.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from scipy.stats import linregress
from scipy.integrate import solve_ivp


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class Config:
    input_dir: str = "results/gdis_v27_2_benchmark_stable_fast"
    output_dir: str = "results/gdis_v28_2_publication_with_attractors"

    results_file: str = "v27_gdis_results_all_systems.csv"
    temporal_file: str = "v27_temporal_all_systems.csv"
    bifurcation_file: str = "v27_bifurcation_points_all_systems.csv"

    dpi: int = 400

    # Journal figure style
    font_family: str = "DejaVu Sans"
    base_font_size: int = 12
    title_font_size: int = 15
    label_font_size: int = 12
    tick_font_size: int = 10
    legend_font_size: int = 10

    # Output formats
    save_png: bool = True
    save_pdf: bool = True

    # Early-warning window around normalized critical point
    early_window_left: float = -0.45
    early_window_right: float = 0.45

    # Lightweight simulations used only for attractor/state-space figures
    attractor_t0: float = 0.0
    attractor_t1: float = 120.0
    attractor_dt: float = 0.02
    attractor_transient_fraction: float = 0.50
    attractor_max_points: int = 5000

    # Representative low / near-transition / high parameter values
    lorenz_attractor_params: tuple = (10.0, 24.5, 40.0)
    rossler_attractor_params: tuple = (2.5, 4.0, 8.0)
    chen_attractor_params: tuple = (10.0, 20.0, 28.0)
    logistic_attractor_params: tuple = (3.2, 3.57, 3.9)


# ============================================================
# 2. Utilities
# ============================================================

class Utils:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def finite_array(x):
        x = np.asarray(x, dtype=float)
        finite = np.isfinite(x)
        if not np.any(finite):
            return np.zeros_like(x)
        fill = np.nanmedian(x[finite])
        return np.nan_to_num(x, nan=fill, posinf=np.nanmax(x[finite]), neginf=np.nanmin(x[finite]))

    @staticmethod
    def safe_corr(x, y, method="pearson"):
        x = Utils.finite_array(x)
        y = Utils.finite_array(y)
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        try:
            if method == "spearman":
                return float(spearmanr(x, y).correlation)
            return float(pearsonr(x, y)[0])
        except Exception:
            return 0.0

    @staticmethod
    def normalize_01(x):
        x = Utils.finite_array(x)
        xmin, xmax = np.min(x), np.max(x)
        if abs(xmax - xmin) < 1e-12:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    @staticmethod
    def auc_roc(y_true, score):
        """
        Simple ROC AUC implementation without sklearn dependency.
        Uses rank-based Mann-Whitney equivalent.
        """
        y_true = np.asarray(y_true).astype(int)
        score = np.asarray(score, dtype=float)

        pos = score[y_true == 1]
        neg = score[y_true == 0]

        if len(pos) == 0 or len(neg) == 0:
            return np.nan

        combined = np.concatenate([pos, neg])
        ranks = pd.Series(combined).rank(method="average").values
        ranks_pos = ranks[:len(pos)]

        auc = (np.sum(ranks_pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        return float(auc)

    @staticmethod
    def best_threshold(y_true, score):
        y_true = np.asarray(y_true).astype(int)
        score = np.asarray(score, dtype=float)

        thresholds = np.linspace(0, 1, 1001)
        best = None

        for th in thresholds:
            y_pred = (score >= th).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            sensitivity = tp / (tp + fn + 1e-12)
            specificity = tn / (tn + fp + 1e-12)
            precision = tp / (tp + fp + 1e-12)
            f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-12)
            accuracy = (tp + tn) / len(y_true)
            balanced_accuracy = 0.5 * (sensitivity + specificity)

            row = {
                "threshold": float(th),
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_accuracy),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "f1": float(f1),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }

            if best is None or row["balanced_accuracy"] > best["balanced_accuracy"]:
                best = row

        return best

    @staticmethod
    def normalized_critical_coordinate(parameter, critical_value):
        """
        Normalize parameter axis so critical point is 0.

        Uses full range for scaling:
            delta = (p - pc) / (pmax - pmin)
        """
        parameter = np.asarray(parameter, dtype=float)
        width = np.max(parameter) - np.min(parameter)
        if width <= 1e-12:
            return np.zeros_like(parameter)
        return (parameter - critical_value) / width


# ============================================================
# 3. Data and metrics
# ============================================================

class DataManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.input_dir = cfg.input_dir
        self.output_dir = cfg.output_dir
        self.fig_dir = os.path.join(cfg.output_dir, "figures")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

    def load(self):
        result_path = os.path.join(self.input_dir, self.cfg.results_file)
        temporal_path = os.path.join(self.input_dir, self.cfg.temporal_file)
        bif_path = os.path.join(self.input_dir, self.cfg.bifurcation_file)

        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Missing results file: {result_path}")

        results = pd.read_csv(result_path)

        if os.path.exists(temporal_path):
            temporal = pd.read_csv(temporal_path)
        else:
            temporal = pd.DataFrame()

        if os.path.exists(bif_path):
            bif = pd.read_csv(bif_path)
        else:
            bif = pd.DataFrame()

        return results, temporal, bif


class MetricComputer:
    def __init__(self, cfg: Config, utils: Utils):
        self.cfg = cfg
        self.utils = utils

    def compute_validation_metrics(self, df):
        rows = []

        for system, sub in df.groupby("system", sort=False):
            y_true = sub["label_unstable"].astype(int).values
            gdis = sub["GDIS"].values
            ftle = sub["FTLE_scaled"].values if "FTLE_scaled" in sub.columns else Utils.normalize_01(sub["FTLE"].values)

            best = Utils.best_threshold(y_true, gdis)
            auc = Utils.auc_roc(y_true, gdis)

            rows.append({
                "system": system,
                "n": len(sub),
                "critical_value": float(sub["critical_value"].iloc[0]),
                "best_threshold": best["threshold"],
                "accuracy": best["accuracy"],
                "balanced_accuracy": best["balanced_accuracy"],
                "sensitivity": best["sensitivity"],
                "specificity": best["specificity"],
                "precision": best["precision"],
                "f1": best["f1"],
                "auc": auc,
                "GDIS_mean": float(np.mean(gdis)),
                "GDIS_std": float(np.std(gdis)),
                "GDIS_FTLE_pearson": Utils.safe_corr(gdis, ftle, "pearson"),
                "GDIS_FTLE_spearman": Utils.safe_corr(gdis, ftle, "spearman"),
                "GDIS_Phi_pearson": Utils.safe_corr(gdis, sub["Phi"].values, "pearson"),
                "GDIS_I_sustained_pearson": Utils.safe_corr(gdis, sub["I_sustained"].values, "pearson"),
                "GDIS_I_transition_pearson": Utils.safe_corr(gdis, sub["I_transition"].values, "pearson"),
            })

        # Overall
        y_true = df["label_unstable"].astype(int).values
        gdis = df["GDIS"].values
        ftle = df["FTLE_scaled"].values if "FTLE_scaled" in df.columns else Utils.normalize_01(df["FTLE"].values)
        best = Utils.best_threshold(y_true, gdis)
        auc = Utils.auc_roc(y_true, gdis)

        rows.append({
            "system": "ALL",
            "n": len(df),
            "critical_value": np.nan,
            "best_threshold": best["threshold"],
            "accuracy": best["accuracy"],
            "balanced_accuracy": best["balanced_accuracy"],
            "sensitivity": best["sensitivity"],
            "specificity": best["specificity"],
            "precision": best["precision"],
            "f1": best["f1"],
            "auc": auc,
            "GDIS_mean": float(np.mean(gdis)),
            "GDIS_std": float(np.std(gdis)),
            "GDIS_FTLE_pearson": Utils.safe_corr(gdis, ftle, "pearson"),
            "GDIS_FTLE_spearman": Utils.safe_corr(gdis, ftle, "spearman"),
            "GDIS_Phi_pearson": Utils.safe_corr(gdis, df["Phi"].values, "pearson"),
            "GDIS_I_sustained_pearson": Utils.safe_corr(gdis, df["I_sustained"].values, "pearson"),
            "GDIS_I_transition_pearson": Utils.safe_corr(gdis, df["I_transition"].values, "pearson"),
        })

        return pd.DataFrame(rows)

    def compute_regime_summary(self, df):
        rows = []

        for system, sub in df.groupby("system", sort=False):
            for label, g in sub.groupby("label_unstable"):
                rows.append({
                    "system": system,
                    "label_unstable": int(label),
                    "n": len(g),
                    "GDIS_mean": float(g["GDIS"].mean()),
                    "GDIS_std": float(g["GDIS"].std()),
                    "FTLE_mean": float(g["FTLE"].mean()),
                    "Phi_mean": float(g["Phi"].mean()),
                    "I_sustained_mean": float(g["I_sustained"].mean()),
                    "I_transition_mean": float(g["I_transition"].mean()),
                })

        return pd.DataFrame(rows)

    def compute_correlations(self, df):
        rows = []

        comparisons = [
            "J_scaled",
            "S_scaled",
            "A_scaled",
            "FTLE_scaled",
            "Phi",
            "I_sustained",
            "I_transition",
            "transition_energy",
        ]

        for system, sub in df.groupby("system", sort=False):
            for comp in comparisons:
                if comp not in sub.columns:
                    continue
                rows.append({
                    "system": system,
                    "comparison": f"GDIS_vs_{comp}",
                    "pearson": Utils.safe_corr(sub["GDIS"].values, sub[comp].values, "pearson"),
                    "spearman": Utils.safe_corr(sub["GDIS"].values, sub[comp].values, "spearman"),
                })

        return pd.DataFrame(rows)

    def compute_early_warning(self, df):
        """
        Quantify when GDIS crosses thresholds relative to the critical point.
        """
        rows = []

        thresholds = [0.25, 0.50, 0.75, 0.80]

        for system, sub in df.groupby("system", sort=False):
            sub = sub.sort_values("parameter").copy()
            p = sub["parameter"].values
            g = sub["GDIS"].values
            pc = float(sub["critical_value"].iloc[0])
            p_range = p.max() - p.min()

            for th in thresholds:
                above = np.where(g >= th)[0]

                if len(above) == 0:
                    cross_p = np.nan
                    delta = np.nan
                else:
                    cross_p = float(p[above[0]])
                    delta = float((cross_p - pc) / (p_range + 1e-12))

                rows.append({
                    "system": system,
                    "threshold": th,
                    "first_crossing_parameter": cross_p,
                    "critical_value": pc,
                    "normalized_lead_time": delta,
                    "crosses_before_critical": bool(delta < 0) if np.isfinite(delta) else False,
                })

        return pd.DataFrame(rows)


# ============================================================
# 4. Publication plotting
# ============================================================

class PublicationPlotter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.fig_dir = os.path.join(cfg.output_dir, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)
        self.setup_style()

    def setup_style(self):
        plt.rcParams.update({
            "font.family": self.cfg.font_family,
            "font.size": self.cfg.base_font_size,
            "axes.titlesize": self.cfg.title_font_size,
            "axes.labelsize": self.cfg.label_font_size,
            "xtick.labelsize": self.cfg.tick_font_size,
            "ytick.labelsize": self.cfg.tick_font_size,
            "legend.fontsize": self.cfg.legend_font_size,
            "axes.linewidth": 1.0,
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
            "lines.linewidth": 2.2,
            "savefig.bbox": "tight",
        })

    def save(self, name):
        if self.cfg.save_png:
            plt.savefig(os.path.join(self.fig_dir, f"{name}.png"), dpi=self.cfg.dpi, bbox_inches="tight", pad_inches=0.20)
        if self.cfg.save_pdf:
            plt.savefig(os.path.join(self.fig_dir, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.20)
        plt.close()

    def ordered_systems(self, df):
        preferred = ["Lorenz", "Rossler", "Chen", "Logistic"]
        present = list(df["system"].unique())
        return [s for s in preferred if s in present] + [s for s in present if s not in preferred]

    def panel_label(self, ax, label):
        ax.text(
            -0.10, 1.04, label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="right",
        )

    def plot_figure1_universal_benchmark(self, df):
        systems = self.ordered_systems(df)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.84, wspace=0.25, hspace=0.42)
        axes = axes.ravel()

        labels = ["A", "B", "C", "D"]

        for ax, system, lab in zip(axes, systems, labels):
            sub = df[df["system"] == system].sort_values("parameter")
            p = sub["parameter"].values
            pc = float(sub["critical_value"].iloc[0])
            pname = str(sub["parameter_name"].iloc[0])

            ax.plot(p, sub["GDIS"], label="GDIS")
            ax.plot(p, sub["FTLE_scaled"], label="FTLE", linestyle="--")
            ax.axvline(pc, linestyle=":", linewidth=1.6)

            ax.set_title(system)
            ax.set_xlabel(pname)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            self.panel_label(ax, lab)

            if ax in axes[::2]:
                ax.set_ylabel("Normalized value")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.91))
        fig.suptitle("Universal GDIS benchmark across nonlinear systems", y=0.98, fontweight="bold")
        self.save("figure1_universal_benchmark_gdis_ftle")

    def plot_figure2_early_warning(self, df):
        systems = self.ordered_systems(df)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.84, wspace=0.25, hspace=0.42)
        axes = axes.ravel()
        labels = ["A", "B", "C", "D"]

        for ax, system, lab in zip(axes, systems, labels):
            sub = df[df["system"] == system].sort_values("parameter").copy()
            pc = float(sub["critical_value"].iloc[0])
            delta = Utils.normalized_critical_coordinate(sub["parameter"].values, pc)
            sub["delta"] = delta

            mask = (delta >= self.cfg.early_window_left) & (delta <= self.cfg.early_window_right)
            s = sub[mask]

            ax.plot(s["delta"], s["GDIS"], label="GDIS")
            ax.plot(s["delta"], s["FTLE_scaled"], label="FTLE", linestyle="--")
            ax.plot(s["delta"], s["I_transition"], label="transition term", linestyle=":")
            ax.axvline(0.0, linestyle="-.", linewidth=1.4)

            ax.set_title(system)
            ax.set_xlabel("normalized distance from critical point")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            self.panel_label(ax, lab)

            if ax in axes[::2]:
                ax.set_ylabel("Normalized value")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.91))
        fig.suptitle("Early-warning behavior near critical transitions", y=0.98, fontweight="bold")
        self.save("figure2_early_warning_centered")

    def plot_figure3_gdis_ftle_scatter(self, df, metrics_df):
        systems = self.ordered_systems(df)
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90, wspace=0.30, hspace=0.38)
        axes = axes.ravel()
        labels = ["A", "B", "C", "D"]

        for ax, system, lab in zip(axes, systems, labels):
            sub = df[df["system"] == system]
            x = sub["FTLE_scaled"].values
            y = sub["GDIS"].values

            ax.scatter(x, y, s=24, alpha=0.78)

            if np.std(x) > 1e-12 and np.std(y) > 1e-12:
                fit = linregress(x, y)
                xx = np.linspace(np.min(x), np.max(x), 100)
                yy = fit.intercept + fit.slope * xx
                ax.plot(xx, yy, linestyle="--", linewidth=1.8)
                r = Utils.safe_corr(x, y, "pearson")
                rho = Utils.safe_corr(x, y, "spearman")
                txt = f"r={r:.2f}\nρ={rho:.2f}"
                ax.text(0.05, 0.88, txt, transform=ax.transAxes, bbox=dict(boxstyle="round", alpha=0.12))

            ax.set_title(system)
            ax.set_xlabel("FTLE scaled")
            ax.set_ylabel("GDIS")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            self.panel_label(ax, lab)

        fig.suptitle("Association between GDIS and finite-time Lyapunov reference", y=0.98, fontweight="bold")
        self.save("figure3_gdis_ftle_scatter_panels")

    def plot_figure4_validation_heatmap(self, metrics_df):
        df = metrics_df[metrics_df["system"] != "ALL"].copy()
        df = df.set_index("system")

        cols = [
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "auc",
            "GDIS_FTLE_pearson",
            "GDIS_FTLE_spearman",
        ]

        data = df[cols].values
        systems = list(df.index)

        fig, ax = plt.subplots(figsize=(13, 5.8))
        fig.subplots_adjust(left=0.13, right=0.92, bottom=0.26, top=0.88)
        im = ax.imshow(data, aspect="auto", vmin=-1, vmax=1)

        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(
            ["Acc.", "Bal. acc.", "Sens.", "Spec.", "AUC", "Pearson", "Spearman"],
            rotation=35,
            ha="right",
        )
        ax.set_yticks(np.arange(len(systems)))
        ax.set_yticklabels(systems)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("metric value")
        ax.set_title("Benchmark validation metrics", fontweight="bold", pad=16)
        self.save("figure4_validation_metric_heatmap")

    def plot_figure5_accuracy_and_correlation(self, metrics_df):
        df = metrics_df[metrics_df["system"] != "ALL"].copy()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.80, wspace=0.32)

        axes[0].bar(df["system"], df["balanced_accuracy"])
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Balanced accuracy")
        axes[0].set_title("Classification performance")
        axes[0].grid(True, axis="y", alpha=0.25)
        self.panel_label(axes[0], "A")

        axes[1].bar(df["system"], df["GDIS_FTLE_pearson"])
        axes[1].set_ylim(-1.05, 1.05)
        axes[1].set_ylabel("Pearson correlation")
        axes[1].set_title("Agreement with FTLE")
        axes[1].grid(True, axis="y", alpha=0.25)
        self.panel_label(axes[1], "B")

        fig.suptitle("Summary of GDIS benchmark performance", y=0.96, fontweight="bold")
        self.save("figure5_accuracy_and_correlation_summary")

    def plot_figure6_component_panels(self, df):
        systems = self.ordered_systems(df)

        for system in systems:
            sub = df[df["system"] == system].sort_values("parameter")
            p = sub["parameter"].values
            pc = float(sub["critical_value"].iloc[0])
            pname = str(sub["parameter_name"].iloc[0])

            fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True)
            fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.91, hspace=0.42)

            axes[0].plot(p, sub["GDIS"], label="GDIS")
            axes[0].plot(p, sub["I_sustained"], label="sustained")
            axes[0].plot(p, sub["I_transition"], label="transition")
            axes[0].axvline(pc, linestyle=":")
            axes[0].set_ylabel("score")
            axes[0].set_title(f"{system}: GDIS decomposition", fontweight="bold", pad=12)
            axes[0].legend(frameon=False)
            axes[0].grid(True, alpha=0.25)
            self.panel_label(axes[0], "A")

            axes[1].plot(p, sub["J_sat"], label="J")
            axes[1].plot(p, sub["S_sat"], label="S")
            axes[1].plot(p, sub["A_sat"], label="A")
            axes[1].axvline(pc, linestyle=":")
            axes[1].set_xlabel(pname)
            axes[1].set_ylabel("saturated channel")
            axes[1].set_title("Core physical channels", fontweight="bold", pad=12)
            axes[1].legend(frameon=False)
            axes[1].grid(True, alpha=0.25)
            self.panel_label(axes[1], "B")

            self.save(f"figure6_{system}_component_panel")

    def plot_figure7_bifurcation_with_gdis(self, df, bif):
        systems = self.ordered_systems(df)

        for system in systems:
            if bif.empty:
                continue

            sub_b = bif[bif["system"] == system]
            sub = df[df["system"] == system].sort_values("parameter")
            if sub_b.empty or sub.empty:
                continue

            pc = float(sub["critical_value"].iloc[0])
            pname = str(sub["parameter_name"].iloc[0])

            fig, ax1 = plt.subplots(figsize=(12.5, 5.6))
            fig.subplots_adjust(left=0.09, right=0.90, bottom=0.14, top=0.88)

            ax1.scatter(sub_b["parameter"], sub_b["x"], s=0.12, alpha=0.65)
            ax1.axvline(pc, linestyle=":")
            ax1.set_xlabel(pname)
            ax1.set_ylabel("state coordinate x")

            ax2 = ax1.twinx()
            ax2.plot(sub["parameter"], sub["GDIS"], linewidth=2.2)
            ax2.set_ylabel("GDIS")

            ax1.set_title(f"{system}: bifurcation structure and GDIS", fontweight="bold", pad=14)
            self.save(f"figure7_{system}_bifurcation_gdis_overlay")

    def plot_figure8_temporal_examples(self, temporal):
        if temporal.empty:
            return

        for system, sub_sys in temporal.groupby("system", sort=False):
            values = np.sort(sub_sys["parameter"].unique())
            sample_values = np.linspace(values.min(), values.max(), 3)

            fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.8), sharex=False)
            fig.subplots_adjust(left=0.11, right=0.98, bottom=0.08, top=0.88, hspace=0.65)

            for ax, target, lab in zip(axes, sample_values, ["A", "B", "C"]):
                nearest = values[np.argmin(np.abs(values - target))]
                sub = sub_sys[np.isclose(sub_sys["parameter"], nearest)]
                ax.plot(sub["time"], sub["temporal_signal"])
                ax.set_ylabel("temporal instability")
                ax.set_title(f"{system}: parameter={nearest:.3f}", fontweight="bold", pad=10)
                ax.grid(True, alpha=0.25)
                self.panel_label(ax, lab)

            axes[-1].set_xlabel("time / iteration")
            fig.suptitle(f"{system}: temporal instability examples", y=0.97, fontweight="bold")
            self.save(f"figure8_{system}_temporal_examples")

    def plot_figure9_summary_table(self, metrics_df):
        df = metrics_df[metrics_df["system"] != "ALL"].copy()

        cols = ["system", "balanced_accuracy", "auc", "GDIS_FTLE_pearson", "GDIS_FTLE_spearman"]
        display = df[cols].copy()
        display.columns = ["System", "Balanced acc.", "AUC", "Pearson", "Spearman"]

        for c in ["Balanced acc.", "AUC", "Pearson", "Spearman"]:
            display[c] = display[c].map(lambda x: f"{x:.3f}")

        fig, ax = plt.subplots(figsize=(11.5, 3.4))
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.08, top=0.78)
        ax.axis("off")

        table = ax.table(
            cellText=display.values,
            colLabels=display.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.08, 1.75)

        ax.set_title("GDIS benchmark validation summary", pad=24, fontweight="bold")
        self.save("figure9_summary_table")


    def plot_figure10_attractor_gallery(self, attractor_data):
        """
        Publication figure: representative attractor/state-space gallery.

        Rows are systems. Columns are low, near-transition, and chaotic/high
        parameter values. Logistic uses the return map x_n versus x_{n+1}.
        """
        systems = ["Lorenz", "Rossler", "Chen", "Logistic"]
        col_titles = ["Low/stable", "Near transition", "High/chaotic"]

        fig = plt.figure(figsize=(15.5, 13.5))
        fig.subplots_adjust(left=0.04, right=0.98, bottom=0.05, top=0.92, wspace=0.20, hspace=0.35)

        panel_letters = list("ABCDEFGHIJKL")
        panel_idx = 0

        for row, system in enumerate(systems):
            entries = attractor_data.get(system, [])

            for col in range(3):
                param, X = entries[col]

                if system == "Logistic":
                    ax = fig.add_subplot(4, 3, row * 3 + col + 1)
                    x0 = X[:-1]
                    x1 = X[1:]
                    ax.scatter(x0, x1, s=0.18, alpha=0.65)
                    ax.set_xlabel(r"$x_n$")
                    ax.set_ylabel(r"$x_{n+1}$")
                    param_label = f"r={param:.2f}"
                else:
                    ax = fig.add_subplot(4, 3, row * 3 + col + 1, projection="3d")
                    ax.plot(X[:, 0], X[:, 1], X[:, 2], linewidth=0.35, alpha=0.9)
                    ax.set_xlabel("x", labelpad=-2)
                    ax.set_ylabel("y", labelpad=-2)
                    ax.set_zlabel("z", labelpad=-2)
                    ax.tick_params(axis="both", labelsize=8, pad=0)

                    if system == "Lorenz":
                        param_label = f"rho={param:.1f}"
                    else:
                        param_label = f"c={param:.1f}"

                if row == 0:
                    ax.set_title(col_titles[col], fontweight="bold", pad=12)

                ax.text2D(
                    0.02, 0.95,
                    f"{panel_letters[panel_idx]}",
                    transform=ax.transAxes,
                    fontsize=14,
                    fontweight="bold",
                    va="top",
                ) if system != "Logistic" else ax.text(
                    0.02, 0.95,
                    f"{panel_letters[panel_idx]}",
                    transform=ax.transAxes,
                    fontsize=14,
                    fontweight="bold",
                    va="top",
                )

                ax.text2D(
                    0.98, 0.06,
                    param_label,
                    transform=ax.transAxes,
                    ha="right",
                    fontsize=10,
                    bbox=dict(boxstyle="round", alpha=0.12),
                ) if system != "Logistic" else ax.text(
                    0.98, 0.06,
                    param_label,
                    transform=ax.transAxes,
                    ha="right",
                    fontsize=10,
                    bbox=dict(boxstyle="round", alpha=0.12),
                )

                if col == 0:
                    ax.text2D(
                        -0.20, 0.5,
                        system,
                        transform=ax.transAxes,
                        rotation=90,
                        fontsize=13,
                        fontweight="bold",
                        va="center",
                        ha="center",
                    ) if system != "Logistic" else ax.text(
                        -0.20, 0.5,
                        system,
                        transform=ax.transAxes,
                        rotation=90,
                        fontsize=13,
                        fontweight="bold",
                        va="center",
                        ha="center",
                    )

                panel_idx += 1

        fig.suptitle("Representative attractor and state-space structures across benchmark systems", y=0.985, fontweight="bold")
        self.save("figure10_attractor_gallery")

    def plot_figure11_attractor_gdis_pairing(self, df, attractor_data):
        """
        Publication figure: pair each representative phase-space structure
        with its corresponding GDIS value.

        This directly links geometric changes in state space with the scalar
        instability score.
        """
        systems = ["Lorenz", "Rossler", "Chen", "Logistic"]

        for system in systems:
            entries = attractor_data.get(system, [])
            sub = df[df["system"] == system].sort_values("parameter")
            if not entries or sub.empty:
                continue

            fig = plt.figure(figsize=(14, 7.5))
            fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.86, wspace=0.30, hspace=0.35)

            # Top row: GDIS curve with markers for selected attractors.
            ax_curve = fig.add_subplot(2, 3, (1, 3))
            ax_curve.plot(sub["parameter"], sub["GDIS"], linewidth=2.4, label="GDIS")
            ax_curve.plot(sub["parameter"], sub["FTLE_scaled"], linestyle="--", linewidth=2.0, label="FTLE")
            ax_curve.axvline(float(sub["critical_value"].iloc[0]), linestyle=":", linewidth=1.8)

            params = [p for p, _ in entries]
            gvals = []
            for p in params:
                idx = (sub["parameter"] - p).abs().idxmin()
                gvals.append(float(sub.loc[idx, "GDIS"]))
            ax_curve.scatter(params, gvals, s=60, zorder=5)

            ax_curve.set_xlabel(str(sub["parameter_name"].iloc[0]))
            ax_curve.set_ylabel("score")
            ax_curve.set_title(f"{system}: representative dynamics and GDIS", fontweight="bold", pad=12)
            ax_curve.legend(frameon=False, ncol=3)
            ax_curve.grid(True, alpha=0.25)

            # Bottom row: attractors/return maps.
            for col, (param, X) in enumerate(entries):
                if system == "Logistic":
                    ax = fig.add_subplot(2, 3, 4 + col)
                    ax.scatter(X[:-1], X[1:], s=0.18, alpha=0.65)
                    ax.set_xlabel(r"$x_n$")
                    ax.set_ylabel(r"$x_{n+1}$")
                    param_label = f"r={param:.2f}"
                else:
                    ax = fig.add_subplot(2, 3, 4 + col, projection="3d")
                    ax.plot(X[:, 0], X[:, 1], X[:, 2], linewidth=0.35)
                    ax.set_xlabel("x", labelpad=-2)
                    ax.set_ylabel("y", labelpad=-2)
                    ax.set_zlabel("z", labelpad=-2)
                    ax.tick_params(axis="both", labelsize=8, pad=0)

                    if system == "Lorenz":
                        param_label = f"rho={param:.1f}"
                    else:
                        param_label = f"c={param:.1f}"

                idx = (sub["parameter"] - param).abs().idxmin()
                gdis_val = float(sub.loc[idx, "GDIS"])
                ax.set_title(f"{param_label}, GDIS={gdis_val:.2f}", fontweight="bold", pad=10)

            self.save(f"figure11_{system}_attractor_gdis_pairing")

    def make_all(self, df, temporal, bif, metrics_df, attractor_data=None):
        self.plot_figure1_universal_benchmark(df)
        self.plot_figure2_early_warning(df)
        self.plot_figure3_gdis_ftle_scatter(df, metrics_df)
        self.plot_figure4_validation_heatmap(metrics_df)
        self.plot_figure5_accuracy_and_correlation(metrics_df)
        self.plot_figure6_component_panels(df)
        self.plot_figure7_bifurcation_with_gdis(df, bif)
        self.plot_figure8_temporal_examples(temporal)
        self.plot_figure9_summary_table(metrics_df)

        if attractor_data is not None:
            self.plot_figure10_attractor_gallery(attractor_data)
            self.plot_figure11_attractor_gdis_pairing(df, attractor_data)



# ============================================================
# 5. Attractor simulation and plotting support
# ============================================================

class AttractorSimulator:
    """
    Lightweight simulator used only for publication attractor figures.

    This class is intentionally separate from the GDIS calculation.
    It does not change the score or validation metrics.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.t_eval = np.arange(cfg.attractor_t0, cfg.attractor_t1, cfg.attractor_dt)

    def _postprocess(self, sol):
        """Remove transient and downsample for plotting."""
        X = sol.y.T
        start = int(len(X) * self.cfg.attractor_transient_fraction)
        X = X[start:]

        if len(X) > self.cfg.attractor_max_points:
            idx = np.linspace(0, len(X) - 1, self.cfg.attractor_max_points).astype(int)
            X = X[idx]

        return X

    def simulate_lorenz(self, rho: float) -> np.ndarray:
        """Lorenz attractor."""
        sigma = 10.0
        beta = 8.0 / 3.0

        def rhs(t, s):
            x, y, z = s
            return [
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z,
            ]

        sol = solve_ivp(
            rhs,
            (self.cfg.attractor_t0, self.cfg.attractor_t1),
            [1.0, 1.0, 1.0],
            t_eval=self.t_eval,
            method="DOP853",
            rtol=1e-8,
            atol=1e-10,
        )
        return self._postprocess(sol)

    def simulate_rossler(self, c: float) -> np.ndarray:
        """Rössler attractor with a=b=0.2."""
        a = 0.2
        b = 0.2

        def rhs(t, s):
            x, y, z = s
            return [
                -y - z,
                x + a * y,
                b + z * (x - c),
            ]

        sol = solve_ivp(
            rhs,
            (self.cfg.attractor_t0, self.cfg.attractor_t1),
            [1.0, 1.0, 1.0],
            t_eval=self.t_eval,
            method="DOP853",
            rtol=1e-8,
            atol=1e-10,
        )
        return self._postprocess(sol)

    def simulate_chen(self, c: float) -> np.ndarray:
        """Chen attractor."""
        a = 35.0
        b = 3.0

        def rhs(t, s):
            x, y, z = s
            return [
                a * (y - x),
                (c - a) * x - x * z + c * y,
                x * y - b * z,
            ]

        sol = solve_ivp(
            rhs,
            (self.cfg.attractor_t0, self.cfg.attractor_t1),
            [0.1, 0.0, 0.0],
            t_eval=self.t_eval,
            method="DOP853",
            rtol=1e-8,
            atol=1e-10,
        )
        return self._postprocess(sol)

    def simulate_logistic(self, r: float, n: int = 6000, transient: int = 2000) -> np.ndarray:
        """Logistic map trajectory."""
        x = np.zeros(n)
        x[0] = 0.1234567

        for i in range(n - 1):
            x[i + 1] = r * x[i] * (1.0 - x[i])
            if not np.isfinite(x[i + 1]):
                x[i + 1] = 0.5

        return x[transient:]

    def generate_all(self) -> dict:
        """Generate all representative trajectories for attractor figures."""
        data = {
            "Lorenz": [(p, self.simulate_lorenz(float(p))) for p in self.cfg.lorenz_attractor_params],
            "Rossler": [(p, self.simulate_rossler(float(p))) for p in self.cfg.rossler_attractor_params],
            "Chen": [(p, self.simulate_chen(float(p))) for p in self.cfg.chen_attractor_params],
            "Logistic": [(p, self.simulate_logistic(float(p))) for p in self.cfg.logistic_attractor_params],
        }
        return data

# ============================================================
# 6. Report writer
# ============================================================

class ReportWriter:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def write(self, metrics, regime, corr, early, outpath):
        lines = []
        lines.append("GDIS Version 28.2 Publication Validation Report")
        lines.append("=" * 55)
        lines.append("")
        lines.append("Purpose")
        lines.append("-------")
        lines.append("Version 28.2 freezes the GDIS equation and produces publication-quality")
        lines.append("validation metrics, figures, and attractor/state-space galleries from the Version 27.2 benchmark outputs.")
        lines.append("")
        lines.append("Core equation")
        lines.append("-------------")
        lines.append("Phi = -log(1 - I_sustained) + I_transition")
        lines.append("GDIS = 1 - exp(-Phi)")
        lines.append("")
        lines.append("Validation metrics")
        lines.append("------------------")
        lines.append(metrics.to_string(index=False))
        lines.append("")
        lines.append("Regime summary")
        lines.append("--------------")
        lines.append(regime.to_string(index=False))
        lines.append("")
        lines.append("Correlations")
        lines.append("------------")
        lines.append(corr.to_string(index=False))
        lines.append("")
        lines.append("Early-warning metrics")
        lines.append("---------------------")
        lines.append(early.to_string(index=False))
        lines.append("")

        with open(outpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ============================================================
# 6. Main
# ============================================================

def main():
    cfg = Config()
    utils = Utils(cfg)
    data_manager = DataManager(cfg)
    metrics = MetricComputer(cfg, utils)
    plotter = PublicationPlotter(cfg)
    reporter = ReportWriter(cfg)
    attractor_simulator = AttractorSimulator(cfg)

    print("Running GDIS Version 28.2 Publication Validation with Attractor Gallery")
    print(f"Input directory: {cfg.input_dir}")
    print(f"Output directory: {cfg.output_dir}")

    df, temporal, bif = data_manager.load()

    # Ensure expected columns.
    if "FTLE_scaled" not in df.columns:
        df["FTLE_scaled"] = df.groupby("system", group_keys=False)["FTLE"].transform(Utils.normalize_01)

    validation_df = metrics.compute_validation_metrics(df)
    regime_df = metrics.compute_regime_summary(df)
    corr_df = metrics.compute_correlations(df)
    early_df = metrics.compute_early_warning(df)

    # Save tables.
    validation_df.to_csv(os.path.join(cfg.output_dir, "v28_validation_metrics.csv"), index=False)
    regime_df.to_csv(os.path.join(cfg.output_dir, "v28_regime_summary.csv"), index=False)
    corr_df.to_csv(os.path.join(cfg.output_dir, "v28_correlations.csv"), index=False)
    early_df.to_csv(os.path.join(cfg.output_dir, "v28_early_warning_metrics.csv"), index=False)

    # Also save an enriched copy of the benchmark results.
    df.to_csv(os.path.join(cfg.output_dir, "v28_benchmark_results_used.csv"), index=False)

    # Generate representative attractor/state-space trajectories for publication figures.
    # This does not affect GDIS metrics; it is only used for visual validation.
    print("Generating representative attractor/state-space plots...")
    attractor_data = attractor_simulator.generate_all()

    # Generate publication figures.
    plotter.make_all(df, temporal, bif, validation_df, attractor_data=attractor_data)

    report_path = os.path.join(cfg.output_dir, "gdis_v28_publication_report.txt")
    reporter.write(validation_df, regime_df, corr_df, early_df, report_path)

    print("\nVersion 28.2 completed.")
    print(f"Tables saved in: {cfg.output_dir}")
    print(f"Figures saved in: {os.path.join(cfg.output_dir, 'figures')}")
    print(f"Report saved to: {report_path}")

    print("\nValidation metrics:")
    print(validation_df.to_string(index=False))


if __name__ == "__main__":
    main()

