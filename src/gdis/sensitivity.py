"""Sensitivity analyses that reuse a completed GDIS computation."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .potential import instability_potential, potential_to_gdis
from .result import GDISResult
from .scaling import EPS, smooth_series


def transition_weight_sensitivity(
    result: GDISResult,
    weights: Iterable[float] = (0.0, 0.18, 0.25, 0.50, 0.75, 1.0),
) -> pd.DataFrame:
    """Recompute GDIS for candidate transition weights without rerunning descriptors.

    The returned table contains one row per parameter and candidate weight. It
    intentionally does not optimize classification thresholds; validation
    metrics belong in :mod:`gdis.validation` or paper-reproduction scripts.
    """
    if "transition_base" not in result.components:
        raise ValueError("The result does not contain transition_base; recompute it with pyGDIS >= 1.0.0.")
    base = np.asarray(result.components["transition_base"], dtype=float)
    rows = []
    for weight in weights:
        weight = float(weight)
        if weight < 0:
            raise ValueError("Transition weights must be nonnegative.")
        phi = instability_potential(
            result.sustained_instability,
            transition_base=base,
            transition_weight=weight,
        )
        gdis = np.clip(smooth_series(potential_to_gdis(phi)), 0.0, 1.0 - EPS)
        for parameter, score, potential in zip(result.parameters, gdis, phi):
            rows.append(
                {
                    "transition_weight": weight,
                    "parameter": float(parameter),
                    "gdis": float(score),
                    "potential": float(potential),
                }
            )
    return pd.DataFrame(rows)
