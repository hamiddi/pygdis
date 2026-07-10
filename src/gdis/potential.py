from __future__ import annotations

import numpy as np

from .scaling import EPS, bounded_from_potential


def instability_potential(sustained_instability, transition_instability):
    sustained = np.clip(np.asarray(sustained_instability, dtype=float), 0.0, 1.0 - EPS)
    transition = np.maximum(np.asarray(transition_instability, dtype=float), 0.0)
    return -np.log(1.0 - sustained + EPS) + transition


def potential_to_gdis(potential):
    return bounded_from_potential(potential)
