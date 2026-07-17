"""Instability-potential construction and bounded GDIS mapping."""
from __future__ import annotations

import numpy as np

from .scaling import EPS, bounded_from_potential


def instability_potential(
    sustained_instability,
    transition_base=None,
    transition_weight: float = 1.0,
    *,
    transition_instability=None,
):
    """Construct the generalized instability potential.

    Parameters
    ----------
    sustained_instability:
        Bounded sustained-instability values in ``[0, 1)``.
    transition_base:
        Nonnegative unweighted transition-localization values. This is the
        preferred v1.0 interface.
    transition_weight:
        Nonnegative coefficient :math:`\\lambda_t` multiplying
        ``transition_base``.
    transition_instability:
        Backward-compatible argument containing an already weighted transition
        term. Do not provide this together with ``transition_base``.
    """
    sustained = np.clip(np.asarray(sustained_instability, dtype=float), 0.0, 1.0 - EPS)
    if transition_instability is not None and transition_base is not None:
        raise ValueError("Provide transition_base or transition_instability, not both.")
    if transition_weight < 0:
        raise ValueError("transition_weight must be nonnegative.")
    if transition_instability is not None:
        transition = np.maximum(np.asarray(transition_instability, dtype=float), 0.0)
    elif transition_base is None:
        transition = np.zeros_like(sustained)
    else:
        transition = transition_weight * np.maximum(np.asarray(transition_base, dtype=float), 0.0)
    return -np.log(1.0 - sustained + EPS) + transition


def potential_to_gdis(potential):
    """Map a nonnegative instability potential to ``0 <= GDIS < 1``."""
    return bounded_from_potential(potential)
