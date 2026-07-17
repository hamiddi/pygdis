"""Convenience access to canonical benchmark systems.

Benchmark systems are kept outside the core score implementation so users can
apply :class:`gdis.GDIS` to arbitrary trajectories without importing paper-
specific validation code.
"""

from ..systems import ChenSystem, LogisticMap, LorenzSystem, RosslerSystem

__all__ = ["LorenzSystem", "RosslerSystem", "ChenSystem", "LogisticMap"]
