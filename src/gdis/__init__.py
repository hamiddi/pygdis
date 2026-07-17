"""pyGDIS: Generalized Dynamical Instability Score."""

from .result import GDISResult
from .score import GDIS, GDISConfig
from .sensitivity import transition_weight_sensitivity

__all__ = ["GDIS", "GDISConfig", "GDISResult", "transition_weight_sensitivity"]
__version__ = "1.0.0"
