from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass
class GDISResult:
    """Structured result returned by :class:`gdis.GDIS`."""

    parameters: np.ndarray
    gdis: np.ndarray
    potential: np.ndarray
    sustained_instability: np.ndarray
    transition_instability: np.ndarray
    components: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        data = {
            "parameter": self.parameters,
            "gdis": self.gdis,
            "potential": self.potential,
            "sustained_instability": self.sustained_instability,
            "transition_instability": self.transition_instability,
        }
        data.update(self.components)
        return pd.DataFrame(data)

    def plot(self, **kwargs):
        from .plotting import plot_gdis

        return plot_gdis(self, **kwargs)
