import numpy as np
from gdis.components import attractor_expansion, entropy_complexity, stretching_rate


def test_constant_trajectory_is_safe():
    trajectory = np.ones((200, 3))
    assert attractor_expansion(trajectory) == 0.0
    assert np.isfinite(stretching_rate(trajectory))
    assert np.isfinite(entropy_complexity(trajectory))
