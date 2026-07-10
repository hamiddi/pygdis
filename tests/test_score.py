import numpy as np
from gdis import GDIS


def make_trajectory(scale):
    t = np.linspace(0, 20, 1000)
    return np.column_stack([scale * np.sin(t), scale * np.cos(t), 0.5 * scale * np.sin(2 * t)])


def test_gdis_output_shape_and_bounds():
    trajectories = [make_trajectory(s) for s in [0.1, 0.5, 1.0, 2.0, 3.0]]
    result = GDIS().fit_transform(trajectories, np.arange(5, dtype=float))
    assert len(result.gdis) == 5
    assert np.all(result.gdis >= 0.0)
    assert np.all(result.gdis <= 1.0)
