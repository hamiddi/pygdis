from gdis.systems import LogisticMap, LorenzSystem


def test_lorenz_simulation():
    trajectory = LorenzSystem().simulate(28.0)
    assert trajectory.ndim == 2 and trajectory.shape[1] == 3


def test_logistic_simulation():
    trajectory = LogisticMap(parameter_values=[3.2, 3.9], iterations=1000, transient=200).simulate(3.9)
    assert trajectory.ndim == 2 and trajectory.shape[1] == 1
