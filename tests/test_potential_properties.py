import numpy as np

from gdis.potential import instability_potential, potential_to_gdis


def test_bounded_and_monotonic_mapping():
    sustained = np.array([0.0, 0.2, 0.5, 0.9])
    base = np.array([0.0, 0.1, 0.2, 0.3])
    phi = instability_potential(sustained, transition_base=base, transition_weight=0.18)
    score = potential_to_gdis(phi)
    assert np.all(score >= 0.0)
    assert np.all(score < 1.0)
    assert np.all(np.diff(score) > 0.0)


def test_baseline_preservation_without_transition():
    sustained = np.linspace(0.0, 0.95, 20)
    score = potential_to_gdis(instability_potential(sustained))
    assert np.allclose(score, sustained, atol=2e-12)
