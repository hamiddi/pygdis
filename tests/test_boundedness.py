import numpy as np
from gdis.potential import instability_potential, potential_to_gdis


def test_potential_mapping_is_bounded():
    sustained = np.linspace(0.0, 0.99, 100)
    transition = np.linspace(0.0, 0.3, 100)
    gdis = potential_to_gdis(instability_potential(sustained, transition))
    assert np.all(gdis >= 0.0)
    assert np.all(gdis <= 1.0)
