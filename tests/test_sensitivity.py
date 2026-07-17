import numpy as np

from gdis import GDISResult, transition_weight_sensitivity


def test_transition_weight_sensitivity_reuses_components():
    sustained = np.array([0.1, 0.2, 0.3])
    base = np.array([0.0, 0.5, 0.0])
    result = GDISResult(
        parameters=np.array([0.0, 1.0, 2.0]),
        gdis=sustained.copy(),
        potential=-np.log(1 - sustained),
        sustained_instability=sustained,
        transition_instability=np.zeros(3),
        components={"transition_base": base},
    )
    table = transition_weight_sensitivity(result, weights=(0.0, 1.0))
    low = table[table.transition_weight == 0.0].gdis.to_numpy()
    high = table[table.transition_weight == 1.0].gdis.to_numpy()
    assert np.allclose(low, sustained, atol=1e-10)
    assert high[1] > low[1]
