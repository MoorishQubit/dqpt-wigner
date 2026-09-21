import numpy as np

from dqpt_wigner.information_scale import (
    ghz_proper_marginal_distance,
    ghz_return_probability,
)


def test_identical_proper_marginals_can_have_orthogonal_global_returns():
    assert ghz_proper_marginal_distance(5, 0.0, np.pi, keep_sites=3) < 1e-12
    assert np.isclose(ghz_return_probability(5, 0.0), 1.0)
    assert ghz_return_probability(5, np.pi) < 1e-12
