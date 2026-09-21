import numpy as np
import scipy.sparse.linalg as spla

from dqpt_wigner.models import potts3_hamiltonian
from dqpt_wigner.operators import product_state
from dqpt_wigner.wigner_odd import support_map, support_metrics_pure


def test_support_sum_is_branch_probability_and_negative_mass_identity():
    n_sites = 3
    hamiltonian = potts3_hamiltonian(n_sites, h=1.5)
    initial = product_state([0] * n_sites, 3)
    state = spla.expm_multiply((-1.0j * 0.55) * hamiltonian, initial)
    for sector in range(3):
        metrics = support_metrics_pure(state, support_map([sector] * n_sites, 3))
        assert metrics.reconstruction_error < 1e-11
        assert metrics.absolute_weight + 1e-12 >= metrics.probability
        assert np.isclose(
            metrics.absolute_weight - metrics.probability,
            2.0 * metrics.negative_mass,
            atol=1e-11,
        )
        assert metrics.absolute_weight <= 1.0 + 1e-11


def test_initial_stabilizer_branch_has_zero_sign_cost():
    state = product_state([0, 0, 0], 3)
    metrics = support_metrics_pure(state, support_map([0, 0, 0], 3))
    assert np.isclose(metrics.probability, 1.0)
    assert np.isclose(metrics.absolute_weight, 1.0)
    assert np.isclose(metrics.negative_mass, 0.0)
