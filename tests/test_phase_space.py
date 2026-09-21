import numpy as np

from dqpt_wigner.operators import product_state
from dqpt_wigner.wigner_odd import (
    full_wigner_pure,
    local_phase_point_operators,
    validate_local_phase_space,
    wigner_overlap,
)
from dqpt_wigner.wigner_qubit import validate_frame


def test_qutrit_phase_point_basis():
    result = validate_local_phase_space(3)
    assert result["passed"]


def test_qubit_frames_are_valid_operator_bases():
    assert validate_frame("standard")["passed"]
    assert validate_frame("reflected_y")["passed"]


def test_full_qutrit_wigner_normalization_purity_and_overlap():
    psi0 = product_state([0, 0], 3)
    psi1 = product_state([1, 1], 3)
    w0 = full_wigner_pure(psi0, 3, 2, progress=False)
    w1 = full_wigner_pure(psi1, 3, 2, progress=False)
    assert np.isclose(w0.sum(), 1.0, atol=1e-12)
    assert np.isclose(9.0 * np.sum(w0**2), 1.0, atol=1e-12)
    assert np.isclose(wigner_overlap(w0, w0, 9), 1.0, atol=1e-12)
    assert np.isclose(wigner_overlap(w0, w1, 9), 0.0, atol=1e-12)
