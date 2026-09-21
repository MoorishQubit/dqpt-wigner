import numpy as np
from dqpt_wigner.potts_exact import potts_hamiltonian, ordered_state, trajectory


def test_potts_hamiltonian_hermitian():
    H=potts_hamiltonian(4,h=1.3,periodic=True)
    assert np.linalg.norm((H-H.getH()).toarray()) < 1e-12


def test_potts_initial_trajectory():
    tr=trajectory(4,1.5,np.linspace(0,0.1,4))
    assert np.isclose(tr.probabilities[0,0],1.0)
    assert np.allclose(tr.probabilities[0,1:],0.0)
    assert np.isclose(tr.initial_rate[0],0.0)
