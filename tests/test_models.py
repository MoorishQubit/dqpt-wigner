import numpy as np

from dqpt_wigner.models import (
    collective_ising_dense,
    ising_hamiltonian,
    kac_normalization,
    long_range_ising_hamiltonian,
    potts3_hamiltonian,
)


def _assert_hermitian(matrix):
    dense_difference = (matrix - matrix.getH()).toarray() if hasattr(matrix, "getH") else matrix - matrix.conj().T
    assert np.max(np.abs(dense_difference)) < 1e-12


def test_chain_hamiltonians_are_hermitian():
    _assert_hermitian(potts3_hamiltonian(3, h=1.2))
    _assert_hermitian(ising_hamiltonian(4, h=0.8))
    _assert_hermitian(long_range_ising_hamiltonian(4, h=0.8, alpha=1.2))


def test_kac_normalization_aligned_energy_is_extensive():
    n_sites = 6
    alpha = 1.3
    norm = kac_normalization(n_sites, alpha)
    pair_sum = sum((j - i) ** (-alpha) for i in range(n_sites) for j in range(i + 1, n_sites))
    aligned_energy = -pair_sum / norm
    assert np.isclose(aligned_energy, -n_sites / 2.0)


def test_collective_hamiltonian_is_hermitian():
    matrix = collective_ising_dense(10, J=1.0, h=0.6)
    assert np.max(np.abs(matrix - matrix.conj().T)) < 1e-12
