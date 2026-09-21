"""Sparse Hamiltonians for Potts and Ising chains and a collective Ising model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .operators import basis_digits, basis_powers, spin_matrices


def _nearest_bonds(n_sites: int, periodic: bool) -> list[tuple[int, int]]:
    bonds = [(site, site + 1) for site in range(n_sites - 1)]
    if periodic and n_sites > 2:
        bonds.append((n_sites - 1, 0))
    return bonds


def _hermitian_csr(rows: list[np.ndarray], cols: list[np.ndarray], data: list[np.ndarray], dim: int) -> sp.csr_matrix:
    matrix = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim),
        dtype=complex,
    ).tocsr()
    # This also removes numerical asymmetry and combines duplicate entries.
    matrix = 0.5 * (matrix + matrix.getH())
    matrix.eliminate_zeros()
    return matrix.tocsr()


def potts3_hamiltonian(
    n_sites: int,
    *,
    J: float = 1.0,
    h: float = 1.0,
    longitudinal_bias: float = 0.0,
    periodic: bool = True,
) -> sp.csr_matrix:
    r"""Three-state quantum Potts chain.

    .. math::
       H=-J\sum_{\langle i,j\rangle}(Z_i^\dagger Z_j+Z_i Z_j^\dagger)
         -h\sum_i(X_i+X_i^\dagger)
         -b\sum_i(Z_i+Z_i^\dagger).

    With this convention the self-dual equilibrium point is ``h/J = 1``.
    The Hamiltonian is assembled directly in the computational basis rather
    than through repeated Kronecker products.
    """
    if n_sites < 1:
        raise ValueError("n_sites must be positive")
    d = 3
    dim = d**n_sites
    indices = np.arange(dim, dtype=np.int64)
    digits = basis_digits(n_sites, d)
    powers = basis_powers(n_sites, d)

    diagonal = np.zeros(dim, dtype=float)
    for i, j in _nearest_bonds(n_sites, periodic):
        difference = digits[:, j] - digits[:, i]
        diagonal += -2.0 * J * np.cos(2.0 * np.pi * difference / d)
    if longitudinal_bias != 0.0:
        diagonal += -2.0 * longitudinal_bias * np.sum(
            np.cos(2.0 * np.pi * digits / d), axis=1
        )

    rows: list[np.ndarray] = [indices]
    cols: list[np.ndarray] = [indices]
    data: list[np.ndarray] = [diagonal.astype(complex)]
    for site, place in enumerate(powers):
        old = digits[:, site].astype(np.int64)
        for displacement in (-1, 1):
            new = (old + displacement) % d
            new_index = indices + (new - old) * place
            rows.append(new_index)
            cols.append(indices)
            data.append(np.full(dim, -h, dtype=complex))
    return _hermitian_csr(rows, cols, data, dim)


def ising_hamiltonian(
    n_sites: int,
    *,
    J: float = 1.0,
    h: float = 1.0,
    longitudinal_bias: float = 0.0,
    periodic: bool = True,
) -> sp.csr_matrix:
    r"""Nearest-neighbour transverse-field Ising chain.

    .. math:: H=-J\sum_{\langle i,j\rangle}\sigma_i^z\sigma_j^z
                    -h\sum_i\sigma_i^x-g\sum_i\sigma_i^z.
    """
    if n_sites < 1:
        raise ValueError("n_sites must be positive")
    d = 2
    dim = 2**n_sites
    indices = np.arange(dim, dtype=np.int64)
    digits = basis_digits(n_sites, d)
    powers = basis_powers(n_sites, d)
    z_values = 1.0 - 2.0 * digits

    diagonal = np.zeros(dim, dtype=float)
    for i, j in _nearest_bonds(n_sites, periodic):
        diagonal += -J * z_values[:, i] * z_values[:, j]
    if longitudinal_bias != 0.0:
        diagonal += -longitudinal_bias * np.sum(z_values, axis=1)

    rows: list[np.ndarray] = [indices]
    cols: list[np.ndarray] = [indices]
    data: list[np.ndarray] = [diagonal.astype(complex)]
    for site, place in enumerate(powers):
        new_index = indices + (1 - 2 * digits[:, site].astype(np.int64)) * place
        rows.append(new_index)
        cols.append(indices)
        data.append(np.full(dim, -h, dtype=complex))
    return _hermitian_csr(rows, cols, data, dim)


def kac_normalization(n_sites: int, alpha: float, *, periodic_distance: bool = False) -> float:
    """Return the Kac factor that makes the aligned interaction energy ``-JN/2``.

    For pair weights ``w_ij`` the factor is ``2 sum_{i<j} w_ij / N``.
    """
    if n_sites < 2:
        return 1.0
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    weight_sum = 0.0
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            distance = j - i
            if periodic_distance:
                distance = min(distance, n_sites - distance)
            weight_sum += distance ** (-alpha) if alpha != 0 else 1.0
    return 2.0 * weight_sum / n_sites


def long_range_ising_hamiltonian(
    n_sites: int,
    *,
    J: float = 1.0,
    h: float = 1.0,
    alpha: float = 1.5,
    longitudinal_bias: float = 0.0,
    kac_normalized: bool = True,
    periodic_distance: bool = False,
) -> sp.csr_matrix:
    r"""Power-law Ising chain with optional Kac normalization.

    .. math:: H=-\frac{J}{\mathcal N_\alpha}\sum_{i<j}
       \frac{\sigma_i^z\sigma_j^z}{r_{ij}^{\alpha}}
       -h\sum_i\sigma_i^x-g\sum_i\sigma_i^z.
    """
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    d = 2
    dim = 2**n_sites
    indices = np.arange(dim, dtype=np.int64)
    digits = basis_digits(n_sites, d)
    powers = basis_powers(n_sites, d)
    z_values = 1.0 - 2.0 * digits
    norm = kac_normalization(n_sites, alpha, periodic_distance=periodic_distance) if kac_normalized else 1.0

    diagonal = np.zeros(dim, dtype=float)
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            distance = j - i
            if periodic_distance:
                distance = min(distance, n_sites - distance)
            weight = 1.0 if alpha == 0 else distance ** (-alpha)
            diagonal += -(J / norm) * weight * z_values[:, i] * z_values[:, j]
    if longitudinal_bias != 0.0:
        diagonal += -longitudinal_bias * np.sum(z_values, axis=1)

    rows: list[np.ndarray] = [indices]
    cols: list[np.ndarray] = [indices]
    data: list[np.ndarray] = [diagonal.astype(complex)]
    for site, place in enumerate(powers):
        new_index = indices + (1 - 2 * digits[:, site].astype(np.int64)) * place
        rows.append(new_index)
        cols.append(indices)
        data.append(np.full(dim, -h, dtype=complex))
    return _hermitian_csr(rows, cols, data, dim)


@dataclass(frozen=True)
class CollectiveIsingHamiltonian:
    """Tridiagonal representation of the fully connected Ising model."""

    diagonal: np.ndarray
    off_diagonal: np.ndarray
    m_values: np.ndarray


def collective_ising_tridiagonal(
    n_spins: int,
    *,
    J: float = 1.0,
    h: float = 1.0,
    longitudinal_bias: float = 0.0,
) -> CollectiveIsingHamiltonian:
    r"""Fully connected transverse-field Ising Hamiltonian in the symmetric sector.

    .. math:: H=-\frac{2J}{N}S_z^2-2hS_x-2gS_z.

    An additive constant relative to the all-to-all pair Hamiltonian is omitted.
    For a quench from the fully ordered state at zero initial field, the
    classical separatrix is crossed at ``h_c = J/2``.
    """
    if n_spins < 1:
        raise ValueError("n_spins must be positive")
    spin = n_spins / 2.0
    m_values = np.arange(spin, -spin - 1.0, -1.0, dtype=float)
    diagonal = -(2.0 * J / n_spins) * m_values**2 - 2.0 * longitudinal_bias * m_values
    # Matrix element -2 h <m-1|Sx|m> = -h sqrt(S(S+1)-m(m-1)).
    m_upper = m_values[:-1]
    off_diagonal = -h * np.sqrt(spin * (spin + 1.0) - m_upper * (m_upper - 1.0))
    return CollectiveIsingHamiltonian(diagonal, off_diagonal, m_values)


def collective_ising_dense(
    n_spins: int,
    *,
    J: float = 1.0,
    h: float = 1.0,
    longitudinal_bias: float = 0.0,
) -> np.ndarray:
    tri = collective_ising_tridiagonal(
        n_spins, J=J, h=h, longitudinal_bias=longitudinal_bias
    )
    return np.diag(tri.diagonal) + np.diag(tri.off_diagonal, 1) + np.diag(tri.off_diagonal, -1)


def collective_spin_operators(n_spins: int):
    """Convenience wrapper retained near the collective Hamiltonian."""
    return spin_matrices(n_spins)
