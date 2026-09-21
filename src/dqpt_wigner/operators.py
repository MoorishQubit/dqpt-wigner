"""Local operators, basis utilities, and collective-spin matrices."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp


def pauli_matrices(*, sparse: bool = False):
    """Return ``(I, X, Y, Z)`` in the computational basis."""
    eye = np.eye(2, dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)
    if sparse:
        return tuple(sp.csr_matrix(op) for op in (eye, x, y, z))
    return eye, x, y, z


def clock_shift(d: int = 3, *, sparse: bool = False):
    """Return ``(I, X, Z)`` for a ``d``-state clock degree of freedom.

    The convention is ``X|q> = |q+1 mod d>`` and
    ``Z|q> = exp(2 pi i q/d)|q>``.
    """
    if d < 2:
        raise ValueError("d must be at least two")
    omega = np.exp(2.0j * np.pi / d)
    eye = np.eye(d, dtype=complex)
    x = np.zeros((d, d), dtype=complex)
    for q in range(d):
        x[(q + 1) % d, q] = 1.0
    z = np.diag(omega ** np.arange(d)).astype(complex)
    if sparse:
        return tuple(sp.csr_matrix(op) for op in (eye, x, z))
    return eye, x, z


@lru_cache(maxsize=None)
def basis_powers(n_sites: int, d: int) -> np.ndarray:
    """Place values for lexicographically ordered base-``d`` basis states."""
    if n_sites < 1 or d < 2:
        raise ValueError("n_sites >= 1 and d >= 2 are required")
    out = d ** np.arange(n_sites - 1, -1, -1, dtype=np.int64)
    out.setflags(write=False)
    return out


@lru_cache(maxsize=None)
def basis_digits(n_sites: int, d: int) -> np.ndarray:
    """Return an array of shape ``(d**n_sites, n_sites)`` of basis digits."""
    indices = np.arange(d**n_sites, dtype=np.int64)
    powers = basis_powers(n_sites, d)
    out = ((indices[:, None] // powers[None, :]) % d).astype(np.int16)
    out.setflags(write=False)
    return out


def digits_to_index(digits: Sequence[int] | np.ndarray, d: int) -> int:
    arr = np.asarray(digits, dtype=np.int64)
    if arr.ndim != 1 or np.any(arr < 0) or np.any(arr >= d):
        raise ValueError("digits must be a one-dimensional base-d sequence")
    return int(arr @ basis_powers(len(arr), d))


def rows_to_indices(digits: np.ndarray, d: int) -> np.ndarray:
    arr = np.asarray(digits, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError("digits must have shape (n_rows, n_sites)")
    return arr @ basis_powers(arr.shape[1], d)


def product_state(labels: Sequence[int], d: int) -> np.ndarray:
    """Computational product state ``|labels[0],...,labels[N-1]>``."""
    index = digits_to_index(labels, d)
    psi = np.zeros(d ** len(labels), dtype=complex)
    psi[index] = 1.0
    return psi


def plus_product_state(n_sites: int, d: int) -> np.ndarray:
    """Product of local equal superpositions."""
    if n_sites < 1:
        raise ValueError("n_sites must be positive")
    local = np.ones(d, dtype=complex) / np.sqrt(d)
    psi = local.copy()
    for _ in range(n_sites - 1):
        psi = np.kron(psi, local)
    return psi


def kron_all(operators: Iterable[np.ndarray | sp.spmatrix], *, sparse: bool = True):
    ops = list(operators)
    if not ops:
        raise ValueError("at least one operator is required")
    if sparse:
        out = sp.csr_matrix(ops[0])
        for op in ops[1:]:
            out = sp.kron(out, sp.csr_matrix(op), format="csr")
        return out
    out = np.asarray(ops[0], dtype=complex)
    for op in ops[1:]:
        out = np.kron(out, np.asarray(op, dtype=complex))
    return out


def embed_one(local_op, site: int, n_sites: int, d: int, *, sparse: bool = True):
    if not 0 <= site < n_sites:
        raise ValueError("site is out of range")
    identity = sp.eye(d, format="csr", dtype=complex) if sparse else np.eye(d, dtype=complex)
    factors = [identity for _ in range(n_sites)]
    factors[site] = local_op
    return kron_all(factors, sparse=sparse)


def spin_matrices(n_spins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collective ``(Sx, Sy, Sz, m_values)`` in the spin ``S=N/2`` sector.

    The basis is ordered from ``m=S`` to ``m=-S``.
    """
    if n_spins < 1:
        raise ValueError("n_spins must be positive")
    spin = n_spins / 2.0
    m = np.arange(spin, -spin - 1.0, -1.0, dtype=float)
    dim = n_spins + 1
    s_plus = np.zeros((dim, dim), dtype=complex)
    # S_+ maps |m> to sqrt(S(S+1)-m(m+1)) |m+1>.
    for col in range(1, dim):
        m_col = m[col]
        s_plus[col - 1, col] = np.sqrt(spin * (spin + 1.0) - m_col * (m_col + 1.0))
    s_minus = s_plus.conj().T
    sx = 0.5 * (s_plus + s_minus)
    sy = (s_plus - s_minus) / (2.0j)
    sz = np.diag(m).astype(complex)
    return sx, sy, sz, m
