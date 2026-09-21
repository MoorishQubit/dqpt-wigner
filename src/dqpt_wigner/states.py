"""Initial states and symmetry-broken return manifolds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .operators import plus_product_state, product_state


@dataclass(frozen=True)
class GroundState:
    energy: float
    vector: np.ndarray


def ground_state(hamiltonian: np.ndarray | sp.spmatrix, *, tol: float = 1e-12) -> GroundState:
    """Lowest eigenpair, using dense diagonalization only for small matrices."""
    dim = hamiltonian.shape[0]
    if sp.issparse(hamiltonian) and dim > 512:
        values, vectors = spla.eigsh(hamiltonian, k=1, which="SA", tol=tol)
        order = np.argsort(values.real)
        return GroundState(float(values[order[0]].real), vectors[:, order[0]].astype(complex))
    dense = hamiltonian.toarray() if sp.issparse(hamiltonian) else np.asarray(hamiltonian)
    values, vectors = la.eigh(dense)
    return GroundState(float(values[0].real), vectors[:, 0].astype(complex))


def make_initial_state(kind: str, n_sites: int, d: int, hamiltonian=None) -> np.ndarray:
    """Create a named initial state.

    Supported names are ``ordered0``, ``ordered1``, ``ordered2`` (when
    available), ``plus``, and ``ground``.
    """
    normalized = kind.strip().lower()
    if normalized in {"ordered0", "all0", "up"}:
        return product_state([0] * n_sites, d)
    if normalized in {"ordered1", "all1", "down"}:
        return product_state([1] * n_sites, d)
    if normalized in {"ordered2", "all2"}:
        if d < 3:
            raise ValueError("ordered2 requires d >= 3")
        return product_state([2] * n_sites, d)
    if normalized in {"plus", "paramagnetic"}:
        return plus_product_state(n_sites, d)
    if normalized == "ground":
        if hamiltonian is None:
            raise ValueError("a Hamiltonian is required for the ground state")
        return ground_state(hamiltonian).vector
    raise ValueError(f"unknown initial state: {kind}")


def potts_ordered_branch_labels(n_sites: int) -> dict[str, tuple[int, ...]]:
    return {f"sector_{a}": (a,) * n_sites for a in range(3)}


def ising_ordered_branch_labels(n_sites: int) -> dict[str, tuple[int, ...]]:
    return {"up": (0,) * n_sites, "down": (1,) * n_sites}
