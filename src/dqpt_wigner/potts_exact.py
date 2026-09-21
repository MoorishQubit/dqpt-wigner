"""Sparse exact dynamics for the three-state quantum Potts/clock chain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class PottsTrajectory:
    times: np.ndarray
    probabilities: np.ndarray  # shape (nt,3)
    initial_rate: np.ndarray
    competing_rate: np.ndarray
    total_rate: np.ndarray
    delta_rate: np.ndarray


def _digits_all(n_sites: int, d: int = 3) -> np.ndarray:
    dim = d ** n_sites
    values = np.arange(dim, dtype=np.int64)
    out = np.empty((dim, n_sites), dtype=np.int8)
    work = values.copy()
    for k in range(n_sites - 1, -1, -1):
        out[:, k] = work % d
        work //= d
    return out


def potts_hamiltonian(n_sites: int, J: float = 1.0, h: float = 1.0,
                      periodic: bool = True) -> sparse.csr_matrix:
    """H=-J sum(Z_i^dag Z_j+h.c.)-h sum(X_i+X_i^dag)."""
    if n_sites < 2:
        raise ValueError("n_sites must be at least 2")
    d = 3
    dim = d ** n_sites
    digs = _digits_all(n_sites, d)
    bonds = [(i, i + 1) for i in range(n_sites - 1)]
    if periodic and n_sites > 2:
        bonds.append((n_sites - 1, 0))
    diag = np.zeros(dim, dtype=float)
    for i, j in bonds:
        diff = (digs[:, j] - digs[:, i]) % d
        diag += -2.0 * J * np.cos(2.0 * np.pi * diff / d)
    rows = [np.arange(dim, dtype=np.int64)]
    cols = [np.arange(dim, dtype=np.int64)]
    data = [diag.astype(np.complex128)]
    powers = d ** np.arange(n_sites - 1, -1, -1, dtype=np.int64)
    base = np.arange(dim, dtype=np.int64)
    for site in range(n_sites):
        old = digs[:, site].astype(np.int64)
        for shift in (-1, 1):
            new = (old + shift) % d
            target = base + (new - old) * powers[site]
            rows.append(target)
            cols.append(base)
            data.append(np.full(dim, -h, dtype=np.complex128))
    H = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim), dtype=np.complex128
    ).tocsr()
    H.sum_duplicates()
    return H


def ordered_state(n_sites: int, branch: int, d: int = 3) -> np.ndarray:
    dim = d ** n_sites
    index = int(branch * sum(d ** k for k in range(n_sites)))
    psi = np.zeros(dim, dtype=np.complex128)
    psi[index] = 1.0
    return psi


def evolve_state(H: sparse.spmatrix, psi0: np.ndarray, time: float) -> np.ndarray:
    return np.asarray(expm_multiply((-1j * float(time)) * H, psi0))


def ordered_probabilities(states: np.ndarray, n_sites: int) -> np.ndarray:
    states = np.asarray(states)
    if states.ndim == 1:
        states = states[None, :]
    indices = [int(a * sum(3 ** k for k in range(n_sites))) for a in range(3)]
    return np.abs(states[:, indices]) ** 2


def trajectory(n_sites: int, h_final: float, times: Sequence[float], J: float = 1.0,
               periodic: bool = True, floor: float = 1e-300) -> PottsTrajectory:
    t = np.asarray(times, dtype=float)
    H = potts_hamiltonian(n_sites, J=J, h=h_final, periodic=periodic)
    psi0 = ordered_state(n_sites, 0)
    if len(t) == 1:
        states = evolve_state(H, psi0, t[0])[None, :]
    elif np.allclose(t, np.linspace(t[0], t[-1], len(t))):
        states = expm_multiply(-1j * H, psi0, start=t[0], stop=t[-1], num=len(t), endpoint=True)
    else:
        states = np.stack([evolve_state(H, psi0, x) for x in t])
    probs = ordered_probabilities(states, n_sites)
    p0 = np.maximum(probs[:, 0], floor)
    pp = np.maximum(probs[:, 1] + probs[:, 2], floor)
    r0 = -np.log(p0) / n_sites
    rp = -np.log(pp) / n_sites
    total = -np.log(np.maximum(p0 + pp, floor)) / n_sites
    return PottsTrajectory(t, probs, r0, rp, total, r0 - rp)


def first_directional_crossing(times: np.ndarray, delta: np.ndarray,
                               direction: str = "up") -> float | None:
    t = np.asarray(times, dtype=float)
    y = np.asarray(delta, dtype=float)
    for i in range(len(t) - 1):
        cond = (y[i] <= 0 < y[i + 1]) if direction == "up" else (y[i] >= 0 > y[i + 1])
        if cond:
            if y[i + 1] == y[i]:
                return float(t[i])
            return float(t[i] - y[i] * (t[i + 1] - t[i]) / (y[i + 1] - y[i]))
    return None
