"""Exactly solvable Wigner-positive control with a support-driven DQPT-II."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm

from .hierarchy import qutrit_wigner


@dataclass(frozen=True)
class PrecessionControl:
    times: np.ndarray
    one_site_probabilities: np.ndarray
    branch_rates: np.ndarray
    total_rate: np.ndarray
    polarization: np.ndarray
    critical_time: float
    critical_wigner: np.ndarray
    critical_negativity: float


def generator(omega: float = 1.0) -> np.ndarray:
    """Hermitian generator rotating |0> into (|1>+|2>)/sqrt(2)."""
    s = np.array([0.0, 1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    z = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
    return 1j * omega * (np.outer(s, z.conj()) - np.outer(z, s.conj()))


def state(time: float, omega: float = 1.0) -> np.ndarray:
    theta = omega * float(time)
    return np.array([np.cos(theta), np.sin(theta) / np.sqrt(2.0),
                     np.sin(theta) / np.sqrt(2.0)], dtype=np.complex128)


def critical_time(omega: float = 1.0) -> float:
    return float(np.arccos(1.0 / np.sqrt(3.0)) / omega)


def simulate(n_sites: int = 64, tmax: float = 1.45, nt: int = 501,
             omega: float = 1.0, floor: float = 1e-300) -> PrecessionControl:
    times = np.linspace(0.0, tmax, nt)
    p = np.stack([np.abs(state(t, omega)) ** 2 for t in times])
    rates = -np.log(np.maximum(p, floor))
    # Exact finite-N ordered-manifold return; thermodynamic lower envelope is min rates.
    log_terms = -n_sites * rates
    m = np.max(log_terms, axis=1, keepdims=True)
    total = -(m[:, 0] + np.log(np.exp(log_terms - m).sum(axis=1))) / n_sites
    pol = p[:, 0] - 0.5 * (p[:, 1] + p[:, 2])
    tc = critical_time(omega)
    psi_c = state(tc, omega)
    Wc = qutrit_wigner(np.outer(psi_c, psi_c.conj()))
    neg = float(np.clip(-Wc, 0.0, None).sum())
    return PrecessionControl(times, p, rates, total, pol, tc, Wc, neg)
