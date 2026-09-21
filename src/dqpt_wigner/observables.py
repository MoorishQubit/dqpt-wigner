"""Local observables, reduced states, and time-averaged DQPT-I diagnostics."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .operators import basis_digits, spin_matrices


def partial_trace_pure(
    psi: np.ndarray,
    *,
    local_dimension: int,
    n_sites: int,
    keep: Sequence[int],
) -> np.ndarray:
    """Reduced density matrix of selected sites for a pure state."""
    keep_tuple = tuple(int(site) for site in keep)
    if len(set(keep_tuple)) != len(keep_tuple):
        raise ValueError("keep contains duplicate sites")
    if any(site < 0 or site >= n_sites for site in keep_tuple):
        raise ValueError("keep contains an invalid site")
    traced = tuple(site for site in range(n_sites) if site not in keep_tuple)
    tensor = np.asarray(psi, dtype=complex).reshape((local_dimension,) * n_sites)
    permuted = tensor.transpose(keep_tuple + traced)
    d_keep = local_dimension ** len(keep_tuple)
    d_trace = local_dimension ** len(traced)
    matrix = permuted.reshape(d_keep, d_trace)
    return matrix @ matrix.conj().T


def ising_magnetization_z(psi: np.ndarray, n_sites: int) -> float:
    probabilities = np.abs(np.asarray(psi)) ** 2
    z_sum = np.sum(1.0 - 2.0 * basis_digits(n_sites, 2), axis=1)
    return float(probabilities @ z_sum / n_sites)


def potts_complex_order(psi: np.ndarray, n_sites: int) -> complex:
    probabilities = np.abs(np.asarray(psi)) ** 2
    omega = np.exp(2.0j * np.pi / 3.0)
    local_sum = np.sum(omega ** basis_digits(n_sites, 3), axis=1)
    return complex(probabilities @ local_sum / n_sites)


def potts_sector_order(psi: np.ndarray, n_sites: int, reference_sector: int = 0) -> float:
    """Order projected onto one of the three ordered directions."""
    order = potts_complex_order(psi, n_sites)
    angle = 2.0 * np.pi * reference_sector / 3.0
    return float(np.real(np.exp(-1.0j * angle) * order))


def time_average(
    values: np.ndarray,
    times: np.ndarray,
    *,
    start_fraction: float = 0.5,
) -> float:
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.shape != times.shape:
        raise ValueError("values and times must have the same shape")
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("start_fraction must lie in [0,1)")
    start = min(int(np.floor(start_fraction * len(times))), len(times) - 1)
    if len(times[start:]) < 2 or times[-1] == times[start]:
        return float(np.mean(values[start:]))
    return float(np.trapezoid(values[start:], times[start:]) / (times[-1] - times[start]))


def late_time_statistics(values: np.ndarray, times: np.ndarray, *, start_fraction: float = 0.5) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "signed_average": time_average(values, times, start_fraction=start_fraction),
        "absolute_average": time_average(np.abs(values), times, start_fraction=start_fraction),
        "rms": float(np.sqrt(time_average(values**2, times, start_fraction=start_fraction))),
    }


def loschmidt_branch_probability(psi: np.ndarray, branch_index: int) -> float:
    return float(np.abs(np.asarray(psi)[branch_index]) ** 2)


def rate_from_probability(probability: np.ndarray | float, n_sites: int, *, floor: float = 1e-300):
    return -np.log(np.maximum(np.asarray(probability, dtype=float), floor)) / n_sites


def finite_difference_curvature(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if len(values) < 5:
        return np.zeros_like(values)
    return np.abs(np.gradient(np.gradient(values, times), times))


def collective_spin_expectations(states: np.ndarray, n_spins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized Bloch components ``2<S_alpha>/N``."""
    sx, sy, sz, _ = spin_matrices(n_spins)
    states = np.asarray(states, dtype=complex)
    ex = np.einsum("ti,ij,tj->t", states.conj(), sx, states).real
    ey = np.einsum("ti,ij,tj->t", states.conj(), sy, states).real
    ez = np.einsum("ti,ij,tj->t", states.conj(), sz, states).real
    factor = 2.0 / n_spins
    return factor * ex, factor * ey, factor * ez


def one_site_density_from_bloch(mx: float, my: float, mz: float) -> np.ndarray:
    return 0.5 * np.array(
        [[1.0 + mz, mx - 1.0j * my], [mx + 1.0j * my, 1.0 - mz]],
        dtype=complex,
    )
