"""Collective Ising benchmark with symmetry-resolved return branches and order."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp, trapezoid
from scipy.linalg import eigh

Array = NDArray[np.complex128]


def spin_matrices(n_sites: int) -> tuple[Array, Array, Array]:
    S = n_sites / 2.0
    m = np.arange(S, -S - 1.0, -1.0)
    dim = n_sites + 1
    sz = np.diag(m).astype(np.complex128)
    sp = np.zeros((dim, dim), dtype=np.complex128)
    # Basis index k has m=S-k. S+ raises m and therefore maps k -> k-1.
    for k in range(1, dim):
        mk = m[k]
        sp[k - 1, k] = np.sqrt(S * (S + 1.0) - mk * (mk + 1.0))
    sm = sp.conj().T
    sx = 0.5 * (sp + sm)
    sy = (sp - sm) / (2j)
    return sx, sy, sz


def collective_hamiltonian(n_sites: int, J: float, h: float) -> Array:
    sx, _, sz = spin_matrices(n_sites)
    return -(2.0 * J / n_sites) * (sz @ sz) - 2.0 * h * sx


def exact_collective_trajectory(
    n_sites: int,
    J: float,
    h: float,
    times: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    H = collective_hamiltonian(n_sites, J, h)
    evals, evecs = eigh(H)
    psi0 = np.zeros(n_sites + 1, dtype=np.complex128)
    psi0[0] = 1.0
    coeff = evecs.conj().T @ psi0
    phases = np.exp(-1j * np.outer(evals, times))
    states = evecs @ (coeff[:, None] * phases)
    p_up = np.abs(states[0]) ** 2
    p_down = np.abs(states[-1]) ** 2
    _, _, sz = spin_matrices(n_sites)
    mz = np.real(np.einsum("it,ij,jt->t", states.conj(), sz, states, optimize=True)) * (2.0 / n_sites)
    tiny = np.finfo(float).tiny
    r_up = -np.log(np.maximum(p_up, tiny)) / n_sites
    r_down = -np.log(np.maximum(p_down, tiny)) / n_sites
    echo = p_up + p_down
    r_total = -np.log(np.maximum(echo, tiny)) / n_sites
    return {
        "time": times,
        "prob_up": p_up.astype(float),
        "prob_down": p_down.astype(float),
        "return_echo": echo.astype(float),
        "rate_up": r_up.astype(float),
        "rate_down": r_down.astype(float),
        "rate_total": r_total.astype(float),
        "rate_lower_envelope": np.minimum(r_up, r_down).astype(float),
        "delta_rate": (r_up - r_down).astype(float),
        "magnetization_z": mz.astype(float),
    }


def mean_field_rhs(_: float, m: NDArray[np.float64], J: float, h: float) -> NDArray[np.float64]:
    x, y, z = m
    return np.asarray([2.0 * J * z * y, 2.0 * h * z - 2.0 * J * z * x, -2.0 * h * y])


def mean_field_trajectory(
    J: float,
    h: float,
    times: NDArray[np.float64],
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> dict[str, NDArray[np.float64]]:
    sol = solve_ivp(
        lambda t, m: mean_field_rhs(t, m, J, h),
        (float(times[0]), float(times[-1])),
        np.asarray([0.0, 0.0, 1.0]),
        t_eval=times,
        rtol=rtol,
        atol=atol,
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return {"time": times, "mx": sol.y[0], "my": sol.y[1], "mz": sol.y[2]}


def late_time_order(time: NDArray[np.float64], mz: NDArray[np.float64], start: float) -> tuple[float, float]:
    mask = time >= float(start)
    if np.count_nonzero(mask) < 2:
        raise ValueError("late-time interval has fewer than two samples")
    signed = float(trapezoid(mz[mask], time[mask]) / (time[mask][-1] - time[mask][0]))
    absolute = float(trapezoid(np.abs(mz[mask]), time[mask]) / (time[mask][-1] - time[mask][0]))
    return signed, absolute
