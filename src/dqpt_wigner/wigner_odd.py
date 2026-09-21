"""Canonical discrete Wigner functions for odd local dimension.

The convention follows phase-point operators

    A(q,p) = sum_x omega^(p x) |q+x/2><q-x/2|,
    W_rho(q,p) = Tr[A(q,p) rho] / D,

where all arithmetic is modulo ``d`` and ``D=d**N``.  It obeys
``sum_u W(u)=1`` and ``Tr(rho sigma)=D sum_u W_rho(u) W_sigma(u)``.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from typing import Iterable, Sequence

import numpy as np
from tqdm.auto import tqdm

from .operators import basis_digits, basis_powers, digits_to_index, rows_to_indices


@dataclass(frozen=True)
class SupportMap:
    """Precomputed chord indices for one computational-state Wigner support."""

    q: tuple[int, ...]
    minus_indices: np.ndarray
    plus_indices: np.ndarray
    local_dimension: int

    @property
    def n_sites(self) -> int:
        return len(self.q)

    @property
    def hilbert_dimension(self) -> int:
        return self.local_dimension ** self.n_sites


@dataclass(frozen=True)
class SupportMetrics:
    probability: float
    absolute_weight: float
    negative_mass: float
    positive_mass: float
    minimum: float
    maximum: float
    imaginary_residual: float
    direct_probability: float
    reconstruction_error: float


def _validate_odd_dimension(d: int) -> None:
    if d < 3 or d % 2 == 0:
        raise ValueError("the canonical construction requires odd d >= 3")


def local_phase_point_operator(d: int, q: int, p: int) -> np.ndarray:
    """Single-site phase-point operator for odd ``d``."""
    _validate_odd_dimension(d)
    q %= d
    p %= d
    inverse_two = pow(2, -1, d)
    omega = np.exp(2.0j * np.pi / d)
    operator = np.zeros((d, d), dtype=complex)
    for x in range(d):
        ket = (q + inverse_two * x) % d
        bra = (q - inverse_two * x) % d
        operator[ket, bra] += omega ** (p * x)
    return 0.5 * (operator + operator.conj().T)


def local_phase_point_operators(d: int) -> list[np.ndarray]:
    return [local_phase_point_operator(d, q, p) for q in range(d) for p in range(d)]


def support_map(q: Sequence[int], d: int = 3) -> SupportMap:
    """Precompute indices needed for the Wigner line at fixed ``q``."""
    _validate_odd_dimension(d)
    q_array = np.asarray(q, dtype=np.int16)
    if q_array.ndim != 1 or np.any(q_array < 0) or np.any(q_array >= d):
        raise ValueError("q must be a valid one-dimensional base-d point")
    n_sites = len(q_array)
    x_points = basis_digits(n_sites, d).astype(np.int64)
    inverse_two = pow(2, -1, d)
    q_minus = (q_array[None, :] - inverse_two * x_points) % d
    q_plus = (q_array[None, :] + inverse_two * x_points) % d
    minus_indices = rows_to_indices(q_minus, d).astype(np.int64)
    plus_indices = rows_to_indices(q_plus, d).astype(np.int64)
    minus_indices.setflags(write=False)
    plus_indices.setflags(write=False)
    return SupportMap(tuple(int(value) for value in q_array), minus_indices, plus_indices, d)


def support_wigner_line_pure(
    psi: np.ndarray,
    mapping: SupportMap,
    *,
    zero_tolerance: float = 1e-14,
) -> tuple[np.ndarray, float]:
    """All momentum values at fixed ``q`` by a multidimensional inverse FFT.

    The returned array has ``D=d**N`` entries and sums to the probability of
    the computational basis state labelled by ``q``.
    """
    psi = np.asarray(psi, dtype=complex)
    if psi.ndim != 1 or psi.size != mapping.hilbert_dimension:
        raise ValueError("state dimension is incompatible with the support map")
    chord = psi[mapping.minus_indices] * np.conj(psi[mapping.plus_indices])
    transformed = np.fft.ifftn(chord.reshape((mapping.local_dimension,) * mapping.n_sites))
    imaginary_residual = float(np.max(np.abs(transformed.imag)))
    line = transformed.real.reshape(-1)
    line[np.abs(line) < zero_tolerance] = 0.0
    return line, imaginary_residual


def support_metrics_pure(
    psi: np.ndarray,
    mapping: SupportMap,
    *,
    zero_tolerance: float = 1e-14,
) -> SupportMetrics:
    line, residual = support_wigner_line_pure(psi, mapping, zero_tolerance=zero_tolerance)
    probability = float(np.sum(line))
    absolute_weight = float(np.sum(np.abs(line)))
    negative_mass = float(np.sum(np.maximum(-line, 0.0)))
    positive_mass = float(np.sum(np.maximum(line, 0.0)))
    direct = float(np.abs(np.asarray(psi)[digits_to_index(mapping.q, mapping.local_dimension)]) ** 2)
    return SupportMetrics(
        probability=max(probability, 0.0),
        absolute_weight=absolute_weight,
        negative_mass=negative_mass,
        positive_mass=positive_mass,
        minimum=float(np.min(line)),
        maximum=float(np.max(line)),
        imaginary_residual=residual,
        direct_probability=direct,
        reconstruction_error=float(abs(probability - direct)),
    )


def wigner_from_density_matrix(rho: np.ndarray, d: int, n_sites: int) -> np.ndarray:
    """Full Wigner array for a small mixed state using phase-point operators."""
    _validate_odd_dimension(d)
    rho = np.asarray(rho, dtype=complex)
    dimension = d**n_sites
    if rho.shape != (dimension, dimension):
        raise ValueError("rho has the wrong shape")
    local = local_phase_point_operators(d)
    number_points = (d * d) ** n_sites
    values = np.empty(number_points, dtype=float)
    for flat_index, local_indices in enumerate(np.ndindex(*((d * d,) * n_sites))):
        operator = local[local_indices[0]]
        for index in local_indices[1:]:
            operator = np.kron(operator, local[index])
        values[flat_index] = float(np.trace(operator @ rho).real / dimension)
    return values


_FULL_WORKER: dict[str, object] = {}


def _init_full_worker(psi: np.ndarray, d: int, n_sites: int, zero_tolerance: float) -> None:
    _FULL_WORKER.clear()
    _FULL_WORKER.update(
        psi=np.asarray(psi, dtype=complex),
        d=int(d),
        n_sites=int(n_sites),
        zero_tolerance=float(zero_tolerance),
    )


def _full_line_task(q_index: int) -> tuple[int, np.ndarray]:
    d = int(_FULL_WORKER["d"])
    n_sites = int(_FULL_WORKER["n_sites"])
    q = tuple(int(x) for x in basis_digits(n_sites, d)[q_index])
    mapping = support_map(q, d)
    line, _ = support_wigner_line_pure(
        np.asarray(_FULL_WORKER["psi"]),
        mapping,
        zero_tolerance=float(_FULL_WORKER["zero_tolerance"]),
    )
    return q_index, line


def full_wigner_pure(
    psi: np.ndarray,
    d: int,
    n_sites: int,
    *,
    jobs: int = 1,
    progress: bool = True,
    zero_tolerance: float = 1e-14,
) -> np.ndarray:
    """Full pure-state Wigner array with axes ``(q_index, p_index)``.

    Storage is ``d**(2N)`` and therefore this routine is intended only for
    small systems.  Return-support diagnostics avoid this cost.
    """
    _validate_odd_dimension(d)
    psi = np.asarray(psi, dtype=complex)
    dimension = d**n_sites
    if psi.shape != (dimension,):
        raise ValueError("psi has the wrong dimension")
    result = np.empty((dimension, dimension), dtype=float)
    if jobs <= 1:
        for q_index in tqdm(range(dimension), desc="full odd-d Wigner", disable=not progress):
            q = tuple(int(x) for x in basis_digits(n_sites, d)[q_index])
            result[q_index], _ = support_wigner_line_pure(
                psi, support_map(q, d), zero_tolerance=zero_tolerance
            )
        return result

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=context,
        initializer=_init_full_worker,
        initargs=(psi, d, n_sites, zero_tolerance),
    ) as executor:
        for q_index, line in tqdm(
            executor.map(_full_line_task, range(dimension)),
            total=dimension,
            desc="full odd-d Wigner",
            disable=not progress,
        ):
            result[q_index] = line
    return result


def wigner_l1_norm(wigner: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(wigner, dtype=float))))


def mana(wigner: np.ndarray, *, floor: float = 1e-300) -> float:
    return float(np.log(max(wigner_l1_norm(wigner), floor)))


def sum_negativity(wigner: np.ndarray) -> float:
    return float(0.5 * (wigner_l1_norm(wigner) - 1.0))


def wigner_overlap(wigner_a: np.ndarray, wigner_b: np.ndarray, dimension: int) -> float:
    return float(dimension * np.sum(np.asarray(wigner_a) * np.asarray(wigner_b)))


def validate_local_phase_space(d: int = 3, *, tolerance: float = 1e-11) -> dict[str, float | bool]:
    operators = local_phase_point_operators(d)
    trace_error = max(abs(np.trace(operator) - 1.0) for operator in operators)
    gram_error = 0.0
    involution_error = 0.0
    identity = np.eye(d)
    for first_index, first in enumerate(operators):
        involution_error = max(involution_error, float(np.max(np.abs(first @ first - identity))))
        for second_index, second in enumerate(operators):
            target = d if first_index == second_index else 0.0
            gram_error = max(gram_error, float(abs(np.trace(first @ second) - target)))
    return {
        "trace_error": float(trace_error),
        "gram_error": float(gram_error),
        "involution_error": float(involution_error),
        "passed": bool(max(trace_error, gram_error, involution_error) < tolerance),
    }
