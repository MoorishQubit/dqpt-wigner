"""Block-resolved discrete phase-space diagnostics for qutrit DQPTs.

The canonical odd-dimensional Wigner convention used here is

    W(q,p) = d^{-ell} sum_x omega^{-p.x}
             rho[q + x/2, q - x/2],

with all arithmetic over Z_d.  It obeys sum_p W(q,p)=rho[q,q].
For a computational stabilizer branch |a...a>, the fixed-q affine line
therefore gives the branch probability.  Absolute line sums isolate the
support magnitude from signed Wigner cancellation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SupportDiagnostic:
    branch: int
    block_length: int
    probability: float
    unsigned_weight: float
    negative_mass: float
    support_rate: float
    sign_cost: float
    surviving_fraction: float
    minimum_wigner_value: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _digits(indices: Array, length: int, base: int) -> Array:
    """Convert integer indices to big-endian base-``base`` digit arrays."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    out = np.empty((indices.size, length), dtype=np.int64)
    work = indices.copy()
    for k in range(length - 1, -1, -1):
        out[:, k] = work % base
        work //= base
    return out


def _encode(digits: Array, base: int) -> Array:
    digits = np.asarray(digits, dtype=np.int64)
    powers = base ** np.arange(digits.shape[-1] - 1, -1, -1, dtype=np.int64)
    return digits @ powers


def qutrit_phase_point_operators() -> Array:
    """Return the nine canonical qutrit phase-point operators A(q,p)."""
    d = 3
    omega = np.exp(2j * np.pi / d)
    inv2 = pow(2, -1, d)
    ops = np.empty((d, d, d, d), dtype=np.complex128)
    for q in range(d):
        for p in range(d):
            A = np.zeros((d, d), dtype=np.complex128)
            for x in range(d):
                row = (q + inv2 * x) % d
                col = (q - inv2 * x) % d
                A[row, col] += omega ** (-p * x)
            ops[q, p] = A
    return ops


def qutrit_wigner(rho: Array) -> Array:
    """Canonical one-qutrit Wigner function with sum W=1."""
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape != (3, 3):
        raise ValueError("rho must have shape (3,3)")
    ops = qutrit_phase_point_operators()
    W = np.einsum("qpij,ji->qp", ops, rho, optimize=True) / 3.0
    return np.real_if_close(W, tol=1000).real


def line_wigner_from_coherences(coherences: Array, d: int = 3) -> Array:
    """FFT a fixed-q coherence tensor into the corresponding Wigner line.

    ``coherences[x]`` must equal rho[q+x/2,q-x/2]/d**ell.
    """
    values = np.fft.fftn(np.asarray(coherences, dtype=np.complex128))
    if np.max(np.abs(values.imag)) > 5e-9:
        raise ValueError("Wigner line is not real within tolerance")
    return values.real


def _support_from_line(line: Array, probability: float, ell: int, branch: int,
                       eps: float = 1e-300) -> SupportDiagnostic:
    line = np.asarray(line, dtype=float)
    p_line = float(line.sum())
    p = max(float(np.real(probability)), 0.0)
    # Use the phase-space sum when it agrees; retain independently evaluated p
    # to expose numerical inconsistencies in callers/tests.
    if not np.isclose(p_line, p, rtol=5e-7, atol=5e-10):
        raise ValueError(f"line sum {p_line:.12g} != probability {p:.12g}")
    A = float(np.abs(line).sum())
    nu = float(np.clip(-line, 0.0, None).sum())
    if not np.isclose(A - p, 2.0 * nu, rtol=5e-7, atol=5e-10):
        raise ValueError("A-P=2 nu identity failed")
    A_safe = max(A, eps)
    p_safe = max(p, eps)
    s = -np.log(A_safe) / ell
    q = np.log(A_safe / p_safe) / ell
    return SupportDiagnostic(
        branch=int(branch), block_length=int(ell), probability=p,
        unsigned_weight=A, negative_mass=nu, support_rate=float(s),
        sign_cost=float(max(q, 0.0)), surviving_fraction=float(p_safe / A_safe),
        minimum_wigner_value=float(line.min(initial=0.0)),
    )


def central_block_matrix(psi: Array, n_sites: int, block_length: int,
                         d: int = 3) -> Array:
    """Return a matrix whose rows index a central block and columns its environment."""
    if not 1 <= block_length <= n_sites:
        raise ValueError("block_length must lie in [1,n_sites]")
    psi = np.asarray(psi, dtype=np.complex128).reshape((d,) * n_sites)
    start = (n_sites - block_length) // 2
    block_axes = list(range(start, start + block_length))
    env_axes = [i for i in range(n_sites) if i not in block_axes]
    moved = np.transpose(psi, block_axes + env_axes)
    return moved.reshape(d ** block_length, d ** (n_sites - block_length))


def support_line_from_state(psi: Array, n_sites: int, block_length: int,
                            branch: int, d: int = 3,
                            batch_size: int = 4096) -> tuple[Array, float]:
    """Compute a central-block Wigner return line without forming a dense rho.

    Memory is O(batch_size * d**(N-ell)) rather than O(d**(2 ell)).
    """
    if d % 2 == 0:
        raise ValueError("canonical fixed-q construction requires odd d")
    if not 0 <= branch < d:
        raise ValueError("invalid branch")
    M = central_block_matrix(psi, n_sites, block_length, d=d)
    n_x = d ** block_length
    inv2 = pow(2, -1, d)
    q = np.full((1, block_length), branch, dtype=np.int64)
    coh = np.empty(n_x, dtype=np.complex128)
    for lo in range(0, n_x, batch_size):
        hi = min(lo + batch_size, n_x)
        x = _digits(np.arange(lo, hi), block_length, d)
        row = _encode((q + inv2 * x) % d, d)
        col = _encode((q - inv2 * x) % d, d)
        coh[lo:hi] = np.einsum(
            "be,be->b", M[row], M[col].conj(), optimize=True
        ) / n_x
    line = line_wigner_from_coherences(coh.reshape((d,) * block_length), d=d)
    qindex = int(branch * sum(d ** k for k in range(block_length)))
    probability = float(np.vdot(M[qindex], M[qindex]).real)
    return line, probability


def block_support_diagnostic(psi: Array, n_sites: int, block_length: int,
                             branch: int, d: int = 3,
                             batch_size: int = 4096) -> SupportDiagnostic:
    line, p = support_line_from_state(
        psi, n_sites, block_length, branch, d=d, batch_size=batch_size
    )
    return _support_from_line(line, p, block_length, branch)


def block_hierarchy(psi: Array, n_sites: int,
                    block_lengths: Sequence[int] | None = None,
                    branches: Sequence[int] = (0, 1, 2),
                    d: int = 3, batch_size: int = 4096) -> list[SupportDiagnostic]:
    if block_lengths is None:
        block_lengths = tuple(range(1, n_sites + 1))
    out: list[SupportDiagnostic] = []
    for ell in block_lengths:
        for branch in branches:
            out.append(block_support_diagnostic(
                psi, n_sites, int(ell), int(branch), d=d,
                batch_size=batch_size
            ))
    return out


def combine_diagnostics(items: Sequence[SupportDiagnostic], branches: Sequence[int],
                        label: int = -1) -> SupportDiagnostic:
    """Combine disjoint return supports, e.g. Potts branches 1 and 2."""
    chosen = [x for x in items if x.branch in set(branches)]
    if not chosen:
        raise ValueError("no diagnostics selected")
    ell = chosen[0].block_length
    if any(x.block_length != ell for x in chosen):
        raise ValueError("all diagnostics must have the same block length")
    p = sum(x.probability for x in chosen)
    A = sum(x.unsigned_weight for x in chosen)
    nu = sum(x.negative_mass for x in chosen)
    eps = 1e-300
    return SupportDiagnostic(
        branch=label, block_length=ell, probability=p,
        unsigned_weight=A, negative_mass=nu,
        support_rate=-np.log(max(A, eps)) / ell,
        sign_cost=max(np.log(max(A, eps) / max(p, eps)) / ell, 0.0),
        surviving_fraction=max(p, eps) / max(A, eps),
        minimum_wigner_value=min(x.minimum_wigner_value for x in chosen),
    )


def one_site_populations_from_wigner(rho: Array) -> Array:
    """Computational populations as fixed-q Wigner line sums."""
    W = qutrit_wigner(rho)
    return W.sum(axis=1)


def potts_local_polarization(populations: Sequence[float]) -> float:
    """Real Z3 polarization relative to branch 0: p0-(p1+p2)/2."""
    p = np.asarray(populations, dtype=float)
    if p.shape != (3,):
        raise ValueError("three populations required")
    return float(p[0] - 0.5 * (p[1] + p[2]))


def stabilizer_overlap_diagnostic(rho: Array, sigma: Array,
                                  tol: float = 1e-10) -> dict[str, float]:
    """One-qutrit overlap/support diagnostic for an arbitrary stabilizer branch.

    For a pure qutrit stabilizer state, W_sigma is 1/3 on one affine line and
    zero elsewhere.  The returned quantities are invariant under a simultaneous
    Clifford transformation of rho and sigma because the canonical Wigner
    function is permuted by Clifford conjugation.
    """
    Wr = qutrit_wigner(rho)
    Ws = qutrit_wigner(sigma)
    support = Ws > tol
    if support.sum() != 3 or not np.allclose(Ws[support], 1/3, atol=1e-9):
        raise ValueError("sigma is not recognized as a pure qutrit stabilizer state")
    probability = float(3.0 * np.sum(Wr * Ws))
    unsigned = float(3.0 * np.sum(np.abs(Wr * Ws)))
    negative = float(np.clip(-Wr[support], 0.0, None).sum())
    return {
        "probability": probability,
        "unsigned_weight": unsigned,
        "negative_mass": negative,
        "sign_cost": float(np.log(max(unsigned, 1e-300) / max(probability, 1e-300))),
    }
