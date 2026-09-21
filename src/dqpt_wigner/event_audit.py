"""Event-level primitives and independent return-support contractions.

Zero returns are represented explicitly rather than replaced by an arbitrary
positive probability floor.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import math
from typing import Iterator

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply


def support_quantities(probability: float, unsigned: float, negative: float, n: int) -> dict:
    """Derive every nonlinear observable from one common primitive record.

    ``negative`` must be accumulated independently, not computed from (A-P)/2.
    Its discrepancy is retained as a numerical consistency diagnostic.
    """
    p, a, nu = map(float, (probability, unsigned, negative))
    if n < 1 or p < 0 or a < 0 or nu < 0:
        raise ValueError("Nonnegative primitive weights and positive support size required")
    if p > 0 and a == 0:
        raise ValueError("A positive probability cannot have zero unsigned weight")
    return {
        "probability": p, "unsigned_weight": a, "negative_mass": nu,
        "physical_rate": -math.log(p) / n if p > 0 else None,
        "support_rate": -math.log(a) / n if a > 0 else None,
        "sign_cost": (math.log(a) - math.log(p)) / n if p > 0 else None,
        "surviving_fraction": p / a if a > 0 else None,
        "zero_probability": p == 0, "zero_unsigned_weight": a == 0,
        "identity_A_minus_P_minus_2nu": a - p - 2 * nu,
    }


def grouped_support(first: dict, second: dict, n: int) -> dict:
    return support_quantities(*(first[k] + second[k] for k in
                                ("probability", "unsigned_weight", "negative_mass")), n)


def assemble_event(tau: float, n: int, branches: dict, competitor: str) -> dict:
    initial, competing = branches["0"], branches[competitor]
    differences = {
        "delta_r": initial["physical_rate"] - competing["physical_rate"],
        "delta_s": initial["support_rate"] - competing["support_rate"],
        "delta_q": initial["sign_cost"] - competing["sign_cost"],
    }
    symmetry = {}
    for key in ("probability", "unsigned_weight", "negative_mass"):
        a, b = branches["1"][key], branches["2"][key]
        symmetry[key + "_relative_error"] = abs(a - b) / max((a + b) / 2, np.finfo(float).tiny)
    return {"tau": float(tau), "block_length": int(n), "delta_definition": "initial 0 minus " + competitor,
            "competitor": competitor, "branches": branches, **differences,
            "identity_delta_r_minus_s_minus_q": differences["delta_r"] - differences["delta_s"] - differences["delta_q"],
            "independent_sector_symmetry": symmetry}


def downward_primitive_root(times: np.ndarray, differences: np.ndarray) -> tuple[float, int]:
    """P0-Pcompetitor or A0-Acompetitor decreases through zero at upward rate exchange."""
    for index in range(len(times) - 1):
        a, b = differences[index:index + 2]
        if a >= 0 and b < 0:
            return float(times[index] + (times[index + 1] - times[index]) * a / (a - b)), index
    raise ValueError("No positive-to-negative primitive crossing in the supplied interval")


def positive_block_return(state, start: int, length: int, branch: int) -> float:
    """Projector contraction independent of the cancellation-prone Wigner sum."""
    left, right, tensors = state._environments(start, length)
    environment = left
    for tensor in tensors:
        matrix = tensor[:, branch, :]
        environment = matrix.T @ environment @ matrix.conj()
    result = np.einsum("ab,ab->", environment, right) / state.norm()
    if abs(result.imag) > 1e-11 or result.real < -1e-12:
        raise FloatingPointError("Invalid projector contraction")
    return float(result.real)


def streamed_wigner_line(state, start: int, length: int, branch: int,
                         batch_size: int = 128) -> Iterator[np.ndarray]:
    """Exact cell contractions streamed in momentum batches; O(batch*chi^2) memory.

    Absolute values are applied only after full contraction with both physical
    environments. This is not an entrywise-absolute-tensor approximation.
    """
    left, right, tensors = state._environments(start, length)
    norm = state.norm()
    omega = np.exp(2j * np.pi / 3)
    fourier = np.array([[omega ** (-p * x) / 3 for x in range(3)] for p in range(3)])
    total = 3 ** length
    divisors = 3 ** np.arange(length - 1, -1, -1)
    for offset in range(0, total, batch_size):
        digits = (np.arange(offset, min(offset + batch_size, total))[:, None] // divisors) % 3
        environments = np.broadcast_to(left, (len(digits), *left.shape)).copy()
        for site, tensor in enumerate(tensors):
            updated = np.zeros((len(digits), tensor.shape[2], tensor.shape[2]), complex)
            for x in range(3):
                ar = tensor[:, (branch + 2 * x) % 3, :]
                ac = tensor[:, (branch - 2 * x) % 3, :].conj()
                contracted = ar.T @ environments @ ac
                updated += fourier[digits[:, site], x, None, None] * contracted
            environments = updated
        values = np.einsum("kab,ab->k", environments, right) / norm
        if np.max(np.abs(values.imag)) > 5e-10:
            raise FloatingPointError("Nonreal Wigner cell")
        yield values.real


def direct_support(state, start: int, length: int, branch: int, batch_size: int = 128) -> dict:
    signed_parts, unsigned_parts, negative_parts = [], [], []
    minimum, maximum = math.inf, -math.inf
    for values in streamed_wigner_line(state, start, length, branch, batch_size):
        signed_parts.append(float(np.sum(values)))
        unsigned_parts.append(float(np.sum(np.abs(values))))
        negative_parts.append(float(np.sum(-values[values < 0])))
        minimum, maximum = min(minimum, float(values.min())), max(maximum, float(values.max()))
    independent_probability = positive_block_return(state, start, length, branch)
    result = support_quantities(independent_probability, math.fsum(unsigned_parts),
                                math.fsum(negative_parts), length)
    result.update({"wigner_signed_sum": math.fsum(signed_parts),
                   "signed_reconstruction_error": math.fsum(signed_parts) - independent_probability,
                   "minimum_wigner": minimum, "maximum_wigner": maximum,
                   "primitive_source": "independent positive projector; fully contracted Wigner absolute and negative cell sums"})
    return result


def ising_tridiagonal(n: int, J: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(n + 1, dtype=float)
    diagonal = -(2 * J / n) * (n / 2 - k) ** 2
    offdiagonal = -h * np.sqrt(np.arange(1, n + 1) * np.arange(n, 0, -1))
    return diagonal, offdiagonal


def ising_spectral_evaluator(n: int, J: float, h: float):
    """Tridiagonal QR spectral evaluator, independent of dense eigendecomposition."""
    diagonal, offdiagonal = ising_tridiagonal(n, J, h)
    values, vectors = eigh_tridiagonal(diagonal, offdiagonal, lapack_driver="stev")
    coefficients = vectors[0]
    def evaluate(t: float) -> tuple[float, float]:
        state = vectors[[0, -1]] @ (coefficients * np.exp(-1j * values * t))
        return tuple(float(abs(z) ** 2) for z in state)
    return evaluate


def ising_sparse_probabilities(n: int, J: float, h: float, t: float) -> tuple[float, float]:
    diagonal, offdiagonal = ising_tridiagonal(n, J, h)
    matrix = diags((offdiagonal, diagonal, offdiagonal), (-1, 0, 1), format="csc")
    initial = np.zeros(n + 1, complex)
    initial[0] = 1
    state = expm_multiply((-1j * t) * matrix, initial, traceA=(-1j * t) * diagonal.sum())
    return float(abs(state[0]) ** 2), float(abs(state[-1]) ** 2)


def ising_decimal_probabilities(n: int, J: float, h: float, t: float, digits: int = 80,
                                maximum_step: float = 0.04) -> dict:
    """Independent arbitrary-precision sparse Taylor evolution using Decimal.

    Precision convergence and a conservative Taylor tail are recorded separately.
    Decimal roundoff is not interval-certified. Extremely tiny down amplitudes
    may be below the absolute Taylor error and are then not relatively resolved.
    """
    with localcontext() as context:
        context.prec = digits
        dJ, dh, time = Decimal(str(J)), Decimal(str(h)), Decimal(str(t))
        count = max(1, math.ceil(float(time) / maximum_step))
        step = time / count
        diagonal = [-(2 * dJ / n) * (Decimal(n) / 2 - k) ** 2 + dJ * n / 2 for k in range(n + 1)]
        off = [-dh * Decimal((k + 1) * (n - k)).sqrt() for k in range(n)]
        norm_bound = max(abs(diagonal[k]) + (abs(off[k-1]) if k else 0)
                         + (abs(off[k]) if k < n else 0) for k in range(n + 1))
        z = norm_bound * step
        tolerance = Decimal(10) ** (-digits + 15) / count
        real, imag = [Decimal(0)] * (n + 1), [Decimal(0)] * (n + 1)
        real[0] = Decimal(1)
        largest_degree, tail_total = 0, Decimal(0)
        for _ in range(count):
            input_infinity_norm = max((a*a+b*b).sqrt() for a,b in zip(real,imag))
            rr, ii = real.copy(), imag.copy()
            tr, ti = real.copy(), imag.copy()
            scalar_term = Decimal(1)
            for degree in range(1, 2001):
                scale = step / degree
                nr, ni = [], []
                for k in range(n + 1):
                    vr = diagonal[k] * ti[k]
                    vi = -diagonal[k] * tr[k]
                    if k:
                        vr += off[k-1] * ti[k-1]
                        vi -= off[k-1] * tr[k-1]
                    if k < n:
                        vr += off[k] * ti[k+1]
                        vi -= off[k] * tr[k+1]
                    nr.append(scale * vr)
                    ni.append(scale * vi)
                tr, ti = nr, ni
                rr = [a + b for a, b in zip(rr, tr)]
                ii = [a + b for a, b in zip(ii, ti)]
                scalar_term *= z / degree
                if degree + 2 > z:
                    tail = scalar_term * z / (degree + 1) / (1 - z / (degree + 2))
                    if tail < tolerance:
                        # Unitarity in the 2-norm prevents amplification between
                        # steps; ||error||_2 <= sqrt(dim)*||error||_infinity.
                        tail_total += Decimal(n+1).sqrt() * tail * input_infinity_norm
                        largest_degree = max(largest_degree, degree)
                        break
            else:
                raise RuntimeError("Taylor expansion exceeded bounded iteration count")
            real, imag = rr, ii
        up, down = real[0] ** 2 + imag[0] ** 2, real[-1] ** 2 + imag[-1] ** 2
        # Dyson's interaction-picture expansion bounds the end-to-end amplitude
        # by the nonnegative hopping exponential: |amp_down| <= sinh(h*t)^N.
        sinh = ((dh * time).exp() - (-dh * time).exp()) / 2
        down_upper = sinh ** (2 * n)
        return {"precision_decimal_digits": digits, "steps": count, "maximum_taylor_degree": largest_degree,
                "prob_up": str(up), "prob_down": str(down),
                "amplitude_taylor_error_bound_ignoring_roundoff": str(tail_total),
                "taylor_bound_derivation": "With z=step*||H_shift||_infinity and degree k: tail <= z^(k+1)/(k+1)!/(1-z/(k+2)); sum sqrt(N+1)*tail*input_infinity_norm over steps using exact evolution's 2-norm unitarity. Decimal rounding is excluded from this bound.",
                "analytic_down_probability_upper_bound": str(down_upper),
                "upper_bound_formula": "P_down(t) <= sinh(abs(h*t))**(2*N)",
                "roundoff_rigor": "precision comparison, not interval arithmetic"}
