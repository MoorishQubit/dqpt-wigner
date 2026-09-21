"""Two-site infinite TEBD and audited ordered-product return transfers.

The represented state is ... Gamma_A lambda_A Gamma_B lambda_B ... .
One cell contains TWO qutrits.  For cell tensors A,B, the norm transfer has
radius eta and a product-return matrix T_a=A[a] B[a].  A unique, contributing
leading eigenvalue z_a gives r_a=-log(abs(z_a))+log(eta)/2 per site.

No eigensolver residual is treated as a rigorous eigenvalue enclosure.  Open
boundary coefficients, modulus gaps, nonnormal eigenvalue condition numbers,
and explicit finite contractions are exposed instead of silently assuming a
spectral radius always contributes.  These are numerical diagnostics for the
represented iMPS, not error bounds for exact Hamiltonian dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eig, expm, svd
from scipy.sparse.linalg import LinearOperator, eigs

Array = np.ndarray


@lru_cache(maxsize=256)
def evolution_gates(model: str, field: float, dtau: float) -> tuple[Array, Array, Array]:
    """Return onsite half and AB-half/BA-full bond gates, in tau=J*t units.

    ``rotation`` is the independent qutrit control with Omega=1 and J=0;
    its time coordinate is Omega*t, not the Potts tau.
    """
    eye = np.eye(3, dtype=complex)
    x = np.roll(eye, 1, axis=0)
    z = np.diag(np.exp(2j * np.pi * np.arange(3) / 3))
    if model == "potts":
        onsite = -field * (x + x.conj().T)
        bond = -(np.kron(z.conj().T, z) + np.kron(z, z.conj().T))
    elif model == "rotation":
        v = np.array([0, 1, 1], dtype=complex) / np.sqrt(2)
        e0 = eye[:, 0]
        onsite = 1j * (np.outer(v, e0) - np.outer(e0, v))
        bond = np.zeros((9, 9), dtype=complex)
    else:
        raise ValueError(f"unknown model {model}")
    return (expm(-0.5j * dtau * onsite),
            expm(-0.5j * dtau * bond).reshape(3, 3, 3, 3),
            expm(-1j * dtau * bond).reshape(3, 3, 3, 3))


@dataclass
class InfiniteMPS:
    gamma_a: Array
    gamma_b: Array
    lambda_a: Array
    lambda_b: Array
    tau: float = 0.0
    accumulated_discarded_weight: float = 0.0
    maximum_discarded_weight: float = 0.0
    steps: int = 0

    @classmethod
    def product(cls, symbol: int = 0, vector: Array | None = None) -> "InfiniteMPS":
        v = np.eye(3, dtype=complex)[:, symbol] if vector is None else np.asarray(vector, complex)
        v = v / np.linalg.norm(v)
        return cls(v.reshape(1, 3, 1).copy(), v.reshape(1, 3, 1).copy(),
                   np.ones(1), np.ones(1))

    def copy(self) -> "InfiniteMPS":
        return InfiniteMPS(self.gamma_a.copy(), self.gamma_b.copy(),
                           self.lambda_a.copy(), self.lambda_b.copy(), self.tau,
                           self.accumulated_discarded_weight, self.maximum_discarded_weight,
                           self.steps)

    @property
    def bonds(self) -> tuple[int, int]:
        return self.lambda_a.size, self.lambda_b.size

    def _pair(self, gate: Array, orientation: str, max_bond: int, cutoff: float) -> float:
        if orientation == "ab":
            ga, gb, middle, outer = self.gamma_a, self.gamma_b, self.lambda_a, self.lambda_b
        elif orientation == "ba":
            ga, gb, middle, outer = self.gamma_b, self.gamma_a, self.lambda_b, self.lambda_a
        else:
            raise ValueError(orientation)
        # Outer Schmidt values appear on BOTH ends of the updated theta.
        theta = np.einsum("a,asb,b,btc,c->astc", outer, ga, middle, gb, outer, optimize=True)
        theta = np.einsum("uvst,astc->auvc", gate, theta, optimize=True)
        dl, d, _, dr = theta.shape
        u, values, vh = svd(theta.reshape(dl * d, d * dr), full_matrices=False,
                            check_finite=False, lapack_driver="gesdd")
        keep = min(max_bond, max(1, int(np.count_nonzero(values > cutoff))))
        full_weight = float(values @ values)
        discarded = float(values[keep:] @ values[keep:] / full_weight)
        kept = values[:keep]
        kept /= np.linalg.norm(kept)
        # Every stored Schmidt value is positive by the truncation rule; no
        # hidden cutoff or probability clamp is introduced in this inverse.
        if np.any(outer <= 0):
            raise FloatingPointError("nonpositive stored Schmidt value")
        new_a = u[:, :keep].reshape(dl, d, keep) / outer[:, None, None]
        new_b = vh[:keep].reshape(keep, d, dr) / outer[None, None, :]
        if orientation == "ab":
            self.gamma_a, self.gamma_b, self.lambda_a = new_a, new_b, kept
        else:
            self.gamma_b, self.gamma_a, self.lambda_b = new_a, new_b, kept
        self.accumulated_discarded_weight += discarded
        self.maximum_discarded_weight = max(self.maximum_discarded_weight, discarded)
        return discarded

    def step(self, dtau: float, max_bond: int, cutoff: float = 1e-12,
             model: str = "potts", field: float = 1.5) -> None:
        if dtau <= 0 or max_bond < 1 or cutoff <= 0:
            raise ValueError("positive time step, bond cap, and Schmidt cutoff required")
        onsite, half, full = evolution_gates(model, float(field), float(dtau))
        for name in ("gamma_a", "gamma_b"):
            setattr(self, name, np.einsum("st,atb->asb", onsite, getattr(self, name)))
        self._pair(half, "ab", max_bond, cutoff)
        self._pair(full, "ba", max_bond, cutoff)
        self._pair(half, "ab", max_bond, cutoff)
        for name in ("gamma_a", "gamma_b"):
            setattr(self, name, np.einsum("st,atb->asb", onsite, getattr(self, name)))
        self.tau += dtau
        self.steps += 1

    def cell_tensors(self, normalize: bool = False) -> tuple[Array, Array]:
        a = self.gamma_a * self.lambda_a[None, None, :]
        b = self.gamma_b * self.lambda_b[None, None, :]
        if normalize:
            eta = transfer_fixed_points(a, b)["eta"]
            factor = eta ** (-0.25)
            a, b = a * factor, b * factor
        return a, b

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """Archive resumable Vidal data AND normalized conventional cell tensors."""
        a, b = self.cell_tensors()
        fixed = transfer_fixed_points(a, b)
        factor = fixed["eta"] ** (-0.25)
        record = dict(metadata or {})
        record.update({"schema": "dqpt_wigner.itebd.v1", "tau": self.tau,
                       "steps": self.steps, "accumulated_discarded_weight": self.accumulated_discarded_weight,
                       "maximum_discarded_weight": self.maximum_discarded_weight,
                       "raw_norm_eigenvalue_per_two_site_cell": fixed["eta"],
                       "normalized_tensor_convention": "A=GammaA*lambdaA/right; B=GammaB*lambdaB/right; each divided by eta**.25",
                       "unit_cell_length": 2})
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, gamma_a=self.gamma_a, gamma_b=self.gamma_b,
                            lambda_a=self.lambda_a, lambda_b=self.lambda_b,
                            A=a * factor, B=b * factor, left=fixed["left"], right=fixed["right"],
                            metadata=np.array(json.dumps(record, sort_keys=True)))
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> tuple["InfiniteMPS", dict[str, Any]]:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["schema"] != "dqpt_wigner.itebd.v1":
                raise ValueError("unknown checkpoint schema")
            state = cls(data["gamma_a"].copy(), data["gamma_b"].copy(),
                        data["lambda_a"].copy(), data["lambda_b"].copy(), float(meta["tau"]),
                        float(meta["accumulated_discarded_weight"]),
                        float(meta["maximum_discarded_weight"]), int(meta["steps"]))
        return state, meta


def transfer_apply(tensor: Array, matrix: Array, adjoint: bool = False) -> Array:
    if adjoint:
        return sum(tensor[:, s, :].conj().T @ matrix @ tensor[:, s, :] for s in range(3))
    return sum(tensor[:, s, :] @ matrix @ tensor[:, s, :].conj().T for s in range(3))


def cell_transfer(a: Array, b: Array, matrix: Array, adjoint: bool = False) -> Array:
    if adjoint:
        return transfer_apply(b, transfer_apply(a, matrix, True), True)
    return transfer_apply(a, transfer_apply(b, matrix))


def transfer_fixed_points(a: Array, b: Array, tolerance: float = 2e-13,
                          max_iterations: int = 2000) -> dict[str, Any]:
    """Positive power iteration with independently checked left/right residuals.

    Starting from I samples every positive norm sector. Nonconvergence raises;
    uniqueness is NOT certified by this routine. Trace(left @ right)=1.
    """
    dimension = a.shape[0]
    environments = []
    eigenvalues = []
    iterations = []
    for adjoint in (False, True):
        current = np.eye(dimension, dtype=complex) / dimension
        for iteration in range(max_iterations):
            candidate = cell_transfer(a, b, current, adjoint)
            candidate = (candidate + candidate.conj().T) / 2
            eta = float(np.trace(candidate).real)
            if eta <= 0 or not np.isfinite(eta):
                raise FloatingPointError("nonpositive norm transfer value")
            candidate /= eta
            error = np.linalg.norm(candidate - current)
            current = candidate
            if error < tolerance:
                break
        else:
            raise RuntimeError("norm fixed-point iteration did not converge")
        environments.append(current)
        eigenvalues.append(eta)
        iterations.append(iteration + 1)
    right, left = environments
    overlap = np.trace(left @ right).real
    left = left / overlap
    eta = float(sum(eigenvalues) / 2)
    return {"eta": eta, "left": left, "right": right,
            "right_residual": float(np.linalg.norm(cell_transfer(a, b, right) - eta * right) / np.linalg.norm(right)),
            "left_residual": float(np.linalg.norm(cell_transfer(a, b, left, True) - eta * left) / np.linalg.norm(left)),
            "left_right_eigenvalue_difference": abs(eigenvalues[0] - eigenvalues[1]),
            "iterations": iterations,
            "minimum_left_eigenvalue": float(np.linalg.eigvalsh(left).min()),
            "minimum_right_eigenvalue": float(np.linalg.eigvalsh(right).min())}


def branch_spectrum(a: Array, b: Array, symbol: int, norm_eta: float = 1.0,
                    left_boundary: Array | None = None,
                    right_boundary: Array | None = None) -> dict[str, Any]:
    """Inspect all eigenvalues, open-boundary coefficients, and periodic sectors.

    Periodic trace coefficients are one per simple eigenvalue, so a unique
    leading modulus necessarily contributes. For arbitrary open boundaries a
    dominant eigenvalue can be projected out; its coefficient is reported and
    ``open_selected_index`` is chosen only above a declared numerical threshold.
    Degenerate leading moduli are flagged rather than assigned a limit.
    """
    matrix = a[:, symbol, :] @ b[:, symbol, :]
    values, left, right = eig(matrix, left=True, right=True, check_finite=False)
    order = np.argsort(-np.abs(values))
    values, left, right = values[order], left[:, order], right[:, order]
    d = matrix.shape[0]
    v = np.ones(d, dtype=complex) / np.sqrt(d) if left_boundary is None else np.asarray(left_boundary, complex)
    w = np.ones(d, dtype=complex) / np.sqrt(d) if right_boundary is None else np.asarray(right_boundary, complex)
    coefficients, conditions = [], []
    for i in range(len(values)):
        overlap = np.vdot(left[:, i], right[:, i])
        if abs(overlap) < 1e-300:
            coefficients.append(complex(np.nan, np.nan))
            conditions.append(float("inf"))
        else:
            coefficients.append(np.vdot(v, right[:, i]) * np.vdot(left[:, i], w) / overlap)
            conditions.append(float(1.0 / abs(overlap)))
    radius = float(abs(values[0]))
    gap = 1.0 if len(values) == 1 else (float(1 - abs(values[1]) / radius) if radius else 0.0)
    selected = next((i for i, value in enumerate(coefficients) if abs(value) > 1e-10), None)
    residual = np.linalg.norm(matrix @ right[:, 0] - values[0] * right[:, 0])
    denom = max(np.linalg.norm(matrix), radius)
    return {"branch": symbol, "rate": (-math.log(radius) + 0.5 * math.log(norm_eta)) if radius else float("inf"),
            "leading_eigenvalue": [float(values[0].real), float(values[0].imag)],
            "radius": radius, "relative_modulus_gap": gap,
            "unique_leading_modulus_numerically": bool(radius > 0 and gap > 1e-8),
            "eigenvalue_condition": conditions[0],
            "scaled_eigen_residual": float(residual / denom) if denom else 0.0,
            "open_leading_coefficient": [float(coefficients[0].real), float(coefficients[0].imag)],
            "open_leading_coefficient_abs": float(abs(coefficients[0])),
            "open_selected_index": selected,
            "open_selected_rate": None if selected is None or abs(values[selected]) == 0 else float(-np.log(abs(values[selected])) + 0.5 * np.log(norm_eta)),
            "coefficient_selection_threshold": 1e-10,
            "eigenvalues": [[float(z.real), float(z.imag)] for z in values]}


def finite_open_return(a: Array, b: Array, symbol: int, cells: int,
                       left_boundary: Array | None = None,
                       right_boundary: Array | None = None) -> dict[str, float]:
    """Independent, rescaled finite-MPS probability and norm contractions.

    This contracts v^dagger (A_a B_a)^L w and the complete finite norm, not a
    leading-eigenvalue substitution. Boundaries define this finite MPS family,
    not the independently evolved finite open Potts chain.
    """
    if cells < 1:
        raise ValueError("positive cell count required")
    d = a.shape[0]
    v = np.ones(d, complex) / np.sqrt(d) if left_boundary is None else np.asarray(left_boundary, complex)
    w = np.ones(d, complex) / np.sqrt(d) if right_boundary is None else np.asarray(right_boundary, complex)
    amplitude_vector = w.copy()
    env = np.outer(w, w.conj())
    amplitude_log = 0.0
    norm_log = 0.0
    matrix = a[:, symbol, :] @ b[:, symbol, :]
    zero = False
    for _ in range(cells):
        if not zero:
            amplitude_vector = matrix @ amplitude_vector
            scale = np.linalg.norm(amplitude_vector)
            if scale == 0:
                zero = True
            else:
                amplitude_vector /= scale
                amplitude_log += np.log(scale)
        env = cell_transfer(a, b, env)
        norm_scale = np.linalg.norm(env)
        if norm_scale == 0:
            raise FloatingPointError("zero finite MPS norm")
        env /= norm_scale
        norm_log += np.log(norm_scale)
    endpoint = abs(np.vdot(v, amplitude_vector))
    norm_endpoint = np.vdot(v, env @ v).real
    if norm_endpoint <= 0:
        raise FloatingPointError("nonpositive finite MPS norm endpoint")
    norm_log += np.log(norm_endpoint)
    log_probability = -float("inf") if zero or endpoint == 0 else float(2 * (amplitude_log + np.log(endpoint)) - norm_log)
    return {"cells": cells, "sites": 2 * cells, "log_probability": log_probability,
            "log_norm": float(norm_log), "rate": -log_probability / (2 * cells)}


def finite_periodic_state(a: Array, b: Array, cells: int) -> Array:
    """Explicit small-ring state for tests; exponential output is intentional."""
    if cells > 5:
        raise ValueError("explicit state is only for small-system validation")
    tensors = [a, b] * cells
    vector = []
    for string in itertools.product(range(3), repeat=2 * cells):
        matrix = np.eye(a.shape[0], dtype=complex)
        for tensor, physical in zip(tensors, string):
            matrix = matrix @ tensor[:, physical, :]
        vector.append(np.trace(matrix))
    return np.asarray(vector)


def norm_spectrum_diagnostics(a: Array, b: Array, tolerance: float = 1e-11) -> dict[str, Any]:
    """Matrix-free peripheral norm spectrum; numerical gap, not a certificate."""
    dimension = a.shape[0]
    size = dimension**2
    action = lambda x: cell_transfer(a, b, x.reshape(dimension, dimension)).reshape(-1)
    if size <= 4:
        dense = np.column_stack([action(v) for v in np.eye(size, dtype=complex)])
        values, vectors = np.linalg.eig(dense)
    else:
        operator = LinearOperator((size, size), matvec=action, dtype=np.complex128)
        rng = np.random.default_rng(17491)
        v0 = rng.normal(size=size) + 1j * rng.normal(size=size)
        values, vectors = eigs(operator, k=3, which="LM", v0=v0, tol=tolerance, maxiter=2000)
    order = np.argsort(-np.abs(values))
    values, vectors = values[order], vectors[:, order]
    residuals = [float(np.linalg.norm(action(vectors[:, i]) - values[i] * vectors[:, i]))
                 for i in range(values.size)]
    return {"eigenvalues": [[float(z.real), float(z.imag)] for z in values],
            "relative_modulus_gap": 1.0 if len(values) == 1 else float(1 - abs(values[1] / values[0])),
            "residuals": residuals, "requested_tolerance": tolerance,
            "rigorous_enclosure": False}


def boundary_asymptotic_check(a: Array, b: Array, symbol: int,
                              cells: tuple[int, ...]) -> dict[str, Any]:
    """Check the explicit open contractions against selected-sector prefactors."""
    fixed = transfer_fixed_points(a, b)
    spectrum = branch_spectrum(a, b, symbol, fixed["eta"])
    v = np.ones(a.shape[0], complex) / np.sqrt(a.shape[0])
    w = v.copy()
    norm_coefficient = float((np.vdot(v, fixed["right"] @ v)
                              * np.vdot(w, fixed["left"] @ w)).real)
    coefficient = spectrum["open_leading_coefficient_abs"]
    rows = []
    for length in cells:
        measured = finite_open_return(a, b, symbol, length, v, w)
        prediction = (spectrum["rate"] - (2 * np.log(coefficient) - np.log(norm_coefficient)) / (2 * length)
                      if coefficient > 0 and norm_coefficient > 0 else None)
        rows.append({**measured, "selected_sector_prediction_with_prefactor": prediction,
                     "prediction_residual": None if prediction is None else measured["rate"] - prediction})
    return {"norm_boundary_coefficient": norm_coefficient,
            "return_boundary_coefficient_abs": coefficient,
            "boundary": "v=w=normalized all-ones in stored virtual gauge; not claimed to equal original OBC quench boundary",
            "finite_contractions": rows}


def bulk_return_asymptotic_check(a: Array, b: Array, symbol: int,
                                 cells: tuple[int, ...]) -> dict[str, Any]:
    """Signed bulk-block return with actual infinite-state environments.

    For a simple leading T eigenpair (u,v), the coefficient is
    (u† L u)(v† R v)/|v†u|², with Tr(LR)=1. A strictly positive coefficient
    gives the same exponential rate as the periodic global amplitude. This
    statement applies only to SIGNED product probabilities of the represented
    iMPS; it does not exchange environmental trace and absolute Wigner sums.
    """
    fixed = transfer_fixed_points(a, b)
    matrix = a[:, symbol, :] @ b[:, symbol, :]
    values, left_eigen, right_eigen = eig(matrix, left=True, right=True)
    index = int(np.argmax(np.abs(values)))
    u, v = right_eigen[:, index], left_eigen[:, index]
    coefficient = float((np.vdot(u, fixed["left"] @ u)
                         * np.vdot(v, fixed["right"] @ v)).real / abs(np.vdot(v, u))**2)
    radius = abs(values[index])
    rate = -np.log(radius) + .5 * np.log(fixed["eta"]) if radius else float("inf")
    x = fixed["right"].copy()
    log_scale = 0.0
    rows = []
    targets = set(cells)
    for length in range(1, max(cells) + 1):
        x = matrix @ x @ matrix.conj().T
        scale = np.linalg.norm(x)
        if scale == 0:
            log_scale = -float("inf")
        else:
            x /= scale
            log_scale += np.log(scale)
        if length not in targets:
            continue
        endpoint = float(np.trace(fixed["left"] @ x).real)
        if endpoint < 0:
            raise FloatingPointError("negative bulk product probability endpoint")
        log_p = (float(np.log(endpoint) + log_scale - length * np.log(fixed["eta"]))
                 if endpoint > 0 else -float("inf"))
        prediction = rate - np.log(coefficient) / (2 * length) if coefficient > 0 else None
        rows.append({"cells": length, "sites": 2 * length, "log_probability": log_p,
                     "rate": -log_p / (2 * length),
                     "selected_sector_prediction_with_prefactor": prediction,
                     "prediction_residual": None if prediction is None else -log_p / (2 * length) - prediction})
    return {"leading_boundary_coefficient": coefficient,
            "coefficient_positive_numerically": bool(coefficient > 1e-12),
            "finite_bulk_block_returns": rows,
            "scope": "signed product probabilities of the fixed represented iMPS; no unsigned/global limit identification"}


def measure_state(state: InfiniteMPS, finite_cells: tuple[int, ...] = ()) -> dict[str, Any]:
    a, b = state.cell_tensors()
    fixed = transfer_fixed_points(a, b)
    branches = [branch_spectrum(a, b, symbol, fixed["eta"]) for symbol in range(3)]
    # One-site projectors in the infinite environment, independent of Schmidt
    # canonicality and using the norm transfer's actual fixed points.
    left, right, eta = fixed["left"], fixed["right"], fixed["eta"]
    rb = transfer_apply(b, right)
    local_a = [float(np.trace(left @ a[:, s, :] @ rb @ a[:, s, :].conj().T).real / eta) for s in range(3)]
    la = transfer_apply(a, left, True)
    local_b = [float(np.trace(la @ b[:, s, :] @ right @ b[:, s, :].conj().T).real / eta) for s in range(3)]
    result = {"tau": state.tau, "bonds": list(state.bonds), "branches": branches,
              "rate_initial": branches[0]["rate"], "rate_competing1": branches[1]["rate"],
              "rate_competing2": branches[2]["rate"],
              "delta_rate": branches[0]["rate"] - branches[1]["rate"],
              "branch_symmetry_error": abs(branches[1]["rate"] - branches[2]["rate"]),
              "norm": {k: v for k, v in fixed.items() if k not in ("left", "right")},
              "local_populations_a": local_a, "local_populations_b": local_b,
              "translation_population_error": float(np.max(np.abs(np.array(local_a) - local_b))),
              "accumulated_discarded_weight": state.accumulated_discarded_weight,
              "maximum_discarded_weight": state.maximum_discarded_weight,
              "finite_open": {str(symbol): [finite_open_return(a, b, symbol, cells) for cells in finite_cells] for symbol in range(3)}}
    return result
