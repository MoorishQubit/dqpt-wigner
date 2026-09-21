"""High-system-size MPS/TEBD and block Wigner diagnostics for the qutrit Potts chain.

The implementation is dependency-light (NumPy/SciPy only) and uses a Vidal
Gamma--Lambda representation.  It is designed for reproducible N=50--100
checks of symmetry-resolved Loschmidt branches and for central-block canonical
qutrit Wigner lines.  The latter are evaluated without forming the full
3^(2 ell) Wigner array.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Sequence
import math
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

Array = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def qutrit_operators() -> tuple[Array, Array, Array]:
    """Return identity, clock Z, and cyclic shift X in the computational basis."""
    omega = np.exp(2j * np.pi / 3)
    eye = np.eye(3, dtype=np.complex128)
    z = np.diag([1.0, omega, omega**2]).astype(np.complex128)
    # X|s> = |s+1 mod 3>.
    x = np.roll(eye, 1, axis=0)
    return eye, z, x


@lru_cache(maxsize=128)
def potts_gates(J: float, h: float, dt: float) -> tuple[Array, Array]:
    """Second-order TEBD gates for an open three-state quantum Potts chain.

    H = -J sum_i (Z_i^dag Z_{i+1} + Z_i Z_{i+1}^dag)
        -h sum_i (X_i + X_i^dag).

    One full step uses U_h(dt/2), an even/odd/even interaction Strang split,
    and U_h(dt/2).  ``u_bond_half`` is returned; the full odd-bond gate is its
    square.  The tensor index order is (out_i, out_j, in_i, in_j).
    """
    eye, z, x = qutrit_operators()
    onsite = -h * (x + x.conj().T)
    hint = -J * (np.kron(z.conj().T, z) + np.kron(z, z.conj().T))
    u_site_half = expm(-1j * onsite * (dt / 2.0))
    u_bond_half = expm(-1j * hint * (dt / 2.0)).reshape(3, 3, 3, 3)
    return u_site_half.astype(np.complex128), u_bond_half.astype(np.complex128)


@dataclass
class TruncationRecord:
    discarded_weight: float = 0.0
    maximum_bond: int = 1


class VidalMPS:
    """Open-boundary Vidal Gamma--Lambda matrix-product state."""

    def __init__(self, gammas: list[Array], lambdas: list[RealArray]):
        self.gammas = [np.asarray(g, dtype=np.complex128) for g in gammas]
        self.lambdas = [np.asarray(l, dtype=np.float64) for l in lambdas]
        self.n_sites = len(gammas)
        self.local_dim = int(gammas[0].shape[1])
        if len(lambdas) != max(0, self.n_sites - 1):
            raise ValueError("Expected N-1 Schmidt-value arrays")

    @classmethod
    def product(cls, n_sites: int, symbol: int = 0, local_dim: int = 3) -> "VidalMPS":
        if n_sites < 1:
            raise ValueError("n_sites must be positive")
        if not 0 <= symbol < local_dim:
            raise ValueError("invalid product-state symbol")
        gammas: list[Array] = []
        for _ in range(n_sites):
            g = np.zeros((1, local_dim, 1), dtype=np.complex128)
            g[0, symbol, 0] = 1.0
            gammas.append(g)
        lambdas = [np.ones(1, dtype=np.float64) for _ in range(n_sites - 1)]
        return cls(gammas, lambdas)

    def copy(self) -> "VidalMPS":
        return VidalMPS([g.copy() for g in self.gammas], [l.copy() for l in self.lambdas])

    @property
    def maximum_bond_dimension(self) -> int:
        return max([1] + [int(x.size) for x in self.lambdas])

    def _left_lambda(self, site: int) -> RealArray:
        return np.ones(1) if site == 0 else self.lambdas[site - 1]

    def _right_lambda(self, site: int) -> RealArray:
        return np.ones(1) if site == self.n_sites - 1 else self.lambdas[site]

    def apply_one_site(self, gate: Array, site: int) -> None:
        self.gammas[site] = np.einsum(
            "st,atb->asb", gate, self.gammas[site], optimize=True
        )

    def apply_two_site(
        self,
        gate: Array,
        site: int,
        max_bond: int,
        cutoff: float,
    ) -> TruncationRecord:
        """Apply a nearest-neighbor gate and truncate the updated Schmidt bond."""
        if site < 0 or site >= self.n_sites - 1:
            raise IndexError("invalid bond")
        gl = self.gammas[site]
        gr = self.gammas[site + 1]
        lam_l = self._left_lambda(site)
        lam_m = self.lambdas[site]
        lam_r = self._right_lambda(site + 1)

        theta = np.einsum(
            "a,asb,b,btc,c->astc",
            lam_l,
            gl,
            lam_m,
            gr,
            lam_r,
            optimize=True,
        )
        theta = np.einsum("uvst,astc->auvc", gate, theta, optimize=True)
        chi_l, d1, d2, chi_r = theta.shape
        matrix = theta.reshape(chi_l * d1, d2 * chi_r)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)

        keep_mask = s > float(cutoff)
        keep = int(min(max_bond, max(1, int(np.count_nonzero(keep_mask)))))
        discarded = float(np.sum(np.square(s[keep:])))
        u = u[:, :keep]
        s = s[:keep]
        vh = vh[:keep, :]
        snorm = float(np.linalg.norm(s))
        if snorm == 0.0:
            raise FloatingPointError("TEBD produced a zero Schmidt spectrum")
        s = s / snorm

        eps = 1e-14
        inv_l = np.where(np.abs(lam_l) > eps, 1.0 / lam_l, 0.0)
        inv_r = np.where(np.abs(lam_r) > eps, 1.0 / lam_r, 0.0)
        new_gl = u.reshape(chi_l, d1, keep) * inv_l[:, None, None]
        new_gr = vh.reshape(keep, d2, chi_r) * inv_r[None, None, :]
        self.gammas[site] = new_gl.astype(np.complex128, copy=False)
        self.lambdas[site] = s.astype(np.float64, copy=False)
        self.gammas[site + 1] = new_gr.astype(np.complex128, copy=False)
        return TruncationRecord(discarded_weight=discarded, maximum_bond=keep)

    def step(
        self,
        J: float,
        h: float,
        dt: float,
        max_bond: int,
        cutoff: float,
    ) -> TruncationRecord:
        """Apply one second-order open-chain TEBD step."""
        u_site_half, u_bond_half = potts_gates(J, h, dt)
        u_bond_full = np.einsum(
            "abij,ijcd->abcd", u_bond_half, u_bond_half, optimize=True
        )
        for i in range(self.n_sites):
            self.apply_one_site(u_site_half, i)
        total_discarded = 0.0
        max_seen = self.maximum_bond_dimension
        # Strang split H_even(dt/2) H_odd(dt) H_even(dt/2).
        for i in range(0, self.n_sites - 1, 2):
            rec = self.apply_two_site(u_bond_half, i, max_bond, cutoff)
            total_discarded += rec.discarded_weight
            max_seen = max(max_seen, rec.maximum_bond)
        for i in range(1, self.n_sites - 1, 2):
            rec = self.apply_two_site(u_bond_full, i, max_bond, cutoff)
            total_discarded += rec.discarded_weight
            max_seen = max(max_seen, rec.maximum_bond)
        for i in range(0, self.n_sites - 1, 2):
            rec = self.apply_two_site(u_bond_half, i, max_bond, cutoff)
            total_discarded += rec.discarded_weight
            max_seen = max(max_seen, rec.maximum_bond)
        for i in range(self.n_sites):
            self.apply_one_site(u_site_half, i)
        return TruncationRecord(total_discarded, max_seen)

    def as_left_weighted_tensors(self) -> list[Array]:
        """Return conventional MPS tensors A_i=Gamma_i Lambda_i."""
        tensors: list[Array] = []
        for i, gamma in enumerate(self.gammas):
            if i < self.n_sites - 1:
                tensors.append(gamma * self.lambdas[i][None, None, :])
            else:
                tensors.append(gamma.copy())
        return tensors

    def product_log_amplitude(self, symbol: int) -> tuple[float, complex]:
        """Return log(abs(amplitude)) and unit phase for |symbol>^N."""
        vec = np.ones(1, dtype=np.complex128)
        log_scale = 0.0
        for tensor in self.as_left_weighted_tensors():
            vec = vec @ tensor[:, symbol, :]
            scale = float(np.linalg.norm(vec))
            if scale == 0.0:
                return -np.inf, 0.0j
            vec /= scale
            log_scale += math.log(scale)
        amp = complex(vec[0])
        abs_amp = abs(amp)
        if abs_amp == 0.0:
            return -np.inf, 0.0j
        return log_scale + math.log(abs_amp), amp / abs_amp

    def product_probability(self, symbol: int) -> float:
        logabs, _ = self.product_log_amplitude(symbol)
        if not np.isfinite(logabs):
            return 0.0
        return float(np.exp(2.0 * logabs))

    def product_rate(self, symbol: int) -> float:
        logabs, _ = self.product_log_amplitude(symbol)
        if not np.isfinite(logabs):
            return np.inf
        return float(-2.0 * logabs / self.n_sites)

    def one_site_density(self, site: int) -> Array:
        """One-site reduced state using the canonical Gamma--Lambda tensor."""
        lam_l = self._left_lambda(site)
        lam_r = self._right_lambda(site)
        theta = np.einsum(
            "a,asb,b->asb", lam_l, self.gammas[site], lam_r, optimize=True
        )
        rho = np.einsum("asb,atb->st", theta, theta.conj(), optimize=True)
        tr = np.trace(rho)
        if abs(tr) > 0:
            rho /= tr
        return rho

    def ordered_local_moment(self, site: int | None = None) -> float:
        """Potts ordered moment P0-(P1+P2)/2 at one site."""
        j = self.n_sites // 2 if site is None else int(site)
        rho = self.one_site_density(j)
        probs = np.real(np.diag(rho))
        return float(probs[0] - 0.5 * (probs[1] + probs[2]))

    def norm(self) -> float:
        tensors = self.as_left_weighted_tensors()
        env = np.ones((1, 1), dtype=np.complex128)
        for a in tensors:
            env = np.einsum("ab,asr,bsv->rv", env, a, a.conj(), optimize=True)
        return float(np.real_if_close(env[0, 0]))

    def _environments(self, start: int, length: int) -> tuple[Array, Array, list[Array]]:
        if start < 0 or length < 1 or start + length > self.n_sites:
            raise ValueError("invalid block")
        tensors = self.as_left_weighted_tensors()
        left = np.ones((1, 1), dtype=np.complex128)
        for a in tensors[:start]:
            left = np.einsum("ab,asr,bsv->rv", left, a, a.conj(), optimize=True)
        right = np.ones((1, 1), dtype=np.complex128)
        for a in reversed(tensors[start + length :]):
            right = np.einsum("asr,bsv,rv->ab", a, a.conj(), right, optimize=True)
        return left, right, tensors[start : start + length]

    def block_wigner_line(
        self,
        start: int,
        length: int,
        branch: int,
    ) -> RealArray:
        """Canonical qutrit Wigner values on q=(branch,...,branch).

        The returned array has 3**length values indexed by the momentum string.
        It uses a Fourier-transfer contraction whose cost is O(3^ell chi^3)
        and never forms the full 3^(2 ell) Wigner function.
        """
        if branch not in (0, 1, 2):
            raise ValueError("branch must be 0, 1, or 2")
        left, right, tensors = self._environments(start, length)
        omega = np.exp(2j * np.pi / 3)
        fourier = np.asarray(
            [[omega ** (-p * x) / 3.0 for x in range(3)] for p in range(3)],
            dtype=np.complex128,
        )
        matrices = left[None, :, :]
        for tensor in tensors:
            updated: list[Array] = []
            for x in range(3):
                row = (branch + 2 * x) % 3
                col = (branch - 2 * x) % 3
                ar = tensor[:, row, :]
                ac = tensor[:, col, :].conj()
                updated.append(
                    np.einsum("lr,klm,mv->krv", ar, matrices, ac, optimize=True)
                )
            stack = np.stack(updated, axis=1)  # prefix, x, right, right'
            new = np.einsum("px,kxab->kpab", fourier, stack, optimize=True)
            matrices = new.reshape(-1, new.shape[-2], new.shape[-1])
        values = np.einsum("kab,ab->k", matrices, right, optimize=True)
        if np.max(np.abs(np.imag(values))) > 5e-9:
            raise FloatingPointError("Wigner line acquired a non-negligible imaginary part")
        return np.real(values).astype(np.float64)


@dataclass(frozen=True)
class WignerSupportDiagnostic:
    branch: str
    block_length: int
    probability: float
    unsigned_weight: float
    negative_mass: float
    physical_rate: float
    support_rate: float
    sign_cost: float
    surviving_fraction: float
    minimum_wigner: float
    maximum_wigner: float


def diagnose_wigner_line(values: RealArray, block_length: int, branch: str) -> WignerSupportDiagnostic:
    vals = np.asarray(values, dtype=float)
    probability = float(np.sum(vals))
    unsigned = float(np.sum(np.abs(vals)))
    negative = float(np.sum(np.abs(vals[vals < 0.0])))
    tiny = np.finfo(float).tiny
    pclip = max(probability, tiny)
    aclip = max(unsigned, pclip)
    r = -math.log(pclip) / block_length
    s = -math.log(aclip) / block_length
    q = math.log(aclip / pclip) / block_length
    return WignerSupportDiagnostic(
        branch=branch,
        block_length=block_length,
        probability=probability,
        unsigned_weight=unsigned,
        negative_mass=negative,
        physical_rate=r,
        support_rate=s,
        sign_cost=q,
        surviving_fraction=probability / unsigned if unsigned > 0 else 1.0,
        minimum_wigner=float(np.min(vals)),
        maximum_wigner=float(np.max(vals)),
    )


def combine_support_diagnostics(
    diagnostics: Sequence[WignerSupportDiagnostic],
    branch: str = "competing",
) -> WignerSupportDiagnostic:
    if not diagnostics:
        raise ValueError("at least one diagnostic is required")
    ell = diagnostics[0].block_length
    if any(d.block_length != ell for d in diagnostics):
        raise ValueError("all diagnostics must use the same block length")
    p = sum(d.probability for d in diagnostics)
    a = sum(d.unsigned_weight for d in diagnostics)
    nu = sum(d.negative_mass for d in diagnostics)
    tiny = np.finfo(float).tiny
    return WignerSupportDiagnostic(
        branch=branch,
        block_length=ell,
        probability=p,
        unsigned_weight=a,
        negative_mass=nu,
        physical_rate=-math.log(max(p, tiny)) / ell,
        support_rate=-math.log(max(a, max(p, tiny))) / ell,
        sign_cost=math.log(max(a, max(p, tiny)) / max(p, tiny)) / ell,
        surviving_fraction=p / a if a > 0 else 1.0,
        minimum_wigner=min(d.minimum_wigner for d in diagnostics),
        maximum_wigner=max(d.maximum_wigner for d in diagnostics),
    )


def evolve_to_times(
    n_sites: int,
    times: Sequence[float],
    J: float,
    h: float,
    dt: float,
    max_bond: int,
    cutoff: float,
    initial_symbol: int = 0,
) -> Iterator[tuple[float, VidalMPS, float]]:
    """Evolve once and yield copies at sorted target times.

    The final piece of every interval is taken with an exact fractional TEBD
    step, avoiding a hidden rounding of requested measurement times.
    """
    targets = np.asarray(times, dtype=float)
    if targets.ndim != 1 or np.any(np.diff(targets) < -1e-13):
        raise ValueError("times must be a sorted one-dimensional sequence")
    state = VidalMPS.product(n_sites, initial_symbol, 3)
    current = 0.0
    cumulative_discarded = 0.0
    for target in targets:
        if target < current - 1e-12:
            raise ValueError("target times must be nondecreasing")
        while current + dt < target - 1e-12:
            rec = state.step(J, h, dt, max_bond, cutoff)
            cumulative_discarded += rec.discarded_weight
            current += dt
        remainder = float(target - current)
        if remainder > 1e-12:
            rec = state.step(J, h, remainder, max_bond, cutoff)
            cumulative_discarded += rec.discarded_weight
            current = float(target)
        yield float(target), state.copy(), cumulative_discarded


def exact_potts_hamiltonian(n_sites: int, J: float, h: float) -> Array:
    """Dense open-chain Hamiltonian for small-system validation only."""
    eye, z, x = qutrit_operators()
    dim = 3**n_sites
    H = np.zeros((dim, dim), dtype=np.complex128)

    def kron_ops(ops: list[Array]) -> Array:
        out = ops[0]
        for op in ops[1:]:
            out = np.kron(out, op)
        return out

    for i in range(n_sites):
        ops = [eye] * n_sites
        ops[i] = x + x.conj().T
        H -= h * kron_ops(ops)
    for i in range(n_sites - 1):
        ops = [eye] * n_sites
        ops[i] = z.conj().T
        ops[i + 1] = z
        H -= J * kron_ops(ops)
        ops = [eye] * n_sites
        ops[i] = z
        ops[i + 1] = z.conj().T
        H -= J * kron_ops(ops)
    return H


def exact_product_probabilities(n_sites: int, J: float, h: float, t: float) -> tuple[float, float, float]:
    H = exact_potts_hamiltonian(n_sites, J, h)
    dim = 3**n_sites
    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[0] = 1.0
    psi = expm(-1j * H * t) @ psi0
    probs: list[float] = []
    for a in range(3):
        index = sum(a * (3 ** (n_sites - 1 - i)) for i in range(n_sites))
        probs.append(float(abs(psi[index]) ** 2))
    return tuple(probs)  # type: ignore[return-value]
