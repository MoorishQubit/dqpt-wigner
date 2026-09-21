"""Small, dependency-light qutrit MPS/TEBD validator for the Potts quench.

This is a validation engine, not a replacement for a mature tensor-network
library. It uses open boundaries, second-order Suzuki--Trotter evolution,
explicit SVD truncation, and bounded-batch reduced-block Wigner line
contractions.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import expm
from tqdm.auto import tqdm

from .hierarchy import SupportDiagnostic, _support_from_line, _digits, line_wigner_from_coherences

Array = np.ndarray


def qutrit_ops() -> tuple[Array, Array]:
    d = 3
    omega = np.exp(2j * np.pi / d)
    Z = np.diag([omega ** k for k in range(d)]).astype(np.complex128)
    X = np.zeros((d, d), dtype=np.complex128)
    for k in range(d):
        X[(k + 1) % d, k] = 1.0
    return X, Z


@dataclass
class MPS:
    tensors: list[Array]

    @classmethod
    def product(cls, n_sites: int, branch: int = 0, d: int = 3) -> "MPS":
        tensors: list[Array] = []
        for _ in range(n_sites):
            A = np.zeros((1, d, 1), dtype=np.complex128)
            A[0, branch, 0] = 1.0
            tensors.append(A)
        return cls(tensors)

    @property
    def n_sites(self) -> int:
        return len(self.tensors)

    def copy(self) -> "MPS":
        return MPS([x.copy() for x in self.tensors])

    def norm2(self) -> float:
        env = np.ones((1, 1), dtype=np.complex128)
        for A in self.tensors:
            env = np.einsum("ab,asr,bsu->ru", env, A, A.conj(), optimize=True)
        return float(env[0, 0].real)

    def normalize(self) -> None:
        n = self.norm2()
        if n <= 0:
            raise FloatingPointError("nonpositive MPS norm")
        self.tensors[0] /= np.sqrt(n)

    def product_amplitude(self, branch: int) -> complex:
        v = np.ones((1,), dtype=np.complex128)
        for A in self.tensors:
            v = v @ A[:, branch, :]
        return complex(v[0])

    def apply_one_site(self, gate: Array, site: int) -> None:
        self.tensors[site] = np.einsum("sp,lpr->lsr", gate, self.tensors[site], optimize=True)

    def apply_two_site(self, gate: Array, site: int, max_bond: int,
                       cutoff: float) -> float:
        A, B = self.tensors[site], self.tensors[site + 1]
        theta = np.einsum("aib,bjc->aijc", A, B, optimize=True)
        theta = np.einsum("uvij,aijc->auvc", gate, theta, optimize=True)
        left, d1, d2, right = theta.shape
        mat = theta.reshape(left * d1, d2 * right)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        keep = min(max_bond, max(1, int(np.count_nonzero(S > cutoff))))
        discarded = float(np.sum(S[keep:] ** 2))
        U, S, Vh = U[:, :keep], S[:keep], Vh[:keep]
        self.tensors[site] = U.reshape(left, d1, keep)
        self.tensors[site + 1] = (S[:, None] * Vh).reshape(keep, d2, right)
        return discarded

    def left_environment(self, stop: int) -> Array:
        env = np.ones((1, 1), dtype=np.complex128)
        for A in self.tensors[:stop]:
            env = np.einsum("ab,asr,bsu->ru", env, A, A.conj(), optimize=True)
        return env

    def right_environment(self, start: int) -> Array:
        env = np.ones((1, 1), dtype=np.complex128)
        for A in reversed(self.tensors[start:]):
            env = np.einsum("asr,bsu,ru->ab", A, A.conj(), env, optimize=True)
        return env

    def block_matrix_elements(self, start: int, rows: Array, cols: Array,
                              batch_size: int = 256) -> Array:
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if rows.shape != cols.shape or rows.ndim != 2:
            raise ValueError("rows and cols must have shape (batch,ell)")
        ell = rows.shape[1]
        EL = self.left_environment(start)
        ER = self.right_environment(start + ell)
        out = np.empty(rows.shape[0], dtype=np.complex128)
        for lo in range(0, rows.shape[0], batch_size):
            hi = min(lo + batch_size, rows.shape[0])
            M = np.broadcast_to(EL, (hi - lo,) + EL.shape).copy()
            for k, A in enumerate(self.tensors[start:start + ell]):
                Ar = np.transpose(A[:, rows[lo:hi, k], :], (1, 0, 2))
                Ac = np.transpose(A[:, cols[lo:hi, k], :], (1, 0, 2))
                M = np.einsum("bij,bik,bjl->bkl", M, Ar, Ac.conj(), optimize=True)
            out[lo:hi] = np.einsum("bij,ij->b", M, ER, optimize=True)
        return out

    def block_support(self, block_length: int, branch: int,
                      batch_size: int = 256) -> SupportDiagnostic:
        d = 3
        start = (self.n_sites - block_length) // 2
        n_x = d ** block_length
        inv2 = 2
        coh = np.empty(n_x, dtype=np.complex128)
        q = np.full((1, block_length), branch, dtype=np.int64)
        for lo in range(0, n_x, batch_size):
            hi = min(lo + batch_size, n_x)
            x = _digits(np.arange(lo, hi), block_length, d)
            rows = (q + inv2 * x) % d
            cols = (q - inv2 * x) % d
            coh[lo:hi] = self.block_matrix_elements(
                start, rows, cols, batch_size=batch_size
            ) / n_x
        line = line_wigner_from_coherences(coh.reshape((d,) * block_length), d=d)
        qq = np.full((1, block_length), branch, dtype=np.int64)
        p = float(self.block_matrix_elements(start, qq, qq, batch_size=1)[0].real)
        return _support_from_line(line, p, block_length, branch)


@lru_cache(maxsize=128)
def gates(J: float, h: float, dt: float) -> tuple[Array, Array]:
    X, Z = qutrit_ops()
    h1 = -h * (X + X.conj().T)
    h2 = -J * (np.kron(Z.conj().T, Z) + np.kron(Z, Z.conj().T))
    u1 = expm(-1j * dt * h1).reshape(3, 3)
    u2 = expm(-1j * dt * h2).reshape(3, 3, 3, 3)
    return u1, u2


def tebd_step(mps: MPS, J: float, h: float, dt: float, max_bond: int,
              cutoff: float) -> float:
    if dt == 0:
        return 0.0
    u1h, _ = gates(float(J), float(h), float(dt) / 2.0)
    _, u2h = gates(float(J), float(h), float(dt) / 2.0)
    _, u2 = gates(float(J), float(h), float(dt))
    for i in range(mps.n_sites):
        mps.apply_one_site(u1h, i)
    discarded = 0.0
    for i in range(0, mps.n_sites - 1, 2):
        discarded += mps.apply_two_site(u2h, i, max_bond, cutoff)
    for i in range(1, mps.n_sites - 1, 2):
        discarded += mps.apply_two_site(u2, i, max_bond, cutoff)
    for i in range(0, mps.n_sites - 1, 2):
        discarded += mps.apply_two_site(u2h, i, max_bond, cutoff)
    for i in range(mps.n_sites):
        mps.apply_one_site(u1h, i)
    mps.normalize()
    return discarded


@dataclass(frozen=True)
class TEBDEvent:
    times: Array
    probabilities: Array
    rates: Array
    delta_rate: Array
    crossing_time: float | None
    accumulated_discarded_weight: float


def evolve_event(n_sites: int, h_final: float, J: float = 1.0,
                 tmax: float = 1.0, dt: float = 0.01, max_bond: int = 96,
                 cutoff: float = 1e-11, progress: bool = True) -> TEBDEvent:
    mps = MPS.product(n_sites, branch=0)
    nsteps = int(np.ceil(tmax / dt))
    times = [0.0]
    probs = [[1.0, 0.0, 0.0]]
    disc = 0.0
    iterator: Iterable[int] = range(nsteps)
    if progress:
        iterator = tqdm(iterator, desc=f"TEBD N={n_sites}", unit="step")
    for step in iterator:
        hdt = min(dt, tmax - times[-1])
        if hdt <= 1e-15:
            break
        disc += tebd_step(mps, J, h_final, hdt, max_bond, cutoff)
        times.append(times[-1] + hdt)
        probs.append([abs(mps.product_amplitude(a)) ** 2 for a in range(3)])
    t = np.asarray(times)
    p = np.asarray(probs)
    floor = 1e-300
    r0 = -np.log(np.maximum(p[:, 0], floor)) / n_sites
    rp = -np.log(np.maximum(p[:, 1] + p[:, 2], floor)) / n_sites
    delta = r0 - rp
    tc = None
    for i in range(len(t) - 1):
        if delta[i] <= 0 < delta[i + 1]:
            tc = float(t[i] - delta[i] * (t[i + 1] - t[i]) / (delta[i + 1] - delta[i]))
            break
    return TEBDEvent(t, p, np.column_stack([r0, rp]), delta, tc, disc)


def evolve_exact_time(n_sites: int, h_final: float, time: float, J: float = 1.0,
                      dt: float = 0.01, max_bond: int = 96,
                      cutoff: float = 1e-11, progress: bool = False) -> MPS:
    mps = MPS.product(n_sites, branch=0)
    full = int(np.floor(time / dt + 1e-12))
    iterator: Iterable[int] = range(full)
    if progress:
        iterator = tqdm(iterator, desc=f"TEBD to t={time:.6f}", unit="step")
    for _ in iterator:
        tebd_step(mps, J, h_final, dt, max_bond, cutoff)
    remainder = time - full * dt
    if remainder > 1e-13:
        tebd_step(mps, J, h_final, remainder, max_bond, cutoff)
    return mps
