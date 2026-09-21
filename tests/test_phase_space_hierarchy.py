import numpy as np

from dqpt_wigner.product_rotation import critical_time, simulate, state
from dqpt_wigner.hierarchy import (
    block_support_diagnostic,
    central_block_matrix,
    one_site_populations_from_wigner,
    qutrit_wigner,
)
from dqpt_wigner.potts_exact import ordered_state


def test_qutrit_wigner_normalization_and_marginal():
    rng = np.random.default_rng(14)
    psi = rng.normal(size=3) + 1j * rng.normal(size=3)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    W = qutrit_wigner(rho)
    assert np.isclose(W.sum(), 1.0, atol=1e-12)
    assert np.allclose(W.sum(axis=1), np.diag(rho).real, atol=1e-12)
    assert np.allclose(one_site_populations_from_wigner(rho), np.abs(psi) ** 2)


def test_precession_control_is_wigner_positive_at_crossing():
    result = simulate(n_sites=31, nt=301)
    assert np.isclose(result.critical_time, critical_time(), atol=1e-14)
    assert result.critical_negativity < 1e-12
    assert result.critical_wigner.min() > -1e-12
    p = np.abs(state(result.critical_time)) ** 2
    assert np.allclose(p, np.ones(3) / 3, atol=1e-12)


def test_full_block_support_matches_ordered_overlap_and_identity():
    rng = np.random.default_rng(7)
    N = 4
    psi = rng.normal(size=3**N) + 1j * rng.normal(size=3**N)
    psi /= np.linalg.norm(psi)
    for a in range(3):
        d = block_support_diagnostic(psi, N, N, a)
        idx = a * sum(3**k for k in range(N))
        assert np.isclose(d.probability, abs(psi[idx])**2, rtol=1e-10, atol=1e-12)
        assert np.isclose(d.unsigned_weight-d.probability, 2*d.negative_mass,
                          rtol=1e-10, atol=1e-12)
        assert np.isclose(d.support_rate+d.sign_cost,
                          -np.log(max(d.probability,1e-300))/N, atol=1e-10)


def test_one_site_endpoint_is_local_population():
    rng = np.random.default_rng(9)
    N = 5
    psi = rng.normal(size=3**N) + 1j * rng.normal(size=3**N)
    psi /= np.linalg.norm(psi)
    M = central_block_matrix(psi, N, 1)
    rho = M @ M.conj().T
    for a in range(3):
        d = block_support_diagnostic(psi, N, 1, a)
        assert np.isclose(d.probability, rho[a,a].real, atol=1e-11)


def test_product_state_has_zero_sign_cost_at_all_blocks():
    N = 5
    psi = ordered_state(N, 0)
    for ell in range(1, N+1):
        d = block_support_diagnostic(psi, N, ell, 0)
        assert np.isclose(d.probability, 1.0, atol=1e-12)
        assert d.negative_mass < 1e-12
        assert d.sign_cost < 1e-12


def test_clifford_covariance_of_return_conditioned_diagnostic():
    from dqpt_wigner.hierarchy import stabilizer_overlap_diagnostic
    rng=np.random.default_rng(123)
    psi=rng.normal(size=3)+1j*rng.normal(size=3); psi/=np.linalg.norm(psi)
    rho=np.outer(psi,psi.conj())
    ket0=np.array([1,0,0],dtype=complex); sigma=np.outer(ket0,ket0.conj())
    omega=np.exp(2j*np.pi/3)
    F=np.array([[omega**(j*k) for k in range(3)] for j in range(3)],dtype=complex)/np.sqrt(3)
    a=stabilizer_overlap_diagnostic(rho,sigma)
    b=stabilizer_overlap_diagnostic(F@rho@F.conj().T,F@sigma@F.conj().T)
    for key in ("probability","unsigned_weight","negative_mass","sign_cost"):
        assert np.isclose(a[key],b[key],rtol=1e-10,atol=1e-11)
