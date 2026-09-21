import numpy as np
from scipy.sparse.linalg import expm_multiply

from dqpt_wigner.hierarchy import block_support_diagnostic
from dqpt_wigner.potts_exact import ordered_state, potts_hamiltonian
from dqpt_wigner.tebd import MPS, evolve_exact_time


def test_mps_product_block_support():
    mps = MPS.product(6, branch=0)
    for ell in (1,2,3):
        d0 = mps.block_support(ell,0,batch_size=16)
        d1 = mps.block_support(ell,1,batch_size=16)
        assert np.isclose(d0.probability,1.0,atol=1e-12)
        assert d0.sign_cost < 1e-12
        assert d1.probability < 1e-12


def test_tebd_agrees_with_exact_short_open_chain():
    N=4; h=1.1; t=0.04; dt=0.01
    mps=evolve_exact_time(N,h,t,dt=dt,max_bond=81,cutoff=1e-14)
    H=potts_hamiltonian(N,h=h,periodic=False)
    psi=np.asarray(expm_multiply(-1j*t*H,ordered_state(N,0)))
    for a in range(3):
        p_mps=abs(mps.product_amplitude(a))**2
        idx=a*sum(3**k for k in range(N))
        assert np.isclose(p_mps,abs(psi[idx])**2,rtol=3e-4,atol=3e-7)
    dm=mps.block_support(2,0,batch_size=16)
    de=block_support_diagnostic(psi,N,2,0)
    assert np.isclose(dm.probability,de.probability,rtol=5e-4,atol=5e-7)
    assert np.isclose(dm.unsigned_weight,de.unsigned_weight,rtol=8e-4,atol=8e-7)
