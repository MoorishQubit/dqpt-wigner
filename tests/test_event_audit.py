"""Non-tautological controls for common-event and independently contracted data."""
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from dqpt_wigner.event_audit import (direct_support,
    grouped_support, ising_decimal_probabilities, ising_sparse_probabilities,
    positive_block_return, streamed_wigner_line, support_quantities)
from dqpt_wigner.potts_mps import evolve_to_times, exact_potts_hamiltonian
from dqpt_wigner.collective_ising import collective_hamiltonian

ROOT = Path(__file__).resolve().parents[1]




def test_direct_projector_and_streamed_wigner_against_dense_state():
    size, length, tau = 4, 3, .49
    _, state, _ = next(evolve_to_times(size,[tau],1.,1.5,.002,81,1e-13))
    initial = np.zeros(3**size,complex)
    initial[0]=1
    exact = expm(-1j*exact_potts_hamiltonian(size,1.,1.5)*tau)@initial
    exact = exact.reshape([3]*size)
    for branch in range(3):
        dense_probability = np.sum(np.abs(exact[branch,branch,branch,:])**2)
        result = direct_support(state,0,length,branch,batch_size=5)
        assert result["probability"] == pytest.approx(dense_probability,abs=3e-6)
        assert abs(result["signed_reconstruction_error"]) < 1e-13
        assert abs(result["identity_A_minus_P_minus_2nu"]) < 1e-13
        direct_values = state.block_wigner_line(0,length,branch)/state.norm()
        streamed = np.concatenate(list(streamed_wigner_line(state,0,length,branch,5)))
        np.testing.assert_allclose(streamed,direct_values,atol=2e-14)
    assert direct_support(state,0,length,1)["negative_mass"] > 0.01


def test_grouped_events_require_separate_signed_and_unsigned_symmetry():
    a = support_quantities(.1,.3,.1,3)
    b = support_quantities(.1,.5,.2,3)
    grouped = grouped_support(a,b,3)
    assert grouped["physical_rate"] == pytest.approx(a["physical_rate"]-np.log(2)/3)
    assert abs(grouped["sign_cost"]-a["sign_cost"])>.05
    symmetric = grouped_support(a,a,3)
    assert symmetric["sign_cost"] == pytest.approx(a["sign_cost"])


def test_zero_primitive_is_explicit_and_never_probability_clamped():
    result=support_quantities(0,.2,.1,2)
    assert result["zero_probability"] and result["physical_rate"] is None
    assert result["sign_cost"] is None and result["surviving_fraction"] == 0
    with pytest.raises(ValueError):
        support_quantities(-1e-15,.2,.1,2)


def test_independent_decimal_and_sparse_ising_against_dense_exponential():
    n,tau,h=12,.684,.7
    initial=np.zeros(n+1,complex)
    initial[0]=1
    state=expm(-1j*collective_hamiltonian(n,1.,h)*tau)@initial
    decimal=ising_decimal_probabilities(n,1.,h,tau,50)
    sparse=ising_sparse_probabilities(n,1.,h,tau)
    expected=(abs(state[0])**2,abs(state[-1])**2)
    np.testing.assert_allclose(sparse,expected,rtol=2e-12,atol=1e-25)
    np.testing.assert_allclose([float(decimal["prob_up"]),float(decimal["prob_down"])],expected,rtol=2e-12,atol=1e-25)
    assert Decimal(decimal["prob_down"]) < Decimal(decimal["analytic_down_probability_upper_bound"])




def test_audited_events_when_generated():
    for path in (ROOT / "results/potts/events").glob("event_N*.json"):
        event=json.loads(path.read_text())
        for key in ("0","1","2"):
            branch=event["branches"][key]
            assert abs(branch["signed_reconstruction_error"])<1e-10
            assert abs(branch["identity_A_minus_P_minus_2nu"])<1e-10
            assert branch["probability"]<=branch["unsigned_weight"]+1e-10<=1+2e-10
        assert abs(event["delta_r"] if event["kind"]=="physical" else event["delta_s"])<1e-7
