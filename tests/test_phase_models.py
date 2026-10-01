import os
import sys

import numpy as np
import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vcsel_lib import VCSEL


def _nd(phase_model="chumley_2026"):
    return {
        "T": 20.0,
        "s": 0.3,
        "nbar": 0.05,
        "p": 2.0,
        "alpha": 2.0,
        "beta_n": 0.0,
        "beta_const": 0.0,
        "delta_p": np.zeros(2),
        "phase_model": phase_model,
        "phi_p": np.zeros((2, 2)),
        "kappa": np.zeros((2, 2)),
        "coupling": 0.0,
        "self_feedback": 0.0,
        "tau": 0.0,
        "N_lasers": 2,
    }


@pytest.mark.parametrize("phase_model", ["ma_2019", "standard_lk", "chumley_2026"])
def test_phase_model_formulas_and_jacobian(phase_model):
    vcsel = VCSEL({})
    n = np.array([0.2, -0.1])
    S = np.array([1.3, 0.7])
    s = 0.3
    nbar = 0.05
    alpha = 2.0
    denom = 1.0 + s * S

    drift, d_drift_dn, d_drift_dS = vcsel._intrinsic_phase_gain(
        n, S, s, nbar, alpha, phase_model
    )
    if phase_model == "ma_2019":
        expected_drift = n - nbar
        expected_dn = np.ones_like(n)
        expected_dS = np.zeros_like(S)
    elif phase_model == "standard_lk":
        expected_drift = (1.0 + n) / denom - 1.0
        expected_dn = 1.0 / denom
        expected_dS = -s * (1.0 + n) / denom**2
    else:
        expected_drift = (n - nbar) / denom
        expected_dn = 1.0 / denom
        expected_dS = -s * (n - nbar) / denom**2

    assert np.allclose(drift, expected_drift)
    assert np.allclose(d_drift_dn, expected_dn)
    assert np.allclose(d_drift_dS, expected_dS)

    # compute_jacobians receives [n1, S1, n2, S2, phi2, omega].
    A0, _, _ = vcsel.compute_jacobians(
        np.array([n[0], S[0], n[1], S[1], 0.0, 0.0]),
        _nd(phase_model),
    )
    assert np.allclose(A0[2, [0, 1]], [expected_dn[0], expected_dS[0]])
    assert np.allclose(A0[5, [3, 4]], [expected_dn[1], expected_dS[1]])


@pytest.mark.parametrize("phase_model", ["ma_2019", "standard_lk", "chumley_2026"])
def test_phase_model_is_used_by_dynamics_and_equilibrium_residual(phase_model):
    vcsel = VCSEL({})
    nd = _nd(phase_model)

    # f_nd state order is [n1, S1, phi1, n2, S2, phi2].
    state = np.array([[0.2, 1.3, 0.0, -0.1, 0.7, 0.0]])
    derivative = vcsel.f_nd(state, state, state, 0, nd["phi_p"], nd)
    expected_drift, _, _ = vcsel._intrinsic_phase_gain(
        np.array([0.2, -0.1]), np.array([1.3, 0.7]),
        nd["s"], nd["nbar"], nd["alpha"], phase_model,
    )
    assert np.allclose(derivative[0, [2, 5]], expected_drift)

    root_state = np.array([1.3, 0.7, 0.0, 0.0])
    residual = vcsel.residuals(root_state, nd)
    S = root_state[:2]
    n = (1.0 + S / (1.0 + nd["s"] * S)) ** -1 * (
        nd["p"] - S / (1.0 + nd["s"] * S)
    )
    residual_drift, _, _ = vcsel._intrinsic_phase_gain(
        n, S, nd["s"], nd["nbar"], nd["alpha"], phase_model
    )
    assert np.isclose(residual[2], residual_drift[1] - residual_drift[0])


def test_missing_phase_model_preserves_chumley_2026_behavior():
    vcsel = VCSEL({})
    legacy_nd = _nd("chumley_2026")
    default_nd = dict(legacy_nd)
    default_nd.pop("phase_model")
    state = np.array([[0.2, 1.3, 0.0, -0.1, 0.7, 0.0]])

    legacy = vcsel.f_nd(state, state, state, 0, legacy_nd["phi_p"], legacy_nd)
    default = vcsel.f_nd(state, state, state, 0, default_nd["phi_p"], default_nd)
    assert np.allclose(default, legacy)


def test_invalid_phase_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown phase_model"):
        VCSEL._intrinsic_phase_gain(0.0, 1.0, 0.0, 0.0, 2.0, "not_a_model")
