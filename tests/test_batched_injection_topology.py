import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from vcsel_lib import VCSEL


def _parameters(topology, rates):
    return dict(T=20.0, s=0.01, nbar=0.1, p=2.0, alpha=2.0,
                beta_n=0.0, beta_const=0.0, delta_p=np.zeros(4),
                phase_model="standard_lk", phi_p=np.zeros((4, 4)),
                kappa=np.zeros((4, 4)), coupling=0.0, self_feedback=0.0,
                injection=True, injection_topology=np.asarray(topology),
                injected_strength=1.3, injected_phase_diff=np.array([0.2]),
                injected_frequency=0.0, kappa_inj=np.asarray(rates), dt=1.0)


def test_batched_injection_masks_match_individual_cases():
    vcsel = VCSEL({})
    state = np.array([[0.2, 1.2, 0.1] * 4,
                      [0.3, 1.4, -0.2] * 4])
    masks = np.array([[1, 0, 0, 0], [0, 1, 1, 1]], dtype=bool)
    rates = np.array([[0.3], [0.1]])
    batched_nd = _parameters(masks, rates)
    batched = vcsel.f_nd(state, state, state, 0, batched_nd["phi_p"], batched_nd)

    for case in range(2):
        single_nd = _parameters(masks[case], rates[case:case + 1])
        single = vcsel.f_nd(state[case:case + 1], state[case:case + 1],
                            state[case:case + 1], 0, single_nd["phi_p"], single_nd)
        np.testing.assert_allclose(batched[case], single[0])

    for case in range(2):
        inactive = ~masks[case]
        no_injection_nd = _parameters(np.zeros(4, dtype=bool), rates[case:case + 1])
        baseline = vcsel.f_nd(state[case:case + 1], state[case:case + 1],
                              state[case:case + 1], 0,
                              no_injection_nd["phi_p"], no_injection_nd)
        np.testing.assert_allclose(batched[case, 1::3][inactive],
                                   baseline[0, 1::3][inactive])
        np.testing.assert_allclose(batched[case, 2::3][inactive],
                                   baseline[0, 2::3][inactive])


def test_batched_injection_mask_checks_case_count():
    vcsel = VCSEL({})
    state = np.ones((2, 12))
    nd = _parameters(np.ones((3, 4), dtype=bool), np.ones((2, 1)))
    with pytest.raises(ValueError, match="injection_topology"):
        vcsel.f_nd(state, state, state, 0, nd["phi_p"], nd)


def test_two_complex_sources_superpose_on_one_laser():
    vcsel = VCSEL({})
    state = np.array([[0.2, 1.2, 0.1] * 4])
    baseline_nd = _parameters(np.zeros(4, dtype=bool), np.zeros((1, 1)))
    baseline = vcsel.f_nd(state, state, state, 0,
                          baseline_nd["phi_p"], baseline_nd)

    source_a = 0.3 * np.sqrt(1.3) * np.exp(0.2j)
    source_b = 0.2 * np.sqrt(1.3) * np.exp(-0.4j)
    responses = []
    for source in (source_a, source_b, source_a + source_b):
        nd = _parameters(np.zeros(4, dtype=bool), np.zeros((1, 1)))
        nd["injection_forcing"] = np.array([[source], [0], [0], [0]])
        responses.append(vcsel.f_nd(state, state, state, 0,
                                    nd["phi_p"], nd) - baseline)

    np.testing.assert_allclose(responses[2], responses[0] + responses[1],
                               atol=1e-15)
    np.testing.assert_allclose(responses[2][0, :],
                               np.where(np.isin(np.arange(12), [1, 2]),
                                        responses[2][0, :], 0))

    legacy_nd = _parameters(np.array([1, 0, 0, 0], dtype=bool),
                            np.array([[0.3]]))
    legacy = vcsel.f_nd(state, state, state, 0,
                        legacy_nd["phi_p"], legacy_nd)
    np.testing.assert_allclose(baseline + responses[0], legacy)


def test_batched_complex_forcing_can_differ_by_case():
    vcsel = VCSEL({})
    state = np.array([[0.2, 1.2, 0.1] * 4,
                      [0.3, 1.4, -0.2] * 4])
    forcing = np.zeros((2, 4, 1), dtype=complex)
    forcing[0, 0, 0] = 0.3 + 0.1j
    forcing[1, 1, 0] = 0.2 - 0.1j
    nd = _parameters(np.zeros((2, 4), dtype=bool), np.zeros((2, 1)))
    nd["injection_forcing"] = forcing
    batched = vcsel.f_nd(state, state, state, 0, nd["phi_p"], nd)
    for case in range(2):
        single_nd = _parameters(np.zeros(4, dtype=bool), np.zeros((1, 1)))
        single_nd["injection_forcing"] = forcing[case]
        single = vcsel.f_nd(state[case:case + 1], state[case:case + 1],
                            state[case:case + 1], 0, single_nd["phi_p"], single_nd)
        np.testing.assert_allclose(batched[case], single[0])
