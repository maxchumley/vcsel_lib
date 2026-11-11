"""Numerical regressions for one-correction trapezoidal delay integration."""

from types import MethodType

import numpy as np
import pytest

from vcsel_lib import VCSEL


def make_delay_model(dt, delay_steps, steps, interpolation, save_every):
    model = VCSEL.__new__(VCSEL)
    model.nd = dict(
        kappa=np.zeros((1, 1)),
        dt=dt,
        steps=steps,
        delay_steps=delay_steps,
        tau=delay_steps * dt,
        tau_p=1.0,
        noise_amplitude=None,
        phi_p=np.zeros((1, 1)),
        delay_interp=interpolation,
        save_every=save_every,
        store_freqs=False,
    )
    return model


@pytest.mark.parametrize("interpolation", [None, "linear", "cubic"])
@pytest.mark.parametrize("save_every", [1, 2])
def test_delayed_states_follow_each_rhs_stage(interpolation, save_every):
    """Both delays advance at the corrector without adding RHS evaluations."""
    model = make_delay_model(0.1, 4, 14, interpolation, save_every)
    calls = []

    def rhs(self, current, delayed, double_delayed, step, phase, nd=None):
        calls.append((step, delayed[0, 2], double_delayed[0, 2]))
        return np.zeros_like(current)

    model.f_nd = MethodType(rhs, model)
    history = np.zeros((1, 3, 8))
    history[:, 1] = 1.0
    history[:, 2] = np.arange(8)
    model.integrate(history, max_iter=1, smooth_freqs=False)

    # The first eight calls initialize the supplied history. Check the first
    # two integrated steps, whose delayed values are known history samples.
    np.testing.assert_allclose(
        calls[8:12],
        [(7, 3, 0), (8, 4, 0), (8, 4, 0), (9, 5, 1)],
        rtol=0,
        atol=1e-12,
    )
    assert len(calls) == 8 + 2 * (model.nd["steps"] - 8)


@pytest.mark.parametrize("interpolation", [None, "linear", "cubic"])
@pytest.mark.parametrize("save_every", [1, 2])
def test_one_correction_has_second_order_delay_convergence(interpolation, save_every):
    """For phi'(t) = e*phi(t-1), exact history gives phi(t) = exp(t)."""
    errors = []
    for dt in (0.1, 0.05, 0.025):
        delay_steps = round(1.0 / dt)
        model = make_delay_model(
            dt, delay_steps, round(6.0 / dt) + 1, interpolation, save_every
        )

        def rhs(self, current, delayed, double_delayed, step, phase, nd=None):
            derivative = np.zeros_like(current)
            derivative[:, 2] = np.e * delayed[:, 2]
            return derivative

        model.f_nd = MethodType(rhs, model)
        history_time = np.arange(2 * delay_steps) * dt
        history = np.zeros((1, 3, len(history_time)))
        history[:, 1] = 1.0
        history[:, 2] = np.exp(history_time)
        time, states, _ = model.integrate(
            history, max_iter=1, smooth_freqs=False
        )
        errors.append(abs(states[0, 2, -1] - np.exp(time[-1])))

    # Halving the step should reduce the error by approximately four. This
    # fails for the old histories, which produced approximately first order.
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all((orders > 1.9) & (orders < 2.1)), (errors, orders)
