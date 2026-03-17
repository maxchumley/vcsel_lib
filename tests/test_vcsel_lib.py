import os
import sys

import numpy as np

# Ensure local vcsel_lib.py is used instead of any installed package.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vcsel_lib import VCSEL


def make_phys(N_lasers=2, steps=20):
    tau_p = 5.4e-12
    tau_n = 0.25e-9
    g0 = 8.75e-4 * 1e9
    N0 = 2.86e5
    s = 4e-6
    q = 1.602e-19
    beta = 1.0e-3
    alpha = 2.0
    eta = 0.9
    current_threshold = 3.0
    I = eta * current_threshold * q / tau_n * (N0 + 1.0 / (g0 * tau_p))

    dt = 0.1 * tau_p
    Tmax = steps * dt
    tau = 1.0e-9
    time_arr = np.linspace(0.0, Tmax, steps)

    detuning = 0.1  # GHz
    delta = detuning * 2 * np.pi * 1e9
    delta_dist = delta / 2 * np.linspace(-1, 1, N_lasers)

    kappa_c = 0.5e9
    ramp_start = 0
    ramp_shape = 0.001
    kappa_arr = VCSEL.build_coupling_matrix(
        time_arr=time_arr,
        kappa_initial=0.0,
        kappa_final=kappa_c,
        N_lasers=N_lasers,
        ramp_start=ramp_start,
        ramp_shape=ramp_shape,
        tau=tau,
        scheme="ATA",
    )

    phi_p_vals = np.array([0.0])
    phi_p_mat = np.ones((1, N_lasers, N_lasers)) * phi_p_vals[:, None, None]

    phys = {
        "tau_p": tau_p,
        "tau_n": tau_n,
        "g0": g0,
        "N0": N0,
        "s": s,
        "beta": beta,
        "kappa_c_mat": kappa_arr,
        "phi_p_mat": phi_p_mat,
        "I": I,
        "q": q,
        "alpha": alpha,
        "delta": delta_dist,
        "coupling": 1.0,
        "self_feedback": 0.0,
        "noise_amplitude": 1.0,
        "dt": dt,
        "Tmax": Tmax,
        "tau": tau,
        "N_lasers": N_lasers,
        "time_arr": time_arr,
    }
    return phys


def test_cosine_ramp_endpoints():
    t_start = 1.0
    rise = 2.0
    total_duration = rise / 0.8
    t = np.array([t_start, t_start + total_duration])
    out = VCSEL.cosine_ramp(t, t_start, rise, kappa_initial=2.0, kappa_final=5.0)
    assert np.allclose(out[0], 2.0)
    assert np.allclose(out[1], 5.0)


def test_scale_params_shapes():
    phys = make_phys(N_lasers=3, steps=10)
    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    assert nd["kappa"].shape == (10, 3, 3)
    assert nd["phi_p"].shape == (1, 3, 3)
    assert nd["delta_p"].shape == (3,)
    assert nd["steps"] == 10
    assert nd["delay_steps"] == int(phys["tau"] / phys["dt"])


def test_generate_history_shapes():
    phys = make_phys(N_lasers=2, steps=12)
    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    history, freq_hist, eq, results = vcsel.generate_history(nd, shape="FR", n_cases=2)
    assert history.shape == (2, 3 * nd["N_lasers"], 2 * nd["delay_steps"])
    assert freq_hist.shape == (2, nd["N_lasers"], 2 * nd["delay_steps"])
    assert eq is None

    history, freq_hist, eq, results = vcsel.generate_history(nd, shape="ZF", n_cases=3)
    assert history.shape == (3, 3 * nd["N_lasers"], 2 * nd["delay_steps"])
    assert freq_hist.shape == (3, nd["N_lasers"], 2 * nd["delay_steps"])
    assert eq is None

    nd_eq = dict(nd)
    nd_eq["kappa"] = nd["kappa"][-1]
    nd_eq["phi_p"] = nd["phi_p"][0]
    history, freq_hist, eq, results = vcsel.generate_history(nd_eq, shape="EQ", n_cases=1)
    
    assert history.shape[0] == 2
    assert history.shape[1] == 3 * nd["N_lasers"]
    assert history.shape[2] == 2 * nd["delay_steps"]
    assert freq_hist.shape[1:] == (nd["N_lasers"], 2 * nd["delay_steps"])
    assert results is None or results.shape[1] == 3 * nd["N_lasers"]


def test_invert_scaling_vectorized():
    phys = make_phys(N_lasers=3, steps=8)
    vcsel = VCSEL(phys)
    vcsel.scale_params()

    n_cases = 2
    steps = 8
    N_lasers = phys["N_lasers"]
    y = np.random.randn(n_cases, 3 * N_lasers, steps)
    y_copy = y.copy()
    y_phys = vcsel.invert_scaling(y, phys)

    g0 = phys["g0"]
    tau_p = phys["tau_p"]
    tau_n = phys["tau_n"]
    N0 = phys["N0"]

    n_idx = slice(0, None, 3)
    s_idx = slice(1, None, 3)
    assert np.allclose(
        y_phys[:, n_idx, :],
        y_copy[:, n_idx, :] / (g0 * tau_p) + N0 + 1.0 / (g0 * tau_p),
    )
    assert np.allclose(
        y_phys[:, s_idx, :],
        y_copy[:, s_idx, :] / (g0 * tau_n),
    )


def test_compute_noise_sample_zero():
    phys = make_phys(N_lasers=2, steps=6)
    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    y = np.zeros((1, 6))
    noise = vcsel.compute_noise_sample(y, noise_amplitude=0.0, dt=nd["dt"], nd=nd)
    assert np.allclose(noise, 0.0)


def test_f_nd_shapes_and_finite():
    phys = make_phys(N_lasers=2, steps=6)
    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()

    n_cases = 3
    N_lasers = phys["N_lasers"]
    x = np.random.randn(n_cases, 3 * N_lasers)
    x_tau = np.random.randn(n_cases, 3 * N_lasers)
    x_2tau = np.random.randn(n_cases, 3 * N_lasers)
    out = vcsel.f_nd(x, x_tau, x_2tau, j=0, phi_p=0.0, nd=nd)
    assert out.shape == (n_cases, 3 * N_lasers)
    assert np.isfinite(out).all()


def test_integrate_short_run():
    phys = make_phys(N_lasers=2, steps=20)
    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    # Speed up: shrink delay window for this test.
    nd["tau"] = nd["dt"]
    nd["delay_steps"] = 1
    history, _, _, _ = vcsel.generate_history(nd, shape="FR", n_cases=1)
    t_dim, y, freqs = vcsel.integrate(history, nd=nd, progress=False, max_iter=1)
    assert t_dim.shape[0] == nd["steps"]
    assert y.shape == (1, 3 * nd["N_lasers"], nd["steps"])
    assert freqs.shape == (1, nd["N_lasers"], nd["steps"])


def test_solve_equilibria_runs():
    phys = make_phys(N_lasers=2, steps=10)
    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    nd["kappa"] = nd["kappa"][-1]
    nd["phi_p"] = nd["phi_p"][0]
    final_root, results, E_tot = vcsel.solve_equilibria(
        nd=nd, counts={"phase_count": 2, "freq_count": 2}, n_jobs=1
    )
    assert results is None or isinstance(results, (list, np.ndarray))
    if isinstance(results, np.ndarray):
        assert results.ndim == 2
        assert results.shape[1] == 3 * nd["N_lasers"]
    assert (final_root is None) or (final_root.shape[0] == 3 * nd["N_lasers"])
    assert (E_tot is None) or isinstance(E_tot, np.ndarray)


def test_residuals_shape_and_finite():
    phys = make_phys(N_lasers=2, steps=10)
    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    nd["kappa"] = nd["kappa"][-1]
    nd["phi_p"] = nd["phi_p"][0]
    x = np.array([nd["sbar"], nd["sbar"], 0.0, 0.0])
    res = vcsel.residuals(x, nd=nd)
    assert res.shape == (nd["N_lasers"] + (nd["N_lasers"] - 1) + 1,)
    assert np.isfinite(res).all()


def test_order_parameter_unity():
    n_cases = 1
    N_lasers = 3
    steps = 5
    y = np.zeros((n_cases, 3 * N_lasers, steps))
    y[:, 1::3, :] = 1.0
    y[:, 2::3, :] = 0.0
    vcsel = VCSEL(make_phys(N_lasers=N_lasers, steps=steps))
    r = vcsel.order_parameter(y)
    assert np.allclose(r, 1.0)


def test_build_coupling_matrix_all_schemes():
    N_lasers = 3
    steps = 6
    time_arr = np.linspace(0.0, 1.0e-9, steps)
    kappa_initial = 0.0
    kappa_final = 1.0
    ramp_start = 1
    ramp_shape = 2
    tau = 1.0e-12
    aMAT = np.ones((N_lasers, N_lasers))

    for scheme in ["ATA", "NN", "CUSTOM", "RANDOM", "DECAYED"]:
        kwargs = {}
        if scheme == "CUSTOM":
            kwargs["aMAT"] = aMAT
        if scheme == "DECAYED":
            kwargs["dx"] = 0.5
        kappa_mat = VCSEL.build_coupling_matrix(
            time_arr=time_arr,
            kappa_initial=kappa_initial,
            kappa_final=kappa_final,
            N_lasers=N_lasers,
            ramp_start=ramp_start,
            ramp_shape=ramp_shape,
            tau=tau,
            scheme=scheme,
            **kwargs,
        )
        assert kappa_mat.shape == (steps, N_lasers, N_lasers)
        assert np.isfinite(kappa_mat).all()


def test_compute_jacobians_and_spectrum_and_stability():
    phys = make_phys(N_lasers=2, steps=10)
    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    nd["kappa"] = nd["kappa"][-1]
    nd["phi_p"] = nd["phi_p"][0]

    x_eq = np.zeros(3 * nd["N_lasers"])
    x_eq[0::2][: nd["N_lasers"]] = nd["nbar"]
    x_eq[1::2][: nd["N_lasers"]] = nd["sbar"]
    x_eq[-1] = 0.0

    A0, A1, A2 = vcsel.compute_jacobians(x_eq, nd)
    assert A0.shape == (3 * nd["N_lasers"], 3 * nd["N_lasers"])
    assert A1.shape == A0.shape
    assert A2.shape == A0.shape
    assert np.isfinite(A0).all()

    A_list = [A0, A1]
    tau_list = [0.0, nd["tau"]]
    eigs = vcsel.compute_spectrum(A_list, tau_list, N=3, stable_margin=None)
    assert isinstance(eigs, np.ndarray)

    stable, eigvals = vcsel.compute_stability(x_eq, nd, N=3)
    assert stable in (0.0, 1.0)
    assert isinstance(eigvals, np.ndarray)
