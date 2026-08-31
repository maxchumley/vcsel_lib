#%%
"""Time-averaged far-field intensity for VCSEL/laser arrays."""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.constants import c, hbar
from vcsel_lib import VCSEL

try:
    from examples._paths import VISUALIZATION_RESULTS_DIR
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples._paths import VISUALIZATION_RESULTS_DIR


rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
plt.rc("font", family="serif")


def emitter_positions(n_emitters, d=None, x=None):
    """Return centered emitter positions in meters."""
    if x is not None:
        positions = np.asarray(x, dtype=float)
        if positions.ndim != 1 or len(positions) != n_emitters:
            raise ValueError(
                f"x must be a 1D array with {n_emitters} entries; "
                f"received shape {positions.shape}."
            )
        return positions

    if d is None:
        raise ValueError("Provide either emitter spacing d or positions x.")
    if d <= 0:
        raise ValueError("Emitter spacing d must be positive.")

    indices = np.arange(n_emitters, dtype=float)
    return (indices - 0.5 * (n_emitters - 1)) * float(d)


def angle_grid(theta_range_deg=(-90.0, 90.0), n_theta=2001):
    """Return an angular grid in degrees."""
    theta_values = np.asarray(theta_range_deg, dtype=float)
    if theta_values.ndim != 1:
        raise ValueError("theta_range_deg must be a 1D array or a (min, max) pair.")

    if theta_values.size == 2:
        theta_min, theta_max = theta_values
        if theta_max <= theta_min:
            raise ValueError("theta_range_deg must satisfy theta_max > theta_min.")
        return np.linspace(theta_min, theta_max, int(n_theta))

    if theta_values.size < 2:
        raise ValueError("An explicit theta grid must contain at least two values.")
    return theta_values


def evaluate_array_factor(steering, emitter_fields):
    """Evaluate the array factor without a BLAS-backed matrix multiply."""
    steering = np.asarray(steering, dtype=np.complex128)
    emitter_fields = np.asarray(emitter_fields, dtype=np.complex128)
    if steering.ndim != 2 or emitter_fields.ndim != 2:
        raise ValueError("steering and emitter_fields must both be two-dimensional.")
    if steering.shape[1] != emitter_fields.shape[0]:
        raise ValueError(
            "The steering emitter dimension must match the field emitter dimension."
        )
    if not np.all(np.isfinite(steering)) or not np.all(np.isfinite(emitter_fields)):
        raise ValueError("The array-factor inputs contain non-finite values.")

    # optimize=False uses NumPy's direct contraction path. This avoids rare
    # spurious overflow warnings observed from BLAS matmul with bounded complex
    # arrays in long notebook runs.
    return np.einsum(
        "ae,et->at",
        steering,
        emitter_fields,
        optimize=False,
    )


def far_field_intensity(
    S,
    phi,
    lambda0,
    d=None,
    x=None,
    theta_range_deg=(-90.0, 90.0),
    n_theta=2001,
    final_fraction=0.2,
    db_floor=-80.0,
    time_chunk_size=4096,
    element_fwhm_deg=None,
):
    """Calculate the time-averaged normalized far-field intensity.

    Parameters
    ----------
    S : ndarray, shape (N_emitters, Nt) or (N_cases, N_emitters, Nt)
        Photon-number or intensity-like time traces. Multiple noise cases are
        averaged when a three-dimensional array is supplied.
    phi : ndarray, shape (N_emitters, Nt) or (N_cases, N_emitters, Nt)
        Optical phase time traces in radians.
    lambda0 : float
        Free-space wavelength in meters.
    d : float, optional
        Uniform emitter spacing in meters.
    x : ndarray, optional
        Explicit emitter positions in meters. When supplied, x overrides d.
    theta_range_deg : tuple or ndarray
        Either (theta_min, theta_max) in degrees or an explicit angle grid.
    n_theta : int
        Number of angles when theta_range_deg is a two-value range.
    final_fraction : float
        Fraction of the end of the time trace used for averaging.
    db_floor : float or None
        Display floor in dB. Use None to retain -inf at exact zeros.
    time_chunk_size : int
        Number of time samples processed at once. Chunking keeps memory usage
        modest for long simulations and dense angular grids.
    element_fwhm_deg : float or None
        Full width at half maximum of a Gaussian single-emitter intensity
        pattern in degrees. If None, return the bare array factor.

    Returns
    -------
    theta_deg : ndarray
        Far-field angles in degrees.
    I_norm : ndarray
        Time-averaged intensity normalized to its maximum.
    I_dB : ndarray
        Normalized intensity in dB.
    """
    S = np.asarray(S, dtype=float)
    phi = np.asarray(phi, dtype=float)

    if S.ndim not in (2, 3) or phi.ndim != S.ndim:
        raise ValueError(
            "S and phi must both have shape (N_emitters, Nt) or "
            "(N_cases, N_emitters, Nt)."
        )
    if S.shape != phi.shape:
        raise ValueError(f"S and phi must have the same shape, got {S.shape} and {phi.shape}.")
    if min(S.shape) < 1:
        raise ValueError("S and phi must contain at least one emitter and one time point.")
    if lambda0 <= 0:
        raise ValueError("lambda0 must be positive.")
    if not 0.0 < final_fraction <= 1.0:
        raise ValueError("final_fraction must lie in the interval (0, 1].")
    if int(time_chunk_size) < 1:
        raise ValueError("time_chunk_size must be at least 1.")
    if element_fwhm_deg is not None and element_fwhm_deg <= 0.0:
        raise ValueError("element_fwhm_deg must be positive when supplied.")

    if S.ndim == 2:
        S = S[None, :, :]
        phi = phi[None, :, :]

    n_cases, n_emitters, n_times = S.shape
    positions = emitter_positions(n_emitters, d=d, x=x)
    theta_deg = angle_grid(theta_range_deg, n_theta=n_theta)
    theta_rad = np.deg2rad(theta_deg)

    analysis_points = max(1, int(np.ceil(final_fraction * n_times)))
    first_analysis_index = n_times - analysis_points
    S_analysis = S[:, :, first_analysis_index:]
    phi_analysis = phi[:, :, first_analysis_index:]

    if not np.all(np.isfinite(S_analysis)):
        invalid_count = int(np.size(S_analysis) - np.count_nonzero(np.isfinite(S_analysis)))
        raise ValueError(
            f"The final analysis window contains {invalid_count} non-finite S values. "
            "The VCSEL integration is numerically unstable; decrease dt, increase "
            "integrator iterations, or reduce the coupling/noise strength."
        )
    if not np.all(np.isfinite(phi_analysis)):
        invalid_count = int(np.size(phi_analysis) - np.count_nonzero(np.isfinite(phi_analysis)))
        raise ValueError(
            f"The final analysis window contains {invalid_count} non-finite phase values. "
            "The VCSEL integration is numerically unstable; decrease dt, increase "
            "integrator iterations, or reduce the coupling/noise strength."
        )

    # A common intensity scale cancels when I_average is normalized below.
    # Applying it before sqrt(S) prevents otherwise valid but very large
    # nondimensional photon numbers from overflowing the matrix product.
    intensity_scale = float(np.max(np.clip(S_analysis, 0.0, None)))
    if intensity_scale <= 0.0:
        raise ValueError("The final analysis window contains no positive intensity.")

    # The steering matrix has shape (N_theta, N_emitters). Matrix
    # multiplication evaluates the array factor for every angle and time.
    wave_number = 2.0 * np.pi / float(lambda0)
    steering = np.exp(
        1j
        * wave_number
        * np.outer(np.sin(theta_rad), positions)
    )

    # Accumulate intensity without retaining an N_theta x Nt array. This also
    # averages independent noise cases as intensities, not as complex fields.
    intensity_sum = np.zeros(theta_deg.size, dtype=float)
    sample_count = 0
    chunk_size = int(time_chunk_size)
    for case_index in range(n_cases):
        for start in range(first_analysis_index, n_times, chunk_size):
            stop = min(start + chunk_size, n_times)
            S_chunk = (
                np.clip(S[case_index, :, start:stop], 0.0, None)
                / intensity_scale
            )
            S_chunk = np.clip(S_chunk, 0.0, 1.0)
            phi_chunk = phi[case_index, :, start:stop]
            emitter_fields = np.sqrt(S_chunk) * np.exp(
                1j * np.remainder(phi_chunk, 2.0 * np.pi)
            )
            E_far_field = evaluate_array_factor(steering, emitter_fields)
            intensity_sum += np.sum(np.abs(E_far_field) ** 2, axis=1)
            sample_count += stop - start

    I_average = intensity_sum / sample_count

    # A Gaussian approximation to the single-VCSEL intensity envelope turns
    # the array factor into the observable far-field pattern. Its definition
    # gives I_element(+-FWHM/2) = 1/2.
    if element_fwhm_deg is not None:
        element_envelope = np.exp(
            -4.0
            * np.log(2.0)
            * (theta_deg / float(element_fwhm_deg)) ** 2
        )
        I_average *= element_envelope

    maximum = float(np.max(I_average))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("The far-field intensity maximum is zero or non-finite.")

    I_norm = I_average / maximum
    with np.errstate(divide="ignore"):
        I_dB = 10.0 * np.log10(I_norm)
    if db_floor is not None:
        I_dB = np.maximum(I_dB, float(db_floor))

    return theta_deg, I_norm, I_dB


def far_field_intensity_map(
    S,
    phi,
    lambda0,
    d=None,
    x=None,
    theta_range_deg=(-90.0, 90.0),
    n_theta=801,
    max_time_points=2000,
    time_chunk_size=256,
    element_fwhm_deg=None,
):
    """Calculate an angle-versus-time far-field intensity map.

    Intensities are averaged across noise realizations at each saved time.
    The complete map is then normalized by one global maximum so temporal
    intensity fluctuations remain visible.

    Returns
    -------
    theta_deg : ndarray
        Far-field angles in degrees.
    time_indices : ndarray
        Indices of the simulation samples represented in the map.
    I_map_norm : ndarray, shape (N_theta, N_map_times)
        Ensemble-averaged far-field intensity normalized globally.
    """
    S = np.asarray(S, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if S.ndim == 2:
        S = S[None, :, :]
        phi = phi[None, :, :]
    if S.ndim != 3 or phi.shape != S.shape:
        raise ValueError(
            "S and phi must have shape (N_emitters, Nt) or "
            "(N_cases, N_emitters, Nt)."
        )
    if lambda0 <= 0.0:
        raise ValueError("lambda0 must be positive.")
    if int(max_time_points) < 2:
        raise ValueError("max_time_points must be at least 2.")
    if int(time_chunk_size) < 1:
        raise ValueError("time_chunk_size must be at least 1.")
    if element_fwhm_deg is not None and element_fwhm_deg <= 0.0:
        raise ValueError("element_fwhm_deg must be positive when supplied.")

    n_cases, n_emitters, n_times = S.shape
    n_map_times = min(n_times, int(max_time_points))
    time_indices = np.unique(
        np.round(np.linspace(0, n_times - 1, n_map_times)).astype(int)
    )
    S_map = S[:, :, time_indices]
    phi_map = phi[:, :, time_indices]

    if not np.all(np.isfinite(S_map)) or not np.all(np.isfinite(phi_map)):
        raise ValueError(
            "The far-field map contains non-finite simulation values. "
            "Decrease dt or otherwise stabilize the VCSEL integration."
        )

    intensity_scale = float(np.max(np.clip(S_map, 0.0, None)))
    if intensity_scale <= 0.0:
        raise ValueError("The selected trajectory contains no positive intensity.")

    positions = emitter_positions(n_emitters, d=d, x=x)
    theta_deg = angle_grid(theta_range_deg, n_theta=n_theta)
    wave_number = 2.0 * np.pi / float(lambda0)
    steering = np.exp(
        1j
        * wave_number
        * np.outer(np.sin(np.deg2rad(theta_deg)), positions)
    )

    I_map = np.zeros((theta_deg.size, time_indices.size), dtype=float)
    chunk_size = int(time_chunk_size)
    for case_index in range(n_cases):
        for start in range(0, time_indices.size, chunk_size):
            stop = min(start + chunk_size, time_indices.size)
            S_chunk = (
                np.clip(S_map[case_index, :, start:stop], 0.0, None)
                / intensity_scale
            )
            S_chunk = np.clip(S_chunk, 0.0, 1.0)
            phi_chunk = phi_map[case_index, :, start:stop]
            emitter_fields = np.sqrt(S_chunk) * np.exp(
                1j * np.remainder(phi_chunk, 2.0 * np.pi)
            )
            E_far_field = evaluate_array_factor(steering, emitter_fields)
            I_map[:, start:stop] += np.abs(E_far_field) ** 2 / n_cases

    if element_fwhm_deg is not None:
        element_envelope = np.exp(
            -4.0
            * np.log(2.0)
            * (theta_deg / float(element_fwhm_deg)) ** 2
        )
        I_map *= element_envelope[:, None]

    maximum = float(np.max(I_map))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("The far-field intensity map maximum is zero or non-finite.")

    return theta_deg, time_indices, I_map / maximum


def plot_far_field_intensity(
    S,
    phi,
    lambda0,
    d=None,
    x=None,
    theta_range_deg=(-90.0, 90.0),
    n_theta=2001,
    final_fraction=0.2,
    db_floor=-80.0,
    time_chunk_size=4096,
    element_fwhm_deg=None,
    label=None,
    axes=None,
):
    """Calculate and plot linear and dB far-field intensity."""
    theta_deg, I_norm, I_dB = far_field_intensity(
        S=S,
        phi=phi,
        lambda0=lambda0,
        d=d,
        x=x,
        theta_range_deg=theta_range_deg,
        n_theta=n_theta,
        final_fraction=final_fraction,
        db_floor=db_floor,
        time_chunk_size=time_chunk_size,
        element_fwhm_deg=element_fwhm_deg,
    )

    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax_linear, ax_db = axes

    ax_linear.plot(theta_deg, I_norm, linewidth=2, label=label)
    ax_linear.set_ylabel("Normalized far-field intensity")
    ax_linear.set_ylim(bottom=0)
    ax_linear.grid(True, linestyle="--", alpha=0.4)

    ax_db.plot(theta_deg, I_dB, linewidth=2, label=label)
    ax_db.set_xlabel(r"Far-field angle $\theta$ (degrees)")
    ax_db.set_ylabel("Normalized intensity (dB)")
    if db_floor is not None:
        ax_db.set_ylim(float(db_floor), 1.0)
    ax_db.grid(True, linestyle="--", alpha=0.4)

    if label is not None:
        ax_linear.legend()
        ax_db.legend()

    return theta_deg, I_norm, I_dB


def extract_vcsel_fields(y, case_index=None):
    """Extract S and phi from a trajectory returned by VCSEL.integrate().

    Parameters
    ----------
    y : ndarray
        Shape (N_cases, 3*N_emitters, Nt) or (3*N_emitters, Nt).
    case_index : int or None
        Noise/case index used when y contains multiple cases. If None, return
        all cases so their far-field intensities can be averaged.
    """
    y = np.asarray(y)
    if y.ndim == 3:
        if case_index is None:
            if y.shape[1] % 3 != 0:
                raise ValueError("The VCSEL state dimension must be divisible by 3.")
            return y[:, 1::3, :], y[:, 2::3, :]
        if not 0 <= case_index < y.shape[0]:
            raise IndexError(f"case_index={case_index} is outside 0..{y.shape[0] - 1}.")
        y_case = y[case_index]
    elif y.ndim == 2:
        y_case = y
    else:
        raise ValueError(
            "y must have shape (N_cases, 3*N_emitters, Nt) "
            "or (3*N_emitters, Nt)."
        )

    if y_case.shape[0] % 3 != 0:
        raise ValueError("The VCSEL state dimension must be divisible by 3.")

    S = y_case[1::3]
    phi = y_case[2::3]
    return S, phi


def plot_far_field_from_vcsel(
    y,
    lambda0,
    case_index=None,
    **plot_kwargs,
):
    """Plot the far field directly from a vcsel_lib integration trajectory."""
    S, phi = extract_vcsel_fields(y, case_index=case_index)
    return plot_far_field_intensity(
        S=S,
        phi=phi,
        lambda0=lambda0,
        **plot_kwargs,
    )


def two_laser_example():
    """Compare ideal in-phase and out-of-phase two-laser far fields."""
    lambda0 = 980e-9
    d = 10e-6
    n_times = 100

    S = np.ones((2, n_times))
    phi_in_phase = np.zeros((2, n_times))
    phi_out_of_phase = np.vstack([
        np.zeros(n_times),
        np.full(n_times, np.pi),
    ])

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    plot_far_field_intensity(
        S,
        phi_in_phase,
        lambda0=lambda0,
        d=d,
        theta_range_deg=(-30.0, 30.0),
        final_fraction=1.0,
        label=r"In phase: $\Delta\phi=0$",
        axes=axes,
    )
    plot_far_field_intensity(
        S,
        phi_out_of_phase,
        lambda0=lambda0,
        d=d,
        theta_range_deg=(-30.0, 30.0),
        final_fraction=1.0,
        label=r"Out of phase: $\Delta\phi=\pi$",
        axes=axes,
    )
    fig.suptitle(
        r"Two-emitter far field: $\lambda_0=980\,\mathrm{nm}$, "
        r"$d=10\,\mu\mathrm{m}$"
    )
    fig.tight_layout()
    plt.show()


def simulated_two_vcsel_example():
    """Run a coupled two-VCSEL simulation and plot a four-panel summary.

    The model parameters and integration flow follow
    examples/basics/simple_example.py. The geometry and far-field
    controls are grouped here so the spacing or angular range can be changed
    without touching the calculation.
    """
    # ---------------- Far-field controls ----------------
    lambda0 = 910e-9
    emitter_spacing = 10e-6
    element_fwhm_deg = 20.0
    theta_range_deg = (-30.0, 30.0)
    far_field_map_n_theta = 801
    far_field_map_max_time_points = 2000

    # ---------------- VCSEL simulation controls ----------------
    alpha = 2.0
    tau_p = 5.4e-12
    tau_n = 0.25e-9
    g0 = 8.75e-4 * 1e9
    N0 = 2.86e5
    saturation = 4e-6
    q = 1.602e-19
    beta = 1e-3
    tau = 1e-9
    eta = 0.9
    current_threshold = 3.0

    n_lasers = 2
    n_iterations = 100
    detuning_ghz = 4.0
    kappa_final = 8e9
    noise_amplitude = 1.0
    dt = tau_p
    Tmax = 1.0e-6
    smooth_freqs = True

    current = (
        eta
        * current_threshold
        * q
        / tau_n
        * (N0 + 1.0 / (g0 * tau_p))
    )
    delta = detuning_ghz * 2.0 * np.pi * 1e9
    delta_dist = 0.0 * delta * np.linspace(-1.0, 1.0, n_lasers)
    omega0 = 2.0 * np.pi * c / lambda0

    steps = int(np.round(Tmax / dt)) + 1
    time_arr = np.arange(steps, dtype=float) * dt
    Tmax = time_arr[-1]
    delay_steps = int(round(tau / dt))
    adjacency = np.ones((n_lasers, n_lasers)) - np.eye(n_lasers)
    kappa_arr = VCSEL.build_coupling_matrix(
        time_arr=time_arr,
        kappa_initial=0.0,
        kappa_final=kappa_final,
        N_lasers=n_lasers,
        ramp_start=5.0,
        ramp_shape=200.0,
        tau=tau,
        scheme="CUSTOM",
        aMAT=adjacency,
    )

    physical_params = {
        "tau_p": tau_p,
        "tau_n": tau_n,
        "g0": g0,
        "N0": N0,
        "N_bar": N0 + 1.0 / (g0 * tau_p),
        "s": saturation,
        "beta": beta,
        "kappa_c_mat": kappa_arr,
        "phi_p_mat": np.zeros((n_iterations, n_lasers, n_lasers)),
        "I": current,
        "q": q,
        "alpha": alpha,
        "delta": delta_dist,
        "coupling": 1.0,
        "self_feedback": 0.0,
        "noise_amplitude": noise_amplitude,
        "dt": dt,
        "Tmax": Tmax,
        "tau": tau,
        "N_lasers": n_lasers,
        "save_every": 1,
    }

    vcsel = VCSEL(physical_params)
    nd = vcsel.scale_params()

    def noise_ramp(t):
        return VCSEL.cosine_ramp(
            np.array([t]),
            t_start=tau,
            rise_10_90=100.0 * tau,
            kappa_initial=0.0,
            kappa_final=noise_amplitude,
        )[0]

    nd["noise_amplitude"] = noise_ramp
    history, freq_hist, _, _ = vcsel.generate_history(
        nd,
        shape="FR",
        n_cases=n_iterations,
    )
    t, y, freqs = vcsel.integrate(
        history,
        nd=nd,
        progress=True,
        max_iter=5,
        smooth_freqs=smooth_freqs,
    )

    S_all, phi_all = extract_vcsel_fields(y, case_index=None)

    # Match simple_example.py: statistics are computed across all independent
    # noise trajectories before plotting.
    dphi_all = freqs * 1e-9 / (2.0 * np.pi * tau_p)
    save_every = int(max(1, nd.get("save_every", 1)))
    freq_hist_plot = (
        freq_hist[:, :, ::save_every]
        if save_every > 1
        else freq_hist
    )
    hist_len = min(freq_hist_plot.shape[2], dphi_all.shape[2])
    dphi_all[:, :, :hist_len] = freq_hist_plot[:, :, :hist_len]

    dphi_mean = np.mean(dphi_all, axis=0)
    dphi_std = np.std(dphi_all, axis=0)
    S_mean = np.mean(S_all, axis=0)
    S_std = np.std(S_all, axis=0)

    delta_phi_all = phi_all[:, None, :, :] - phi_all[:, :, None, :]
    cos_pd_all = np.cos(delta_phi_all)
    cos_pd_mean = np.mean(cos_pd_all, axis=0)
    cos_pd_std = np.std(cos_pd_all, axis=0)

    intensity_to_mW = (
        1e3 * hbar * omega0 / (g0 * tau_n * tau_p)
    )
    emitter_fields = np.sqrt(np.clip(S_all, 0.0, None)) * np.exp(
        1j * np.remainder(phi_all, 2.0 * np.pi)
    )
    total_power_cases_mW = (
        np.abs(np.sum(emitter_fields, axis=1)) ** 2
        * intensity_to_mW
    )
    total_power_mean_mW = np.mean(total_power_cases_mW, axis=0)
    total_power_std_mW = np.std(total_power_cases_mW, axis=0)

    theta_map_deg, far_field_time_indices, I_map_norm = far_field_intensity_map(
        S=S_all,
        phi=phi_all,
        lambda0=lambda0,
        d=emitter_spacing,
        theta_range_deg=theta_range_deg,
        n_theta=far_field_map_n_theta,
        max_time_points=far_field_map_max_time_points,
        element_fwhm_deg=element_fwhm_deg,
    )
    far_field_time_us = t[far_field_time_indices] * 1e6
    time_plot = t * 1e6

    # Coherent-combining gain relative to the summed individual intensities:
    # G=1 is incoherent and G=N is ideal in-phase combining for N equal fields.
    comparison_times_us = (0.0, 1.0)
    comparison_indices = [
        int(np.argmin(np.abs(time_plot - target_time_us)))
        for target_time_us in comparison_times_us
    ]
    for sample_index in comparison_indices:
        fields_at_time = (
            np.sqrt(np.clip(S_all[:, :, sample_index], 0.0, None))
            * np.exp(
                1j
                * np.remainder(
                    phi_all[:, :, sample_index],
                    2.0 * np.pi,
                )
            )
        )
        coherent_intensity = float(
            np.mean(np.abs(np.sum(fields_at_time, axis=1)) ** 2)
        )
        incoherent_intensity = float(
            np.mean(np.sum(S_all[:, :, sample_index], axis=1))
        )
        combining_gain = (
            coherent_intensity / incoherent_intensity
            if incoherent_intensity > 0.0
            else np.nan
        )
        print(
            f"Coherent-combining gain at t={time_plot[sample_index]:.6g} us: "
            f"G = {combining_gain:.6g} "
            f"(1=incoherent, {n_lasers}=ideal in-phase)"
        )

    # ---------------- Four-panel summary ----------------
    fig, axs = plt.subplots(4, 1, figsize=(14, 18), dpi=200)
    show_uncertainty = noise_amplitude > 0 and n_iterations > 1

    # 1) Phase derivatives
    for emitter_index in range(n_lasers):
        axs[0].plot(
            time_plot,
            dphi_mean[emitter_index],
            linewidth=2,
            label=rf"$\dot{{\phi}}_{emitter_index + 1}$",
        )
        if show_uncertainty:
            axs[0].fill_between(
                time_plot,
                dphi_mean[emitter_index] - dphi_std[emitter_index],
                dphi_mean[emitter_index] + dphi_std[emitter_index],
                alpha=0.3,
            )
    axs[0].set_xlabel(r"Time ($\mu$s)", fontsize=22)
    axs[0].set_ylabel(r"$\dot{\phi}$ (GHz)", fontsize=22)
    axs[0].set_title(
        rf"$\kappa_c$: $0 \rightarrow {kappa_final * 1e-9:.0f}$ "
        rf"ns$^{{-1}}$",
        fontsize=24,
        pad=20,
    )
    axs[0].grid(True, alpha=0.2)
    axs[0].axvspan(
        0.0,
        2.0 * delay_steps * dt * 1e6,
        color="gray",
        alpha=0.2,
    )
    axs[0].tick_params(axis="both", which="major", labelsize=18)
    if n_lasers <= 4:
        axs[0].legend(loc="upper right", fontsize=14)

    # 2) Cosine of phase differences
    for emitter_i, emitter_j in combinations(range(n_lasers), 2):
        axs[1].plot(
            time_plot,
            cos_pd_mean[emitter_i, emitter_j],
            linewidth=2,
            label=(
                rf"$\cos(\Delta\phi_{{{emitter_i + 1},"
                rf"{emitter_j + 1}}})$"
            ),
        )
        if show_uncertainty:
            axs[1].fill_between(
                time_plot,
                (
                    cos_pd_mean[emitter_i, emitter_j]
                    - cos_pd_std[emitter_i, emitter_j]
                ),
                (
                    cos_pd_mean[emitter_i, emitter_j]
                    + cos_pd_std[emitter_i, emitter_j]
                ),
                alpha=0.3,
            )
    axs[1].set_xlabel(r"Time ($\mu$s)", fontsize=22)
    axs[1].set_ylabel(r"$\cos(\Delta\phi)$", fontsize=22)
    axs[1].set_ylim(-1.1, 1.1)
    axs[1].grid(True, alpha=0.2)
    axs[1].axvspan(
        0.0,
        2.0 * delay_steps * dt * 1e6,
        color="gray",
        alpha=0.2,
    )
    axs[1].tick_params(axis="both", which="major", labelsize=18)
    if n_lasers <= 4:
        axs[1].legend(loc="lower right", fontsize=14)

    # 3) Individual and coherent total output powers
    for emitter_index in range(n_lasers):
        power_mean_mW = S_mean[emitter_index] * intensity_to_mW
        power_std_mW = S_std[emitter_index] * intensity_to_mW
        axs[2].plot(
            time_plot,
            power_mean_mW,
            linewidth=2,
            label=rf"$P_{emitter_index + 1}$",
        )
        if show_uncertainty:
            axs[2].fill_between(
                time_plot,
                power_mean_mW - power_std_mW,
                power_mean_mW + power_std_mW,
                alpha=0.3,
            )
    axs[2].plot(
        time_plot,
        total_power_mean_mW,
        color="g",
        linewidth=2,
        label=r"$P_{\rm tot}$",
    )
    if show_uncertainty:
        axs[2].fill_between(
            time_plot,
            total_power_mean_mW - total_power_std_mW,
            total_power_mean_mW + total_power_std_mW,
            color="g",
            alpha=0.3,
        )
    axs[2].set_xlabel(r"Time ($\mu$s)", fontsize=22)
    axs[2].set_ylabel("Output Power (mW)", fontsize=22)
    axs[2].grid(True, alpha=0.2)
    axs[2].axvspan(
        0.0,
        2.0 * delay_steps * dt * 1e6,
        color="gray",
        alpha=0.2,
    )
    axs[2].tick_params(axis="both", which="major", labelsize=18)
    if n_lasers <= 4:
        axs[2].legend(loc="upper right", fontsize=14)

    coupling_axis = axs[2].twinx()
    coupling_axis.set_ylabel(
        r"Coupling Power ($\mu$W)",
        color="black",
        fontsize=20,
    )
    coupling_axis.tick_params(
        axis="y",
        labelcolor="black",
        labelsize=18,
    )
    time_idx = np.round(t / (nd["dt"] * nd["tau_p"])).astype(int)
    time_idx = np.clip(time_idx, 0, kappa_arr.shape[0] - 1)
    kappa_plot = kappa_arr[time_idx, 0, 1]
    coupling_power_uW = (
        hbar
        * omega0
        * kappa_plot
        * nd["sbar"]
        / (g0 * tau_n)
        * 1e6
    )
    coupling_axis.plot(
        time_plot,
        coupling_power_uW,
        "k--",
        alpha=0.5,
        linewidth=2,
    )
    coupling_max = float(np.max(coupling_power_uW))
    if coupling_max > 0.0:
        coupling_axis.set_ylim(0.0, 1.5 * coupling_max)

    # 4) Time-resolved far field, ensemble-averaged at each time
    far_field_image = axs[3].pcolormesh(
        far_field_time_us,
        theta_map_deg,
        I_map_norm,
        shading="auto",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    axs[3].set_xlabel(r"Time ($\mu$s)", fontsize=22)
    axs[3].set_ylabel(
        r"Far-field angle $\theta$ (degrees)",
        fontsize=22,
    )
    axs[3].tick_params(
        axis="both",
        which="major",
        labelsize=18,
    )
    axs[3].set_title(
        rf"Far-field intensity averaged over {n_iterations} noise realizations; "
        rf"$d={emitter_spacing * 1e6:.1f}\,\mu$m, "
        rf"element FWHM $={element_fwhm_deg:.0f}^\circ$",
        fontsize=20,
    )
    # Reserve colorbar space globally so it does not shrink only the far-field
    # panel. The four primary axes therefore retain matching plot widths.
    fig.tight_layout(rect=(0.0, 0.0, 0.92, 1.0))
    map_position = axs[3].get_position()
    colorbar_axis = fig.add_axes(
        [
            map_position.x1 + 0.012,
            map_position.y0,
            0.014,
            map_position.height,
        ]
    )
    far_field_colorbar = fig.colorbar(
        far_field_image,
        cax=colorbar_axis,
    )
    far_field_colorbar.set_label(
        "Normalized far-field intensity",
        fontsize=20,
    )
    far_field_colorbar.ax.tick_params(labelsize=16)

    plot_dir = VISUALIZATION_RESULTS_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure_path = plot_dir / "far_field_intensity_summary.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved summary figure to {figure_path.resolve()}")
    return far_field_time_us, theta_map_deg, I_map_norm


if __name__ == "__main__":
    simulated_two_vcsel_example()
