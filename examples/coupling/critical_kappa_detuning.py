#%%
"""Vectorized detuning sweep with independent joblib jobs for coupling phase.

Parallel structure:
    joblib worker: one phi_p value
        vectorized cases: all detunings x all noise iterations
            continuation: serial sweep through kappa
"""

import gc
import multiprocessing as mp
import os
import queue as queue_module
import threading
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, parallel
from matplotlib import rc, texmanager
from tqdm.auto import tqdm

from vcsel_lib import VCSEL
try:
    from examples._paths import COUPLING_RESULTS_DIR
except ModuleNotFoundError:
    COUPLING_RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# User controls
# ---------------------------------------------------------------------------

# Grid
n_detuning = 100
detuning_ghz_vals = np.linspace(0.0, 6.0, n_detuning)
n_phi_p = 21
phi_p_loop_vals = np.linspace(0.0, 2.0 * np.pi, n_phi_p)
kappa_resolution = 100
kappa_c = np.linspace(0.0, 40.0e9, kappa_resolution)

# Parallelism and noise averaging
n_phi_jobs = min(n_phi_p, 10 or 1)
n_iterations = 50

# Shared integration controls
tau_p = 5.4e-12
dt = 1.0 * tau_p
Tmax = 5.0e-7
noise_amplitude = 1.0
integration_scheme = "trapezoid"
integrator_theta = 0.5
integrator_max_iter = 1
delay_interpolation = "linear"
noise_substeps = 1
trajectory_save_every = 8
trajectory_output_dtype = np.float32

# Coupling continuation
ramp_start_tau = 20.0
ramp_shape_tau = 20.0

# Output
save_progress_figures = True
figure_save_stride = 5
skip_existing_phi_runs = True
order_parameter_threshold = 0.9
show_integration_progress = True
integration_progress_update_steps = 100
leave_integration_progress_bars = False


# ---------------------------------------------------------------------------
# Physical model parameters
# ---------------------------------------------------------------------------

N_lasers = 2
alpha = 2.0
tau_n = 0.25e-9
g0 = 8.75e-4 * 1e9
N0 = 2.86e5
s = 4e-6
q = 1.602e-19
beta = 1e-3
tau = 1e-9
eta = 0.9
current_threshold = 3.0
self_feedback = 0.0
coupling = 1.0


@contextmanager
def tqdm_joblib(tqdm_object):
    """Route joblib task completions to one notebook-friendly progress bar."""

    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def set_latex_plot_style(tex_cache_dir=None):
    """Apply the plotting style used by the other VCSEL examples."""
    if tex_cache_dir is not None:
        tex_cache_dir = os.path.abspath(tex_cache_dir)
        os.makedirs(tex_cache_dir, exist_ok=True)
        texmanager.TexManager._texcache = tex_cache_dir
    rc("text", usetex=True)
    rc("font", family="serif")
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })


def output_directories():
    base = COUPLING_RESULTS_DIR / f"{N_lasers}_lasers/critical_kappa_detuning"
    return {
        "base": base,
        "arrays": f"{base}/numpy_arrays",
        "frames": f"{base}/spectrum_frames_detuning",
        "tex_cache": f"{base}/matplotlib_tex_cache",
    }


def phi_label(phi_idx, phi_value):
    return f"phiidx{int(phi_idx):03d}_phipi{float(phi_value) / np.pi:.3f}"


def time_grid():
    steps = int(np.round(float(Tmax) / float(dt))) + 1
    time_arr = np.arange(steps, dtype=float) * float(dt)
    return time_arr, float(time_arr[-1])


def build_detuning_cases():
    """Return per-detuning and flattened per-noise-case angular detunings."""
    delta_by_detuning = np.empty((n_detuning, N_lasers), dtype=float)
    for detuning_idx, detuning_ghz in enumerate(detuning_ghz_vals):
        total_delta = float(detuning_ghz) * 2.0 * np.pi * 1e9
        delta_by_detuning[detuning_idx] = np.linspace(
            -total_delta / 2.0,
            total_delta / 2.0,
            N_lasers,
        )
    delta_cases = np.repeat(delta_by_detuning, n_iterations, axis=0)
    return delta_by_detuning, delta_cases


def build_physical_parameters(phi_value, delta_cases, kappa_matrix, Tmax_actual):
    n_cases = len(delta_cases)
    drive_current = (
        eta
        * current_threshold
        * q
        / tau_n
        * (N0 + 1.0 / (g0 * tau_p))
    )
    return {
        "tau_p": tau_p,
        "tau_n": tau_n,
        "g0": g0,
        "N0": N0,
        "N_bar": N0 + 1.0 / (g0 * tau_p),
        "s": s,
        "beta": beta,
        "kappa_c_mat": kappa_matrix,
        "phi_p_mat": np.full(
            (n_cases, N_lasers, N_lasers),
            float(phi_value),
            dtype=float,
        ),
        "I": drive_current,
        "q": q,
        "alpha": alpha,
        "delta": delta_cases,
        "coupling": coupling,
        "self_feedback": self_feedback,
        "noise_amplitude": noise_amplitude,
        "dt": dt,
        "Tmax": Tmax_actual,
        "tau": tau,
        "N_lasers": N_lasers,
        "save_every": int(max(1, trajectory_save_every)),
        "output_dtype": np.dtype(trajectory_output_dtype).name,
        "max_output_gb": None,
    }


def apply_integrator_controls(nd):
    nd["integration_scheme"] = integration_scheme
    nd["delay_interp"] = delay_interpolation
    nd["noise_substeps"] = int(max(1, noise_substeps))
    nd["store_freqs"] = False
    return nd


def average_observables(vcsel, y, n_saved_times):
    """Average order and phase coherence over noise iterations."""
    steady_start = max(0, int(n_saved_times // 2))
    steady_y = y[:, :, steady_start:]

    order_cases = vcsel.order_parameter(steady_y)
    order_by_detuning = order_cases.reshape(n_detuning, n_iterations).mean(axis=1)

    phi = steady_y[:, 2::3, :]
    if N_lasers > 1:
        phase_diff = phi[:, 1:, :] - phi[:, :1, :]
        cos_cases = np.mean(np.cos(phase_diff), axis=(1, 2))
    else:
        cos_cases = np.ones(len(y), dtype=float)
    cos_by_detuning = cos_cases.reshape(n_detuning, n_iterations).mean(axis=1)

    return order_by_detuning, cos_by_detuning


def save_order_map(order_map, phi_value, frame_path):
    """Save the detuning-kappa order-parameter map."""
    set_latex_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=250)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("white")
    image = ax.imshow(
        np.ma.masked_invalid(order_map.T),
        origin="lower",
        aspect="auto",
        extent=[
            detuning_ghz_vals[0],
            detuning_ghz_vals[-1],
            kappa_c[0] * 1e-9,
            kappa_c[-1] * 1e-9,
        ],
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        rasterized=True,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Order Parameter", fontsize=20)
    colorbar.ax.tick_params(labelsize=18)
    ax.set_xlabel("Detuning (GHz)", fontsize=22)
    ax.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=22)
    ax.set_title(
        rf"$\phi_p={float(phi_value) / np.pi:.2f}\pi$",
        fontsize=24,
        pad=14,
    )
    ax.tick_params(axis="both", labelsize=18)
    ax.set_yticks(np.linspace(kappa_c[0] * 1e-9, kappa_c[-1] * 1e-9, 6))
    fig.savefig(frame_path, bbox_inches="tight")
    plt.close(fig)


def critical_kappa_from_order(order_map):
    """First kappa where the order parameter crosses the chosen threshold."""
    critical = np.full(n_detuning, np.nan, dtype=float)
    for detuning_idx in range(n_detuning):
        crossings = np.flatnonzero(
            np.asarray(order_map[detuning_idx]) >= float(order_parameter_threshold)
        )
        if crossings.size:
            critical[detuning_idx] = kappa_c[crossings[0]] * 1e-9
    return critical


def run_phi_continuation(phi_idx, phi_value, progress_queue=None):
    """Run all detunings together and continue serially through kappa."""
    phi_value = float(phi_value)
    label = phi_label(phi_idx, phi_value)
    dirs = output_directories()
    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)
    set_latex_plot_style(f"{dirs['tex_cache']}/{label}")

    result_path = f"{dirs['arrays']}/vectorized_{label}.npz"
    frame_path = f"{dirs['frames']}/order_parameter_map_{label}.png"
    if skip_existing_phi_runs and os.path.exists(result_path):
        if progress_queue is not None:
            progress_queue.put(("complete", int(phi_idx), None))
        return {
            "phi_idx": int(phi_idx),
            "phi_p": phi_value,
            "result_path": result_path,
            "skipped": True,
        }

    time_arr, Tmax_actual = time_grid()
    delta_by_detuning, delta_cases = build_detuning_cases()
    n_cases = n_detuning * n_iterations
    adjacency = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)

    initial_kappa_matrix = VCSEL.build_coupling_matrix(
        time_arr=time_arr,
        kappa_initial=float(kappa_c[0]),
        kappa_final=float(kappa_c[0]),
        N_lasers=N_lasers,
        ramp_start=ramp_start_tau,
        ramp_shape=ramp_shape_tau,
        tau=tau,
        scheme="CUSTOM",
        plot=False,
        dx=1.0,
        aMAT=adjacency,
    )
    phys = build_physical_parameters(
        phi_value,
        delta_cases,
        initial_kappa_matrix,
        Tmax_actual,
    )
    vcsel = VCSEL(phys)
    nd = apply_integrator_controls(vcsel.scale_params())
    history, _, _, _ = vcsel.generate_history(nd, shape="FR", n_cases=n_cases)

    order_map = np.full((n_detuning, len(kappa_c)), np.nan, dtype=np.float32)
    cosine_map = np.full_like(order_map, np.nan)
    previous_kappa = float(kappa_c[0])
    pending_progress_steps = 0

    def report_integration_progress(n_steps):
        nonlocal pending_progress_steps
        if progress_queue is None:
            return
        pending_progress_steps += int(n_steps)
        if pending_progress_steps >= int(max(1, integration_progress_update_steps)):
            progress_queue.put(("update", int(phi_idx), pending_progress_steps))
            pending_progress_steps = 0

    def flush_integration_progress():
        nonlocal pending_progress_steps
        if progress_queue is not None and pending_progress_steps > 0:
            progress_queue.put(("update", int(phi_idx), pending_progress_steps))
            pending_progress_steps = 0

    show_inner_bar = int(max(1, n_phi_jobs)) == 1
    kappa_iterator = tqdm(
        enumerate(kappa_c),
        total=len(kappa_c),
        desc=f"phi {phi_idx + 1}/{n_phi_p}",
        unit="kappa",
        leave=False,
        disable=not show_inner_bar,
    )

    for kappa_idx, kappa_value in kappa_iterator:
        kappa_value = float(kappa_value)
        if progress_queue is not None:
            progress_queue.put(
                (
                    "status",
                    int(phi_idx),
                    f"kappa {kappa_idx + 1}/{len(kappa_c)} "
                    f"({kappa_value * 1e-9:.2f} ns^-1)",
                )
            )
        kappa_matrix = VCSEL.build_coupling_matrix(
            time_arr=time_arr,
            kappa_initial=previous_kappa,
            kappa_final=kappa_value,
            N_lasers=N_lasers,
            ramp_start=ramp_start_tau,
            ramp_shape=ramp_shape_tau,
            tau=tau,
            scheme="CUSTOM",
            plot=False,
            dx=1.0,
            aMAT=adjacency,
        )
        phys["kappa_c_mat"] = kappa_matrix
        vcsel = VCSEL(phys)
        nd = apply_integrator_controls(vcsel.scale_params())
        t, y, _, final_history = vcsel.integrate(
            history,
            nd=nd,
            progress=False,
            theta=integrator_theta,
            max_iter=integrator_max_iter,
            smooth_freqs=False,
            integration_scheme=integration_scheme,
            return_final_history=True,
            progress_callback=report_integration_progress,
        )
        # The returned final history is the only state needed by the next
        # kappa step. Release the previous history before analyzing outputs.
        del history
        flush_integration_progress()
        if not np.all(np.isfinite(y)):
            raise FloatingPointError(
                f"Non-finite state for phi index {phi_idx}, "
                f"kappa={kappa_value * 1e-9:.3f} ns^-1."
            )

        order_values, cosine_values = average_observables(vcsel, y, y.shape[-1])
        order_map[:, kappa_idx] = order_values
        cosine_map[:, kappa_idx] = cosine_values
        history = final_history
        del final_history
        previous_kappa = kappa_value

        if (
            save_progress_figures
            and (
                (kappa_idx + 1) % int(max(1, figure_save_stride)) == 0
                or kappa_idx == len(kappa_c) - 1
            )
        ):
            save_order_map(order_map, phi_value, frame_path)

        # Nothing else from this integration is needed after the observables
        # and continuation history have been extracted.
        phys["kappa_c_mat"] = None
        del y, t, order_values, cosine_values
        del vcsel, nd, kappa_matrix
        if kappa_idx % 5 == 0:
            gc.collect()

    critical_kappa_ns = critical_kappa_from_order(order_map)
    np.savez(
        result_path,
        order_parameter=order_map,
        cosine_phase_difference=cosine_map,
        critical_kappa_ns=critical_kappa_ns,
        detuning_ghz=detuning_ghz_vals,
        delta_distribution_rad_s=delta_by_detuning,
        kappa_c=kappa_c,
        phi_p=phi_value,
        dt=dt,
        Tmax=Tmax_actual,
        noise_amplitude=noise_amplitude,
        n_iterations=n_iterations,
    )
    save_order_map(order_map, phi_value, frame_path)
    del history, phys, delta_cases, delta_by_detuning
    del adjacency, time_arr, order_map, cosine_map, critical_kappa_ns
    gc.collect()
    if progress_queue is not None:
        progress_queue.put(("complete", int(phi_idx), None))

    return {
        "phi_idx": int(phi_idx),
        "phi_p": phi_value,
        "result_path": result_path,
        "skipped": False,
    }


def run_all_phi_values():
    """Launch one joblib process per phi_p value."""
    jobs = [
        (phi_idx, float(phi_value))
        for phi_idx, phi_value in enumerate(phi_p_loop_vals)
    ]
    n_jobs = min(int(max(1, n_phi_jobs)), len(jobs))
    progress_manager = None
    progress_queue = None
    progress_thread = None
    stop_progress = threading.Event()
    integration_bars = {}

    integration_steps_per_kappa = max(
        1,
        int(Tmax / dt) - 2 * int(tau / dt),
    )
    integration_steps_per_phi = integration_steps_per_kappa * len(kappa_c)

    if show_integration_progress:
        progress_manager = mp.get_context("fork").Manager()
        progress_queue = progress_manager.Queue()
        for phi_idx, phi_value in jobs:
            integration_bars[phi_idx] = tqdm(
                total=integration_steps_per_phi,
                desc=f"phi {phi_idx + 1}/{n_phi_p}",
                unit="step",
                position=phi_idx + 1,
                leave=leave_integration_progress_bars,
                dynamic_ncols=True,
            )

        def handle_progress_message(message):
            message_type, phi_idx, payload = message
            bar = integration_bars.get(int(phi_idx))
            if bar is None:
                return
            if message_type == "update":
                remaining = max(0, int(bar.total - bar.n))
                bar.update(min(int(payload), remaining))
            elif message_type == "status":
                bar.set_postfix_str(str(payload), refresh=False)
            elif message_type == "complete":
                if bar.n < bar.total:
                    bar.update(bar.total - bar.n)
                bar.refresh()

        def monitor_progress():
            while not stop_progress.is_set():
                try:
                    message = progress_queue.get(timeout=0.1)
                except queue_module.Empty:
                    continue
                handle_progress_message(message)

        progress_thread = threading.Thread(target=monitor_progress, daemon=True)
        progress_thread.start()

    outer_bar = tqdm(
        total=len(jobs),
        desc="phi_p sweep",
        unit="phi",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )
    try:
        with tqdm_joblib(outer_bar):
            results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                delayed(run_phi_continuation)(
                    phi_idx,
                    phi_value,
                    progress_queue,
                )
                for phi_idx, phi_value in jobs
            )
    finally:
        stop_progress.set()
        if progress_thread is not None:
            progress_thread.join(timeout=2.0)
        if progress_queue is not None:
            while True:
                try:
                    handle_progress_message(progress_queue.get_nowait())
                except queue_module.Empty:
                    break
        for bar in integration_bars.values():
            bar.close()
        if progress_manager is not None:
            progress_manager.shutdown()
    return results


set_latex_plot_style()
phi_sweep_results = run_all_phi_values()
