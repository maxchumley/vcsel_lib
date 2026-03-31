#%%

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager
from joblib import parallel
from pathlib import Path
from IPython.display import clear_output

from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from vcsel_lib import VCSEL
from scipy.interpolate import make_interp_spline

plt.ion()

# --- Physical parameters ---
alpha = 2
tau_p = 5.4e-12
tau_n = 0.25e-9
g0 = 8.75e-4 * 1e9
N0 = 2.86e5
s = 4e-6
q = 1.602e-19
beta = 1.0e-3
tau = 1e-9  # delay (s)
eta = 0.9
current_threshold = 3

I = eta * current_threshold * q / tau_n * (N0 + 1 / (g0 * tau_p))

self_feedback = 0.0
coupling = 1.0
noise_amplitude = 1.0

N_lasers = 2
coupling_scheme = "ATA"
dx = 0.7

# --- Sweep controls ---
detuning_vals_ghz = np.linspace(0.0, 5.0, 2)
kappa_vals = np.linspace(0.001e9, 20e9, 2)
n_cases = 100  # number of noise iterations
n_freq_offsets = 100
freq_min_ghz = -0.5
freq_max_ghz = 0.5

dt = 1 * tau_p
Tmax = 5e-7
ramp_start = 10
ramp_shape = 20

# Injection parameters
kappa_inj_width = 5 * tau
kappa_inj_scale = 5.0  # injected amplitude multiplier vs final_kappa
peak_time = 200 * tau

# Plot controls
plot_cmap = "viridis"
n_jobs = -1
n_jobs_stability = -1
label_fs = 18
title_fs = 20
tick_fs = 16
cbar_tick_fs = 18
tag_fs = 20
save_running_plots = True
plot_dir = Path("../injection_tests/injection_freq_phase")

injected_phase = 0.0
freq_offsets = np.linspace(freq_min_ghz, freq_max_ghz, n_freq_offsets)


def run_sweep(detuning_ghz, final_kappa):
    delta = detuning_ghz * 2 * np.pi * 1e9
    delta_dist = np.sort(np.concatenate([delta / 2 * np.linspace(-1, 1, N_lasers)]))

    steps = int(Tmax / dt)
    time_arr = np.linspace(0, Tmax, steps)
    delay_steps = int(tau / dt)
    segment_start = int(2 * steps / 3)

    kappa_arr = VCSEL.build_coupling_matrix(
        time_arr=time_arr,
        kappa_initial=0.0,
        kappa_final=final_kappa,
        N_lasers=N_lasers,
        ramp_start=ramp_start,
        ramp_shape=ramp_shape,
        tau=tau,
        scheme=coupling_scheme,
        plot=False,
        dx=dx,
    )

    phys = {
        "tau_p": tau_p,
        "tau_n": tau_n,
        "g0": g0,
        "N0": N0,
        "N_bar": N0 + 1 / (g0 * tau_p),
        "s": s,
        "beta": beta,
        "kappa_c_mat": kappa_arr[-1, :, :],
        "phi_p_mat": np.zeros((1, N_lasers, N_lasers)),
        "I": I,
        "q": q,
        "alpha": alpha,
        "delta": delta_dist,
        "coupling": coupling,
        "self_feedback": self_feedback,
        "noise_amplitude": noise_amplitude,
        "dt": dt,
        "Tmax": Tmax,
        "tau": tau,
        "N_lasers": N_lasers,
        "sparse": False,
        "injection": False,
    }

    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()

    counts = {'phase_count': 5, 'freq_count': 5}
    eq, results, E_tot = vcsel.solve_equilibria(nd, counts=counts)
    if eq is None or results is None or results.size == 0:
        return None, None

    N = 30
    n_eigenvalues = N * 3 * N_lasers - 1
    stab_results = Parallel(n_jobs=n_jobs_stability)(
        delayed(vcsel.compute_stability)(
            eq_pt,
            nd,
            N=N,
            newton_maxit=10000,
            threshold=1e-10,
            sparse=phys["sparse"],
            spectral_shift=0.01 + 0.01j,
            n_eigenvalues=n_eigenvalues,
        )
        for eq_pt in results
    )
    stable_mask = np.array([res[0] for res in stab_results], dtype=bool)
    if not np.any(stable_mask):
        return None, None

    stable_eq_index = np.argmax(E_tot[stable_mask])
    eq = results[stable_mask][stable_eq_index]
    omega_target = eq[-1] / (2 * np.pi * 1e9 * tau_p)

    # --- Injection setup ---
    phys["injection"] = True
    injection_array = np.zeros(N_lasers)
    injection_array[(N_lasers - 1) // 2] = 1
    phys["injection_topology"] = injection_array
    phys["injected_strength"] = nd["sbar"]
    phys["kappa_injection"] = kappa_inj_scale * final_kappa * np.exp(
        -((time_arr - peak_time) ** 2) / (2 * kappa_inj_width**2)
    )
    n_freq = len(freq_offsets)
    phys["kappa_injection"] = np.tile(phys["kappa_injection"], (n_freq, 1))

    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()

    history, freq_history, _, _ = vcsel.generate_history(nd, shape="FR", n_cases=n_freq)
    nd["phi_p"] = np.tile(phys["phi_p_mat"][0], (n_freq, 1, 1))

    def run_noise_iteration(_iter_idx):
        nd_local = dict(nd)
        nd_local["injected_phase_diff"] = injected_phase * np.ones(n_freq)
        nd_local["injected_frequency"] = (
            np.tile(omega_target + freq_offsets, (len(time_arr), 1)).T
            * 1e9
            * (2 * np.pi * tau_p)
        )
        nd_local["kappa_inj"] = phys["kappa_injection"] * tau_p

        t, y, freqs = vcsel.integrate(
            history, nd=nd_local, progress=False, theta=0.5, max_iter=1
        )

        dphi = freqs * 1e-9 / (2 * np.pi * tau_p)
        dphi[:, :, : 2 * delay_steps] = freq_history

        avg_row = np.mean(np.abs(dphi[:, :, segment_start:] - omega_target), axis=(1, 2))
        std_row = np.std(np.abs(dphi[:, :, segment_start:] - omega_target), axis=(1, 2))
        return avg_row, std_row

    @contextmanager
    def tqdm_joblib(tqdm_object):
        class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_cb = parallel.BatchCompletionCallBack
        parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            parallel.BatchCompletionCallBack = old_cb
            tqdm_object.close()

    with tqdm_joblib(
        tqdm(
            total=n_cases,
            desc=f"Noise iters (Δ={detuning_ghz:.2f}, κ={final_kappa/1e9:.2f})",
        )
    ):
        iter_results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size='auto')(
            delayed(run_noise_iteration)(it) for it in range(n_cases)
        )
    avg_matrix = np.array([res[0] for res in iter_results])  # (n_cases, n_freq)
    std_matrix = np.array([res[1] for res in iter_results])  # (n_cases, n_freq)
    avg_freq_diff = avg_matrix.mean(axis=0, keepdims=True)
    std_freq_diff = std_matrix.mean(axis=0, keepdims=True)
    return avg_freq_diff, std_freq_diff


def parse_args():
    parser = argparse.ArgumentParser(description="Injection noise robustness sweep")
    parser.add_argument("--detuning-idx", type=int, default=None, help="Index into detuning_vals_ghz")
    parser.add_argument("--kappa-idx", type=int, default=None, help="Index into kappa_vals")
    parser.add_argument("--task-id", type=int, default=None, help="Flattened task index over detuning x kappa")
    parser.add_argument("--output-dir", type=str, default="../injection_tests/injection_noise_robustness_hpc")
    parser.add_argument("--save-name", type=str, default=None, help="Optional output filename for single-task mode")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    return parser.parse_args()


def resolve_task_indices(args):
    if args.task_id is not None:
        total = len(detuning_vals_ghz) * len(kappa_vals)
        if args.task_id < 0 or args.task_id >= total:
            raise ValueError(f"task-id {args.task_id} out of range [0, {total-1}]")
        det_idx = args.task_id // len(kappa_vals)
        kap_idx = args.task_id % len(kappa_vals)
        return det_idx, kap_idx
    if args.detuning_idx is not None and args.kappa_idx is not None:
        return args.detuning_idx, args.kappa_idx
    return None, None


def save_single_result(avg_freq_diff, std_freq_diff, detuning_ghz, final_kappa, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_name is not None:
        out_path = output_dir / args.save_name
    else:
        out_path = output_dir / f"result_det{detuning_ghz:.4f}_kappa{final_kappa/1e9:.4f}.npz"
    if avg_freq_diff is None:
        np.savez(
            out_path,
            success=False,
            detuning_ghz=detuning_ghz,
            final_kappa=final_kappa,
            freq_offsets=freq_offsets,
        )
    else:
        np.savez(
            out_path,
            success=True,
            detuning_ghz=detuning_ghz,
            final_kappa=final_kappa,
            freq_offsets=freq_offsets,
            avg_freq_diff=avg_freq_diff,
            std_freq_diff=std_freq_diff,
        )
    print(f"Saved: {out_path}")


def plot_summary(avg_freq_diff, std_freq_diff):
    fig, axes = plt.subplots(1, 1, figsize=(10, 6), dpi=200)

    avg_freq_diff_mean = np.mean(avg_freq_diff, axis=0)
    std_freq_diff_mean = np.mean(std_freq_diff, axis=0)

    spl = make_interp_spline(freq_offsets, avg_freq_diff_mean, k=3)
    freq_offsets_smooth = np.linspace(freq_offsets[0], freq_offsets[-1], 300)
    avg_freq_diff_smooth = spl(freq_offsets_smooth)

    axes.errorbar(
        freq_offsets,
        avg_freq_diff_mean,
        yerr=std_freq_diff_mean,
        fmt='o',
        color='tab:blue',
        ecolor='tab:blue',
        elinewidth=1,
        capsize=5,
        markersize=6,
        alpha=0.9,
        label=r'Mean $\pm$ Std',
    )
    axes.plot(freq_offsets_smooth, avg_freq_diff_smooth, 'k-', linewidth=2, label='Interpolated curve', zorder=0)
    axes.set_xlabel(r'Frequency Offset (GHz)', fontsize=label_fs + 4)
    axes.set_ylabel(r'$\mu(|\dot{\phi}-\omega|)$', fontsize=label_fs + 4)
    axes.set_xticks(np.arange(freq_offsets[0], freq_offsets[-1] + 0.05, 0.1))
    axes.tick_params(axis='both', labelsize=tick_fs + 4)
    axes.grid(True, alpha=0.3)
    axes.legend(loc='lower left', fontsize=tick_fs + 4)
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    det_idx, kap_idx = resolve_task_indices(args)

    if det_idx is not None and kap_idx is not None:
        detuning_ghz = detuning_vals_ghz[det_idx]
        final_kappa = kappa_vals[kap_idx]
        avg_freq_diff, std_freq_diff = run_sweep(detuning_ghz, final_kappa)
        save_single_result(avg_freq_diff, std_freq_diff, detuning_ghz, final_kappa, args)
        if args.no_plot:
            return
        if avg_freq_diff is None:
            return
        plot_summary(avg_freq_diff, std_freq_diff)
        return
    else:
        running_avg = None
        running_std = None
        count = 0
        for detuning_ghz in detuning_vals_ghz:
            for final_kappa in kappa_vals:
                avg_freq_diff, std_freq_diff = run_sweep(detuning_ghz, final_kappa)
                if avg_freq_diff is None:
                    continue
                if running_avg is None:
                    running_avg = np.zeros_like(avg_freq_diff)
                    running_std = np.zeros_like(std_freq_diff)
                running_avg += avg_freq_diff
                running_std += std_freq_diff
                count += 1

        if count == 0:
            raise RuntimeError("No successful sweeps produced results.")

        avg_freq_diff = running_avg / count
        std_freq_diff = running_std / count
        if args.no_plot:
            return
        plot_summary(avg_freq_diff, std_freq_diff)
        return



if __name__ == "__main__":
    main()
