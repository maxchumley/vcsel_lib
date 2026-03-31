#%%

import numpy as np
import matplotlib.pyplot as plt
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
noise_amplitude = 0.0

N_lasers = 2
coupling_scheme = "ATA"
dx = 0.7

# --- Sweep controls ---
detuning_vals_ghz = np.linspace(0.0, 5.0, 10)
kappa_vals = np.linspace(0.001e9, 20e9, 10)
n_cases = 100
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

inj_phases = np.linspace(-np.pi, np.pi, n_cases)
freq_offsets = np.linspace(freq_min_ghz, freq_max_ghz, n_cases)


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
    phys["kappa_injection"] = np.tile(phys["kappa_injection"], (n_cases, 1))

    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()

    history, freq_history, _, _ = vcsel.generate_history(nd, shape="FR", n_cases=n_cases)
    nd["phi_p"] = np.tile(phys["phi_p_mat"][0], (n_cases, 1, 1))

    def run_phase(phase):
        nd_local = dict(nd)
        nd_local["injected_phase_diff"] = phase * np.ones(n_cases)
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
            total=len(inj_phases),
            desc=f"Phases (Δ={detuning_ghz:.2f}, κ={final_kappa/1e9:.2f})",
        )
    ):
        phase_results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size='auto')(
            delayed(run_phase)(phase) for phase in inj_phases
        )
    avg_freq_diff = np.array([res[0] for res in phase_results])
    std_freq_diff = np.array([res[1] for res in phase_results])
    return avg_freq_diff, std_freq_diff


running_avg = None
running_std = None
count = 0

def plot_running(avg_data, std_data, tag, step, filename=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    avg_color_max = np.ceil(np.max(avg_data) * 10) / 10
    std_color_max = np.ceil(np.max(std_data) * 10) / 10

    im1 = axes[0].imshow(
        avg_data,
        aspect="auto",
        cmap=plot_cmap,
        origin="lower",
        extent=[freq_offsets[0], freq_offsets[-1], -1.0, 1.0],
        vmin=0,
        vmax=avg_color_max,
    )
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar1.set_ticks([0, avg_color_max])
    cbar1.set_label(r"$\mu$", fontsize=label_fs)
    cbar1.ax.tick_params(labelsize=cbar_tick_fs)
    axes[0].set_xlabel(r"Uniform $\Delta \omega$ (GHz)", fontsize=label_fs)
    axes[0].set_ylabel(r"Injection Phase / $\pi$", fontsize=label_fs)
    axes[0].set_title(r"Average $\mu(|\dot{\phi}-\omega|)$", fontsize=title_fs)
    axes[0].set_xticks(np.linspace(freq_offsets[0], freq_offsets[-1], 5))
    axes[0].set_yticks([-1.0, 0.0, 1.0], ["-1", "0", "1"])
    axes[0].tick_params(axis="both", labelsize=tick_fs)

    im2 = axes[1].imshow(
        std_data,
        aspect="auto",
        cmap=plot_cmap,
        origin="lower",
        extent=[freq_offsets[0], freq_offsets[-1], -1.0, 1.0],
        vmin=0,
        vmax=std_color_max,
    )
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.set_ticks([0, std_color_max])
    cbar2.ax.set_yticklabels([f'{0:.1f}', f'{std_color_max:.1f}'])
    cbar2.set_label(r"$\sigma$", fontsize=label_fs)
    cbar2.ax.tick_params(labelsize=cbar_tick_fs)
    axes[1].set_xlabel(r"Uniform $\Delta \omega$ (GHz)", fontsize=label_fs)
    axes[1].set_ylabel(r"Injection Phase / $\pi$", fontsize=label_fs)
    axes[1].set_title(r"Average $\sigma(|\dot{\phi}-\omega|)$", fontsize=title_fs)
    axes[1].set_xticks(np.linspace(freq_offsets[0], freq_offsets[-1], 5))
    axes[1].set_yticks([-1.0, 0.0, 1.0], ["-1", "0", "1"])
    axes[1].tick_params(axis="both", labelsize=tick_fs)

    fig.suptitle(tag, fontsize=tag_fs)
    plt.tight_layout()
    if save_running_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"iter_{step:04d}.png"
        fig.savefig(plot_dir / filename, dpi=300)
    # plt.show(block=False)
    # plt.pause(0.001)
    plt.close(fig)

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
        plot_running(
            avg_freq_diff,
            std_freq_diff,
            rf"($\Delta$={detuning_ghz:.2f} GHz, $\kappa$={final_kappa/1e9:.2f} ns$^{{-1}}$)",
            count,
            filename=f"iter_{count:04d}_detuning{detuning_ghz:.2f}_kappa{final_kappa/1e9:.2f}.png",
        )
        plot_running(
            running_avg / count,
            running_std / count,
            f"(n={count})",
            count,
            filename="../running_avg.png",
        )

if count == 0:
    raise RuntimeError("No successful sweeps produced results.")

avg_freq_diff = running_avg / count
std_freq_diff = running_std / count

plot_running(avg_freq_diff, std_freq_diff, "", count, filename="../running_avg.png")



#%%
from scipy.interpolate import make_interp_spline
from mpl_toolkits.mplot3d import Axes3D


fig, axes = plt.subplots(1, 1, figsize=(10, 6), dpi=200)

# Average along the injection phase axis (axis 0)
avg_freq_diff_phase_avg = np.mean(avg_freq_diff, axis=0)
std_freq_diff_phase_avg = np.mean(std_freq_diff, axis=0)

# Create interpolation curve
spl = make_interp_spline(freq_offsets, avg_freq_diff_phase_avg, k=3)
freq_offsets_smooth = np.linspace(freq_offsets[0], freq_offsets[-1], 300)
avg_freq_diff_smooth = spl(freq_offsets_smooth)



# Plot mean colored by standard deviation
scatter = axes.scatter(
    freq_offsets, 
    avg_freq_diff_phase_avg, 
    c=std_freq_diff_phase_avg,
    cmap=plot_cmap,
    s=100,
    linewidth=2,
    edgecolors=None,
    vmin=0
)

# Plot interpolated curve
axes.plot(freq_offsets_smooth, avg_freq_diff_smooth, 'k-', linewidth=5, label='Interpolated curve', zorder=0)

axes.set_xlabel(r'Frequency Offset (GHz)', fontsize=label_fs)
axes.set_ylabel(r'$\mu(|\dot{\phi}-\omega|)$', fontsize=label_fs)
# axes.set_title(r'Mean (colored by Std Dev)', fontsize=title_fs)
axes.set_xticks(np.arange(freq_offsets[0], freq_offsets[-1] + 0.05, 0.1))
axes.tick_params(axis='both', labelsize=tick_fs)
axes.grid(True, alpha=0.3)

cbar = fig.colorbar(scatter, ax=axes)
cbar.set_label(r'$\sigma$', fontsize=label_fs)
cbar.ax.tick_params(labelsize=tick_fs)

plt.tight_layout()
plt.show()


#%%
fig = plt.figure(figsize=(12, 8), dpi=200)
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for freq_offsets and injection phases
freq_mesh, phase_mesh = np.meshgrid(freq_offsets, inj_phases / np.pi)

# Plot surface colored by standard deviation
surf = ax.plot_surface(freq_mesh, phase_mesh, avg_freq_diff, facecolors=plt.cm.get_cmap(plot_cmap)(
    (std_freq_diff - std_freq_diff.min()) / (std_freq_diff.max() - std_freq_diff.min())
), shade=False, alpha=0.8)

ax.set_xlabel(r'Frequency Offset (GHz)', fontsize=label_fs)
ax.set_ylabel(r'Injection Phase / $\pi$', fontsize=label_fs)
ax.set_zlabel(r'$\mu(|\dot{\phi}-\omega|)$', fontsize=label_fs)
ax.set_title(r'Average Frequency Difference (3D)', fontsize=title_fs)

ax.view_init(elev=20, azim=290)

# Normalize for colorbar
norm = plt.Normalize(vmin=std_freq_diff.min(), vmax=std_freq_diff.max())
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plot_cmap), ax=ax, shrink=0.5)
cbar.set_label(r'$\sigma$', fontsize=label_fs)

plt.tight_layout()
plt.show()

