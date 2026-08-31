#%%
# Stability branches for Ma et al. 2019 Figure 4 parameters.
#
# Figure 4 caption:
# tau_d = 5 ns, kappa_c = 5e8 s^-1, phi_p = 0, delta = 0.3 GHz.
#
# In this codebase, CUSTOM two-laser coupling normalizes the two off-diagonal
# links to 0.5 each, so a total kappa budget of 1e9 s^-1 gives a per-link
# coupling of 5e8 s^-1.

from pathlib import Path
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib import rc
from tqdm import tqdm

from vcsel_lib import VCSEL

try:
    from examples._paths import PAPER_DATA_DIR, PAPER_RESULTS_DIR
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples._paths import PAPER_DATA_DIR, PAPER_RESULTS_DIR


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('font', family='serif')


# Controls
run_sweep = True
save_data = True
use_saved_data = False

def get_script_dir():
    if "__file__" in globals():
        return Path(__file__).resolve().parent

    cwd = Path.cwd().resolve()
    if (cwd / "fig_4_stability_manifold.py").exists():
        return cwd

    repo_style_dir = cwd / "examples"
    if (repo_style_dir / "fig_4_stability_manifold.py").exists():
        return repo_style_dir

    return cwd


script_dir = get_script_dir()
data_dir = PAPER_DATA_DIR
plot_dir = PAPER_RESULTS_DIR
data_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

data_path = data_dir / "fig_4_stability_manifold_phi_p0.npy"
plot_path = plot_dir / "fig_4_stability_manifold_phi_p0.png"



# Ma Fig. 4 parameters.
N_lasers = 2
alpha = 4
tau_p = 7.15e-12
tau_n = 0.33e-9
g0 = 1.13e4
N0 = 8.2e6
s = 0.0
q = 1.602e-19
beta = 3.54e-5
tau = 5e-9

detuning = 0.3  # GHz, DFB#2 relative to DFB#1
delta_total = detuning * 2 * np.pi * 1e9
delta = delta_total / 2 * np.linspace(0, 2, N_lasers)

N_bar_free_running = N0 + 1 / (g0 * tau_p)
I = 4.0 * q * N_bar_free_running / tau_n

phi_p = 0.0
phi_p_mat = np.full((N_lasers, N_lasers), phi_p)

kappa_vals = np.linspace(0.0e9, 1.0e9, 250)
kappa_plot = kappa_vals * 1e-9

dt = 0.5 * tau_p
Tmax = 3e-7
steps = int(Tmax / dt)
time_arr = np.linspace(0, Tmax, steps)

coupling_scheme = "CUSTOM"
aMAT = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)
coupling = 1.0
self_feedback = 0.0
noise_amplitude = 0.0
sparse = True


def make_phys(kappa_budget):
    kappa_arr = VCSEL.build_coupling_matrix(
        time_arr=time_arr,
        kappa_initial=kappa_budget,
        kappa_final=kappa_budget,
        N_lasers=N_lasers,
        ramp_start=0.0,
        ramp_shape=1.0,
        tau=tau,
        scheme=coupling_scheme,
        plot=False,
        aMAT=aMAT,
    )
    return {
        'tau_p': tau_p,
        'tau_n': tau_n,
        'g0': g0,
        'N0': N0,
        'N_bar': N0 + 1 / (g0 * tau_p),
        's': s,
        'beta': beta,
        'kappa_c_mat': kappa_arr[-1, :, :],
        'phi_p_mat': phi_p_mat,
        'I': I,
        'q': q,
        'alpha': alpha,
        'delta': delta,
        'coupling': coupling,
        'self_feedback': self_feedback,
        'noise_amplitude': noise_amplitude,
        'dt': dt,
        'Tmax': Tmax,
        'tau': tau,
        'N_lasers': N_lasers,
        'sparse': sparse,
    }


def solve_stability_sweep():
    all_eqs = []
    stable_arr = []
    max_num_eqs = 0
    eqs_previous = None
    avg_time = 0.0

    for kappa_c in tqdm(kappa_vals, desc="Ma Fig. 4 stability manifold"):
        phys = make_phys(kappa_c)
        vcsel = VCSEL(phys)
        nd = vcsel.scale_params()

        guesses = []
        if eqs_previous is not None:
            for eq_pt in eqs_previous:
                if np.any(np.isnan(eq_pt)):
                    continue
                guesses.append(np.concatenate([
                    eq_pt[1::2][:N_lasers],
                    eq_pt[2 * N_lasers:3 * N_lasers - 1],
                    np.array([eq_pt[-2]]),
                ]))

        counts = {
            'phase_count': 9,
            'freq_count': 120,
            'max_refine': 2,
            'refine_factor': 2,
        }
        _, eqs, _ = vcsel.solve_equilibria(nd, guesses=guesses, counts=counts)
        if eqs is None:
            eqs = np.empty((0, 2 * N_lasers + (N_lasers - 1) + 1))

        max_num_eqs = max(max_num_eqs, len(eqs))

        if len(eqs) > 0:
            start = time.time()
            N_delay = 30
            n_eigenvalues = N_delay * 3 * N_lasers - 1
            tmp_stable = Parallel(n_jobs=-1)(
                delayed(vcsel.compute_stability)(
                    eq_pt,
                    nd,
                    N=N_delay,
                    newton_maxit=10000,
                    threshold=1e-10,
                    sparse=phys['sparse'],
                    spectral_shift=0.01 + 0.01j,
                    n_eigenvalues=n_eigenvalues,
                )
                for eq_pt in eqs
            )
            avg_time += time.time() - start
            tmp_stable = [result[0] for result in tmp_stable]
        else:
            tmp_stable = []

        eqs_with_stability = np.column_stack([eqs, np.asarray(tmp_stable)]) if len(eqs) else eqs
        stable_arr.append(np.asarray(tmp_stable, dtype=float))
        all_eqs.append(eqs_with_stability)
        eqs_previous = eqs_with_stability

    expected_cols = 2 * N_lasers + (N_lasers - 1) + 2
    if max_num_eqs == 0:
        raise RuntimeError("No equilibria found anywhere in the kappa sweep.")

    padded_eqs = []
    padded_stability = []
    for eqs, st in zip(all_eqs, stable_arr):
        eqs = np.asarray(eqs, dtype=float)
        if eqs.size == 0:
            eqs_pad = np.nan * np.ones((max_num_eqs, expected_cols))
            st_pad = np.nan * np.ones(max_num_eqs)
        else:
            if eqs.ndim == 1:
                eqs = eqs.reshape(1, -1)
            missing = max_num_eqs - eqs.shape[0]
            if missing > 0:
                eqs = np.vstack([eqs, np.nan * np.ones((missing, expected_cols))])
                st = np.concatenate([st, np.nan * np.ones(missing)])
            eqs_pad = eqs
            st_pad = st
        padded_eqs.append(eqs_pad)
        padded_stability.append(st_pad)

    all_eqs = np.asarray(padded_eqs)
    stable_arr = np.asarray(padded_stability)
    print(f"Average stability solve time per kappa: {avg_time / max(1, len(kappa_vals)):.3f} s")
    return all_eqs, stable_arr



if run_sweep and not use_saved_data:
    all_eqs, stable_arr = solve_stability_sweep()
    if save_data:
        data_dict = {
            'all_eqs': all_eqs,
            'stable_arr': stable_arr,
            'kappa_vals': kappa_vals,
            'kappa_plot': kappa_plot,
            'params': {
                'parameter_set': 'ma2019_fig4_long_delay',
                'N_lasers': N_lasers,
                'alpha': alpha,
                'tau_p': tau_p,
                'tau_n': tau_n,
                'g0': g0,
                'N0': N0,
                's': s,
                'beta': beta,
                'tau': tau,
                'detuning_GHz': detuning,
                'delta': delta,
                'phi_p': phi_p,
                'kappa_budget_convention': 'CUSTOM total budget; each off-diagonal link is kappa_budget/2',
            },
        }
        np.save(data_path, data_dict)
        print(f"Saved {data_path.resolve()}")
else:
    data_dict = np.load(data_path, allow_pickle=True).item()
    all_eqs = data_dict['all_eqs']
    stable_arr = data_dict['stable_arr']
    kappa_vals = data_dict['kappa_vals']
    kappa_plot = data_dict.get('kappa_plot', kappa_vals * 1e-9)


#%%
# Plot frequency branches colored by stability.
omega = all_eqs[:, :, -2]
omega_MHz = omega / (2 * np.pi * 1e6 * tau_p)

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

colors_discrete = ['red', 'blue']
cmap_discrete = mpl.colors.ListedColormap(colors_discrete)
norm_discrete = mpl.colors.BoundaryNorm([0, 0.5, 1.0], cmap_discrete.N)

scatter = None
for idx, kappa_ns in enumerate(kappa_plot):
    y = omega_MHz[idx, :]
    st = stable_arr[idx, :]
    mask = np.isfinite(y) & np.isfinite(st)
    if not np.any(mask):
        continue
    scatter = ax.scatter(
        kappa_ns * np.ones(np.sum(mask)),
        y[mask],
        c=st[mask],
        cmap=cmap_discrete,
        norm=norm_discrete,
        s=7,
        edgecolors='none',
    )

if scatter is None:
    raise RuntimeError("No finite branch data available to plot.")

ax.axvline(1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
           label=r'Fig. 4 $\kappa_c$ budget')
ax.axhline(157.0, color='black', linestyle=':', linewidth=1.4, alpha=0.75,
           label='157 MHz')
ax.set_xlabel(r'$\kappa_c$ budget (ns$^{-1}$)', fontsize=20)
ax.set_ylabel(r'$\omega$ (MHz)', fontsize=20)
ax.set_title(r'Ma et al. Fig. 4 Stability Branches ($\phi_p=0$, $\delta=0.3$ GHz)', fontsize=20, pad=14)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.grid(alpha=0.25)
ax.legend(loc='best', fontsize=13)

cbar = fig.colorbar(scatter, ax=ax, pad=0.02, boundaries=[0, 0.5, 1.0], ticks=[0.25, 0.75])
cbar.ax.set_yticklabels(['Unstable', 'Stable'], fontsize=14, rotation=90)

plt.tight_layout()
fig.savefig(plot_path, bbox_inches='tight', facecolor='white')
print(f"Saved {plot_path.resolve()}")
plt.show()


#%%
# 3D branch plot: kappa budget, frequency, and coherent output power.
S = all_eqs[:, :, 1::2][:, :, :N_lasers]
phase_offsets = np.concatenate(
    (
        np.zeros((all_eqs.shape[0], all_eqs.shape[1], 1)),
        all_eqs[:, :, 2 * N_lasers:3 * N_lasers - 1],
    ),
    axis=2,
)
phi = omega[:, :, None] * tau + phase_offsets
E = np.sqrt(S) * (np.cos(phi) + 1j * np.sin(phi))
E_tot = np.abs(np.sum(E, axis=2)) ** 2

fig = plt.figure(figsize=(9, 7), dpi=300)
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

scatter_3d = None
for idx, kappa_ns in enumerate(kappa_plot):
    x = kappa_ns * np.ones(all_eqs.shape[1])
    y = omega_MHz[idx, :]
    z = E_tot[idx, :]
    st = stable_arr[idx, :]
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(st)
    if not np.any(mask):
        continue
    scatter_3d = ax.scatter(
        x[mask],
        y[mask],
        z[mask],
        c=st[mask],
        cmap=cmap_discrete,
        norm=norm_discrete,
        s=7,
        depthshade=False,
        edgecolors='none',
    )

if scatter_3d is None:
    raise RuntimeError("No finite 3D branch data available to plot.")

ax.set_xlabel(r'$\kappa_c$ budget (ns$^{-1}$)', labelpad=10, fontsize=15)
ax.set_ylabel(r'$\omega$ (MHz)', labelpad=10, fontsize=15)
ax.set_zlabel(r'$|E_{\mathrm{tot}}|^2$', labelpad=10, fontsize=15)
ax.set_title(r'Ma et al. Fig. 4 Branch Manifold ($\phi_p=0$)', fontsize=17, pad=16)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.view_init(elev=20, azim=170)

cbar = fig.colorbar(scatter_3d, ax=ax, pad=0.08, shrink=0.65, boundaries=[0, 0.5, 1.0], ticks=[0.25, 0.75])
cbar.ax.set_yticklabels(['Unstable', 'Stable'], fontsize=12, rotation=90)

plt.tight_layout()
fig_3d_path = plot_dir / "fig_4_stability_manifold_phi_p0_3d_power.png"
fig.savefig(fig_3d_path, bbox_inches='tight', facecolor='white')
print(f"Saved {fig_3d_path.resolve()}")
plt.show()


#%%
# Interactive Plotly 3D branch plot: pan and rotate the same manifold.
import plotly.graph_objects as go

save_plotly_html = True
plotly_html_path = plot_dir / "fig_4_stability_manifold_phi_p0_3d_power.html"

x_all = []
y_all = []
z_all = []
c_all = []
branch_all = []

for idx, kappa_ns in enumerate(kappa_plot):
    x = kappa_ns * np.ones(all_eqs.shape[1])
    y = omega_MHz[idx, :]
    z = E_tot[idx, :]
    st = stable_arr[idx, :]
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(st)
    if not np.any(mask):
        continue

    x_all.append(x[mask])
    y_all.append(y[mask])
    z_all.append(z[mask])
    c_all.append(st[mask])
    branch_all.append(np.where(mask)[0])

if not x_all:
    raise RuntimeError("No finite Plotly 3D branch data available to plot.")

x_plotly = np.concatenate(x_all)
y_plotly = np.concatenate(y_all)
z_plotly = np.concatenate(z_all)
c_plotly = np.concatenate(c_all)
branch_plotly = np.concatenate(branch_all)
stability_text = np.where(c_plotly >= 0.5, "Stable", "Unstable")

fig_plotly = go.Figure(
    data=[
        go.Scatter3d(
            x=x_plotly,
            y=y_plotly,
            z=z_plotly,
            mode="markers",
            marker=dict(
                size=2,
                color=c_plotly,
                cmin=0,
                cmax=1,
                colorscale=[[0.0, "red"], [0.499, "red"], [0.5, "blue"], [1.0, "blue"]],
                colorbar=dict(
                    title="Stability",
                    tickvals=[0.25, 0.75],
                    ticktext=["Unstable", "Stable"],
                ),
                opacity=0.95,
            ),
            customdata=np.column_stack([branch_plotly, stability_text]),
            hovertemplate=(
                r"$\kappa_c$ budget: %{x:.4g} ns^-1<br>"
                r"$\omega$: %{y:.4g} MHz<br>"
                r"$|E_{tot}|^2$: %{z:.4g}<br>"
                "Branch index: %{customdata[0]}<br>"
                "%{customdata[1]}"
                "<extra></extra>"
            ),
            name="Branches",
        ),
        go.Scatter3d(
            x=[1.0, 1.0],
            y=[157.0, 157.0],
            z=[np.nanmin(z_plotly), np.nanmax(z_plotly)],
            mode="lines",
            line=dict(color="black", width=4, dash="dash"),
            hovertemplate=(
                r"Fig. 4 reference<br>"
                r"$\kappa_c$ budget = 1 ns^-1<br>"
                r"$\omega$ = 157 MHz"
                "<extra></extra>"
            ),
            name="Fig. 4 reference",
        ),
    ]
)

fig_plotly.update_layout(
    title=r"Ma et al. Fig. 4 Branch Manifold ($\phi_p=0$)",
    scene=dict(
        xaxis=dict(title=r"$\kappa_c$ budget (ns$^{-1}$)", range=[np.nanmin(kappa_plot), np.nanmax(kappa_plot)]),
        yaxis=dict(title=r"$\omega$ (MHz)"),
        zaxis=dict(title=r"$|E_{tot}|^2$"),
        aspectmode="cube",
    ),
    legend=dict(x=0.02, y=0.98),
    margin=dict(l=0, r=0, b=0, t=45),
)

if save_plotly_html:
    fig_plotly.write_html(plotly_html_path)
    print(f"Saved {plotly_html_path.resolve()}")

fig_plotly.show()
