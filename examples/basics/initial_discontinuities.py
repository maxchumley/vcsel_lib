#%%
# Run simple_example-style simulation for two ramp_shape values
# and compare total output power on side-by-side plots.

import numpy as np
from vcsel_lib import VCSEL
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.constants import hbar, c


rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
plt.rc("font", family="serif")


# Physical parameters
alpha = 2
tau_p = 5.4e-12
tau_n = 0.25e-9
g0 = 8.75e-4 * 1e9
N0 = 2.86e5
s = 4e-6
q = 1.602e-19
beta = 1.0e-3
tau = 1e-9
eta = 0.9
current_threshold = 3
I = eta * current_threshold * q / tau_n * (N0 + 1 / (g0 * tau_p))

lam = 910e-9
omega0 = 2 * np.pi * c / lam

self_feedback = 0.0
coupling = 1.0
noise_amplitude = 0.0

N_lasers = 2
detuning = 0.5 # GHz
delta = detuning * 2 * np.pi * 1e9
delta_dist = delta / 2 * np.linspace(-1, 1, N_lasers)

dt = 1 * tau_p
Tmax = 2e-7
steps = int(Tmax / dt)
time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)

kappa_c = 20e9
ramp_start = 3.5
aMAT = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)

# Two ramp_shape values to compare
ramp_shape_1 = 1e-10
ramp_shape_2 = 100


def run_case(ramp_shape):
    kappa_arr = VCSEL.build_coupling_matrix(
        time_arr=time_arr,
        kappa_initial=0.0,
        kappa_final=kappa_c,
        N_lasers=N_lasers,
        ramp_start=ramp_start,
        ramp_shape=ramp_shape,
        tau=tau,
        scheme="CUSTOM",
        aMAT=aMAT,
    )

    phys = {
        "tau_p": tau_p,
        "tau_n": tau_n,
        "g0": g0,
        "N0": N0,
        "N_bar": N0 + 1 / (g0 * tau_p),
        "s": s,
        "beta": beta,
        "kappa_c_mat": kappa_arr,
        "phi_p_mat": 0*np.pi*np.ones((1, N_lasers, N_lasers)),
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
        "save_every": 1,
    }

    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    history, _, _, _ = vcsel.generate_history(nd, shape="FR", n_cases=1)
    t, y, _ = vcsel.integrate(history, nd=nd, progress=True, max_iter=1, smooth_freqs=False)

    S_all = y[:, 1::3, :]
    phi_all = y[:, 2::3, :]
    E_all = np.sqrt(S_all) * (np.cos(phi_all) + 1j * np.sin(phi_all))
    E_tot = np.abs(np.sum(E_all, axis=1)) ** 2

    intensity_to_mW = 1e3 * hbar * omega0 / (g0 * tau_n * tau_p)
    P_tot_mW = E_tot[0, :] * intensity_to_mW

    # Use off-diagonal coupling element as the plotted ramp (ns^-1)
    kappa_plot_ns_inv = kappa_arr[: len(t), 0, 1] * 1e-9
    return t, P_tot_mW, kappa_plot_ns_inv


t1, P_tot1, kappa_plot1 = run_case(ramp_shape_1)
t2, P_tot2, kappa_plot2 = run_case(ramp_shape_2)


# Side-by-side plot
fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=200, sharey=True)

time_plot1 = t1 * 1e6
time_plot2 = t2 * 1e6

line_p1 = axs[0].plot(time_plot1, P_tot1, color="tab:blue", linewidth=1.5, label=r"$P_{tot}$")[0]
axs[0].axvspan(0, 2 * delay_steps * dt * 1e6, color="gray", alpha=0.2)
# axs[0].set_title(rf"$P_{{tot,1}}$: ramp\_shape={ramp_shape_1}", fontsize=30)
axs[0].set_xlabel(r"Time ($\mu s$)", fontsize=26)
axs[0].set_ylabel("Total Output Power (mW)", fontsize=26)
axs[0].grid(True, alpha=0.2)
ax2_0 = axs[0].twinx()
line_k1 = ax2_0.plot(time_plot1, kappa_plot1, "k--", linewidth=1.8, alpha=0.7, label=r"$\kappa_c$")[0]
# Left panel: hide coupling-axis ticks/labels so only the far-right panel
# displays coupling-axis numbers.
ax2_0.set_ylabel("")
ax2_0.set_ylim(0, max(np.max(kappa_plot1), np.max(kappa_plot2)) * 1.05)
ax2_0.tick_params(
    axis="y",
    which="both",
    right=False,
    left=False,
    labelright=False,
    labelleft=False,
)
ax2_0.spines["right"].set_visible(False)
# axs[0].legend(handles=[line_p1, line_k1], loc="upper left", fontsize=20)

line_p2 = axs[1].plot(time_plot2, P_tot2, color="tab:red", linewidth=1.5, label=r"$P_{tot}$")[0]
axs[1].axvspan(0, 2 * delay_steps * dt * 1e6, color="gray", alpha=0.2)
# axs[1].set_title(rf"$P_{{tot,2}}$: ramp\_shape={ramp_shape_2}", fontsize=30)
axs[1].set_xlabel(r"Time ($\mu s$)", fontsize=26)
axs[1].grid(True, alpha=0.2)
ax2_1 = axs[1].twinx()
line_k2 = ax2_1.plot(time_plot2, kappa_plot2, "k--", linewidth=1.8, alpha=0.7, label=r"$\kappa_c$")[0]
ax2_1.set_ylabel(r"Coupling ($\mathrm{ns}^{-1}$)", fontsize=24)
ax2_1.set_ylim(0, max(np.max(kappa_plot1), np.max(kappa_plot2)) * 1.05)
ax2_1.yaxis.set_label_position("right")
ax2_1.yaxis.set_ticks_position("right")
ax2_1.tick_params(
    axis="y",
    which="both",
    right=True,
    left=False,
    labelright=True,
    labelleft=False,
    labelsize=20,
)
# axs[1].legend(handles=[line_p2, line_k2], loc="upper left", fontsize=20)

for ax in axs:
    ax.tick_params(axis="both", which="major", labelsize=20)

plt.tight_layout()
plt.show()
plt.close(fig)
