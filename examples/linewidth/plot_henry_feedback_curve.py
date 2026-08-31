#%%
"""Plot the weak-feedback Henry linewidth correction for the single-laser case."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

try:
    from examples.linewidth import single_laser_linewidth as sl
except ModuleNotFoundError:
    import single_laser_linewidth as sl


def main():
    time_arr = np.array([0.0, sl.tau_p * sl.dt_multiplier], dtype=float)
    vcsel = sl.VCSEL(sl.make_physical_params(sl.make_kappa_matrix(time_arr, 0.0, 0.0)))
    nd = sl.apply_integrator_controls(vcsel.scale_params())
    delta_nu_0_hz = sl.free_running_henry_linewidth_hz(nd)

    kappa_ns = np.linspace(sl.kappa_initial, sl.kappa_final, 1200) * 1e-9
    kappa_s = kappa_ns * 1e9
    feedback_factor = kappa_s * sl.tau * np.sqrt(1.0 + sl.alpha**2)

    # Henry feedback depends on psi = Omega*tau + phi_p + atan(alpha).
    phase_grid = np.linspace(-np.pi, np.pi, 721)
    denominator = 1.0 + feedback_factor[:, None] * np.cos(phase_grid)[None, :]
    family_mhz = delta_nu_0_hz / np.maximum(denominator**2, 1e-30) * 1e-6
    finite_family = family_mhz[
        np.isfinite(family_mhz)
        & (family_mhz > 0.0)
        & (family_mhz < 1e8)
    ]

    family_min_mhz = np.nanmin(family_mhz, axis=1)
    family_max_mhz = np.nanmax(
        np.where(family_mhz < 1e8, family_mhz, np.nan),
        axis=1,
    )
    branches = sl.henry_ecm_branch_points(kappa_s, delta_nu_0_hz)
    branch_kappa_ns = branches["kappa_values"] * 1e-9
    branch_mhz = branches["linewidth_hz"] * 1e-6
    branch_positive_slope = branches["positive_slope"]

    phase_cuts = [0.0, np.pi / 2.0, np.pi]
    labels = [
        r"$\cos\psi=+1$",
        r"$\cos\psi=0$",
        r"$\cos\psi=-1$",
    ]
    colors = ["tab:blue", "black", "tab:red"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "legend.fontsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=180)
    ax.fill_between(
        kappa_ns,
        family_min_mhz,
        family_max_mhz,
        color="0.8",
        alpha=0.5,
        label="all feedback phases",
    )
    finite_branches = np.isfinite(branch_kappa_ns) & np.isfinite(branch_mhz) & (branch_mhz > 0.0)
    if np.any(finite_branches & ~branch_positive_slope):
        ax.scatter(
            branch_kappa_ns[finite_branches & ~branch_positive_slope],
            branch_mhz[finite_branches & ~branch_positive_slope],
            s=6,
            color="0.55",
            alpha=0.25,
            linewidths=0,
            rasterized=True,
            label="other ECM roots",
        )
    if np.any(finite_branches & branch_positive_slope):
        ax.scatter(
            branch_kappa_ns[finite_branches & branch_positive_slope],
            branch_mhz[finite_branches & branch_positive_slope],
            s=9,
            color="tab:red",
            alpha=0.45,
            linewidths=0,
            rasterized=True,
            label="ECM Henry roots",
        )

    for phase, label, color in zip(phase_cuts, labels, colors):
        curve_mhz = (
            delta_nu_0_hz
            / np.maximum((1.0 + feedback_factor * np.cos(phase)) ** 2, 1e-30)
            * 1e-6
        )
        ax.plot(kappa_ns, curve_mhz, color=color, linewidth=2.2, label=label)

    ax.axhline(
        delta_nu_0_hz * 1e-6,
        color="0.25",
        linestyle="--",
        linewidth=1.5,
        label=rf"free-running: {delta_nu_0_hz * 1e-6:.2f} MHz",
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"feedback strength $\kappa$ (ns$^{-1}$)")
    ax.set_ylabel(r"Henry feedback linewidth $\Delta\nu_H$ (MHz)")
    ax.set_title(
        rf"Weak-feedback Henry correction, "
        rf"$\alpha={sl.alpha:g}$, $\tau={sl.tau * 1e9:.2f}$ ns"
    )
    ax.set_xlim(kappa_ns[0], kappa_ns[-1])
    if finite_family.size:
        ax.set_ylim(
            max(1e-6, np.nanpercentile(finite_family, 0.5) * 0.5),
            min(1e8, np.nanpercentile(finite_family, 99.5) * 2.0),
        )
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", axis="y", linestyle="--", alpha=0.22)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    output_path = (
        sl.LINEWIDTH_RESULTS_DIR
        / "1_laser/self_feedback_autocorr/henry_feedback_theory_curve.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
