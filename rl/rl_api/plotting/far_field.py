"""Far-field calculation and rendering, separate from training and simulation."""
import matplotlib.pyplot as plt
import numpy as np


def far_field_intensity_map(S, phi, lambda0, d, *, theta_range_deg=(-10., 10.),
                            n_theta=801, max_time_points=2000, element_fwhm_deg=None):
    """Pure angle/time map used by both API layouts and the original renderer.

    Same calculation as examples/visualization/far_field_intensity.py, kept
    local to avoid that script's simulator import and global plotting defaults.
    Process 256 times at once to keep temporary arrays small.
    """
    photons, phases = np.asarray(S), np.asarray(phi)
    if photons.ndim == 2:
        photons, phases = photons[None], phases[None]
    if photons.ndim != 3 or photons.shape != phases.shape:
        raise ValueError("Intensity and phase arrays must have matching (cases,M,times) shapes")
    length = photons.shape[-1]
    indices = np.unique(np.round(np.linspace(0, length-1,
        min(length, max_time_points))).astype(int))
    photons, phases = photons[:, :, indices], phases[:, :, indices]
    if not np.isfinite(photons).all() or not np.isfinite(phases).all():
        raise ValueError("Far-field trajectories must be finite")
    scale = float(np.max(np.clip(photons, 0, None)))
    if scale <= 0:
        raise ValueError("The selected trajectory contains no positive intensity")
    count, lasers, _ = photons.shape
    positions = (np.arange(lasers)-.5*(lasers-1))*d
    theta = np.linspace(*theta_range_deg, n_theta)
    steering = np.exp(1j*(2*np.pi/lambda0)*
        np.outer(np.sin(np.deg2rad(theta)), positions))
    intensity = np.zeros((len(theta), len(indices)))
    for case in range(count):
        for start in range(0, len(indices), 256):
            stop = start+256
            amplitude = np.sqrt(np.clip(photons[case, :, start:stop]/scale, 0, 1))
            fields = amplitude*np.exp(1j*np.remainder(phases[case, :, start:stop], 2*np.pi))
            far_field = np.einsum("ae,et->at", steering, fields, optimize=False)
            intensity[:, start:stop] += np.abs(far_field)**2/count
    if element_fwhm_deg is not None:
        intensity *= np.exp(-4*np.log(2)*(theta/element_fwhm_deg)**2)[:, None]
    return theta, indices, intensity/np.max(intensity)


def calculate_far_field(simulation, options):
    """Calculate the array factor and normalize to the final-slice peak."""
    times, states = simulation[:2]
    theta, indices, intensity = far_field_intensity_map(states[:, 1::3], states[:, 2::3],
        options.wavelength_m, options.emitter_spacing_um*1e-6,
        theta_range_deg=options.theta_range_deg, n_theta=options.n_theta,
        max_time_points=options.max_time_points, element_fwhm_deg=options.element_fwhm_deg)
    # Preserve the reference's normalization order before optional blurring.
    if options.angular_blur_fwhm_deg is not None:
        step = float(np.median(np.diff(theta)))
        sigma = options.angular_blur_fwhm_deg / (2*np.sqrt(2*np.log(2)))
        radius = min(int(np.ceil(4*sigma/step)), len(theta)-1)
        angles = np.arange(-radius, radius+1)*step
        kernel = np.exp(-.5*(angles/sigma)**2)
        kernel /= kernel.sum()
        padded = np.pad(intensity, ((radius, radius), (0,0)), mode="edge")
        intensity = np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="valid"),
                                       0, padded)
    peak = float(np.max(intensity[:, -1]))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("The final far-field slice has no positive finite intensity")
    return dict(theta_deg=theta, time_ns=times[indices]*1e9,
                intensity_normalized=intensity/peak)


def plot_far_field(data):
    """Draw time-resolved intensity plus the normalized final angular slice."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True,
                               gridspec_kw={"width_ratios": (4, 1)}, sharey=True)
    intensity = data["intensity_normalized"]
    image = axes[0].pcolormesh(data["time_ns"], data["theta_deg"], intensity,
                             shading="auto", cmap="inferno", vmin=0., vmax=1.)
    axes[0].set(xlabel="time (ns)", ylabel="angle (degrees)", title="Far-field intensity")
    axes[1].plot(intensity[:, -1], data["theta_deg"], color="tab:orange")
    axes[1].set(xlabel="normalized intensity", title="Final slice", xlim=(0, 1.05))
    axes[1].grid(alpha=.25)
    figure.colorbar(image, ax=axes[0], label="normalized intensity")
    return figure
