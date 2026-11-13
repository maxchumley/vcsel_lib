"""Display controls only; these never change a training objective."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PlotOptions:
    """Portable fonts by default; numerical plots match the current metrics."""
    use_tex: bool = False
    dpi: int = 200
    trajectory_mode: str = "all"
    profile_smoothing_bins: float = 3.5
    max_display_edges: int = 10000
    edge_selection: str = "strongest"
    show_ensemble: bool = True
    show_far_field: bool = True
    emitter_spacing_um: float = 10.0
    wavelength_m: float = 910e-9
    theta_range_deg: tuple[float, float] = (-10., 10.)
    n_theta: int = 801
    max_time_points: int = 2000
    element_fwhm_deg: float | None = 5.0
    angular_blur_fwhm_deg: float | None = None
    layout: str = "separate"  # inspection adds ensemble matrices before ensemble trajectories.
    phase_error_deg: float = 3.0
    far_field_final_slice: bool = True
    topology_bins: int | None = None
    legend_location: str = "upper right"  # "best" scans every trajectory on each draw.

    def __post_init__(self):
        if self.layout not in {"separate", "inspection"}:
            raise ValueError("layout must be separate or inspection")
        if not 0 < self.phase_error_deg <= 180:
            raise ValueError("phase_error_deg must lie in (0,180]")
        if self.topology_bins is not None and self.topology_bins < 1:
            raise ValueError("topology_bins must be positive when supplied")
        if self.trajectory_mode not in {"all", "mean_sd"}:
            raise ValueError("trajectory_mode must be all or mean_sd")
        if self.dpi < 1 or self.max_display_edges < 1:
            raise ValueError("dpi and max_display_edges must be positive")
        if self.edge_selection not in {"strongest", "strength_spread"}:
            raise ValueError("edge_selection must be strongest or strength_spread")
        if self.angular_blur_fwhm_deg is not None and self.angular_blur_fwhm_deg <= 0:
            raise ValueError("angular_blur_fwhm_deg must be positive when supplied")
        if self.emitter_spacing_um <= 0 or self.wavelength_m <= 0:
            raise ValueError("Emitter spacing and wavelength must be positive")
        if self.n_theta < 2 or self.max_time_points < 2:
            raise ValueError("Far-field angle/time grids need at least two points")
        if len(self.theta_range_deg) != 2 or self.theta_range_deg[1] <= self.theta_range_deg[0]:
            raise ValueError("theta_range_deg must be an increasing (minimum,maximum) pair")
        if self.element_fwhm_deg is not None and self.element_fwhm_deg <= 0:
            raise ValueError("element_fwhm_deg must be positive when supplied")
