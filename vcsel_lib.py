"""
vcsel_lib.py

VCSEL simulation utilities.

Provides a VCSEL class that encapsulates physical parameters, nondimensional
scaling, the nondimensional rate equations (with delay and optional noise),
integration routines (trapezoidal predictor-corrector with Euler-Maruyama noise),
utilities to generate histories and compute equilibria, and small helpers
(cosine ramp, order parameter, invert scaling).

This module is intended for research/teaching use and keeps a clear separation
between physical parameters (SI units) and nondimensional parameters used by
the integrator.

Author: Max Chumley with documentation assistance from GitHub Copilot and OpenAI CODEX
"""

import numpy as np
import warnings
from sympy import symbols, Eq, solve
from tqdm import tqdm
from scipy.optimize import root
from joblib import Parallel, delayed
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class VCSEL:
    """
    Class representing a Vertical-Cavity Surface-Emitting Laser (VCSEL)
    with delayed coupling and optional self-feedback and noise.

    The object stores:
      - phys: dictionary of physical (SI) parameters provided by the user
      - nd: dictionary of nondimensional parameters produced by scale_params()

    Usage:
      vcsel = VCSEL(phys_dict)
      nd = vcsel.scale_params()            # create nondimensional parameters
      history, freq_hist, eq, results = vcsel.generate_history(nd, shape='FR')
      t_dim, y, freqs = vcsel.integrate(history, nd, progress=True)

    Notes on units and nondimensionalization:
      - Time is nondimensionalized by the photon lifetime tau_p:
            t' = t / tau_p
      - Carrier numbers and photon numbers are rescaled according to g0, tau_p,
        and tau_n so that the nondimensional equations are O(1) in typical
        operating regimes. See scale_params() for exact transforms used.
    """

    def __init__(self, phys_params):
        """
        Initialize VCSEL object.

        Parameters
        ----------
        phys_params : dict
            Dictionary containing physical parameters (SI units) and optional
            simulation control parameters. Expected keys include:
              - tau_p: photon lifetime (s)
              - tau_n: carrier lifetime (s)
              - g0: differential gain (units of 1/(carriers * s))
              - N0: transparency carrier number (carriers)
              - s: gain saturation coefficient (1/photon)
              - beta: spontaneous emission factor (dimensionless)
              - kappa_c_mat: coupling strength (can be scalar or time-dependent array) [rad/s]
              - phi_p_mat: coupling phase (scalar or array) [rad]
              - I: injection current (A)
              - q: electron charge (C)
              - alpha: linewidth enhancement factor (dimensionless)
              - delta: frequency detuning (rad/s)
              - coupling: inter-laser coupling coefficient (dimensionless, default 1.0)
              - self_feedback: self-feedback coefficient (dimensionless, default 0.0)
              - noise_amplitude: noise scaling prefactor (dimensionless, default 0.0)
              - dt: timestep for integration (s)
              - Tmax: total integration time (s)
              - tau: feedback/coupling delay (s)
            Additional keys such as 'steps', 'delay_steps' can be supplied but
            are computed if absent.

        After initialization, call scale_params() to compute nd (nondimensional
        parameters) required by the integrator.
        """
        self.phys = phys_params
        self.nd = None  # will hold nondimensional parameters after scale_params()

    # -------------------------------------------------------------------------
    # Parameter Scaling
    # -------------------------------------------------------------------------
    def scale_params(self):
        """
        Convert provided physical parameters into nondimensional form.

        Purpose
        -------
        Many of the VCSEL model parameters can be very large or very small in SI
        units. This routine nondimensionalizes time, carrier and photon
        variables so the numerical integrator works with values of order one.

        Conventions & transforms
        -----------------------
        - nondimensional time: t' = t / tau_p  (photon lifetime used as time scale)
        - nondimensional carrier offset:
             n' = g0 * tau_p * (N - N0 - 1/(g0*tau_p))
             (so n + n0 + 1 = g0 * tau_p * N for noise prefactors)
        - nondimensional photon number:
             s' = g0 * tau_n * S
        - gain parameter: Gs = g0 * tau_p
        - p: normalized pump parameter computed from I, tau_n, q, etc.

        Returns
        -------
        nd : dict
            Dictionary containing nondimensional parameters required by the
            rest of the code (stored also on self.nd). Keys include:
              - tau: nondimensional delay (tau / tau_p)
              - tau_p, T (tau_n/tau_p), s (rescaled), kappa (rescaled by gamma)
              - p, alpha, phi_p, n0, nbar, sbar, Gs, beta_n, beta_const, delta_p
              - coupling, self_feedback, noise_amplitude
              - dt, Tmax, steps, delay_steps (time discretization info)
        """
        phys = self.phys

        # Extract fundamental physical parameters
        tau_p = phys['tau_p']
        tau_n = phys['tau_n']
        g0 = phys['g0']
        N0 = phys['N0']
        s = phys['s']
        beta = phys['beta']
        kappa = phys['kappa_c_mat']
        phi_p = phys['phi_p_mat']
        alpha = phys['alpha']
        delta = phys['delta']
        current_I = phys.get('I', None)
        q = phys['q']
        coupling = phys.get('coupling', 1.0)
        self_feedback = phys.get('self_feedback', 0.0)
        noise_amplitude = phys.get('noise_amplitude', 0.0)
        dt = phys.get('dt', 1e-12)
        Tmax = phys.get('Tmax', 3e-6)
        tau = phys.get('tau', 1e-9)  # feedback delay in seconds
        save_every = phys.get('save_every', 1)
        max_output_gb = phys.get('max_output_gb', 10.0)

        # Derived discrete parameters
        steps = int(Tmax / dt)
        delay_steps = int(tau / dt)

        injection = phys.get('injection', False)
        injection_topology = phys.get('injection_topology', None)
        injected_strength = phys.get('injected_strength', 0.0)
        # injected_phase_diff is handled in nd when injection is enabled
        kappa_inj = phys.get('kappa_injection', 0.0)

        # ------------------------------
        # Compute free-running steady-state in physical units
        # ------------------------------
        # Solve the laser rate equations for steady-state carrier number N_bar
        # and photon number S_bar (free-running, no coupling/self-feedback).
        # Use sympy to get numerical solution of the algebraic steady-state system.
        N_bar_sym = symbols('N_bar', real=True)
        S_bar = symbols('S_bar', real=True,positive=True) 

        # ------------------------------
        # Build nondimensional parameter dictionary
        # ------------------------------
        gamma = 1.0 / tau_p
        gamma_e = 1.0 / tau_n

        nd = {}
        nd['tau'] = gamma * tau  # nondimensional delay = tau / tau_p
        nd['tau_p'] = tau_p
        nd['T'] = tau_n / tau_p  # ratio of carrier to photon lifetimes
        # Rescale saturation to nondimensional units used in equations
        nd['s'] = s * gamma_e / g0
        nd['kappa'] = kappa / gamma  # nondimensional coupling (kappa / gamma)
        # Pump parameter p (normalized excess pump above threshold)
        p = g0 * tau_p * (current_I * tau_n / (q) - N0) - 1
        nd['p'] = p
        nd['alpha'] = alpha
        nd['phi_p'] = phi_p
        nd['n0'] = g0 * tau_p * N0  # nondimensional carrier offset
        # nondimensional steady-state carrier and photon numbers
        # n_bar = (N_bar - N0 - 1.0 / (g0 * tau_p)) / (1.0 / (g0 * tau_p))
        
        nd['Gs'] = g0 * tau_p  # g0 * tau_p, nondimensional gain scale
        nd['beta_n'] = beta
        # beta_const appears in the photon equation as an additive offset
        nd['beta_const'] = beta * (g0 * tau_p * N0 + 1.0)
        nd['delta_p'] = delta * tau_p  # nondimensional detuning (rad normalized by gamma)
        nd['coupling'] = coupling
        nd['self_feedback'] = self_feedback
        nd['noise_amplitude'] = noise_amplitude
        nd['save_every'] = save_every
        nd['max_output_gb'] = max_output_gb
        # nondimensional time step and total time
        nd['dt'] = dt / tau_p
        nd['Tmax'] = Tmax / tau_p
        nd['steps'] = steps
        nd['delay_steps'] = delay_steps
        nd['N_lasers'] = phys.get('N_lasers', 2)



        eq1 = Eq((1/nd['T']) * (nd['p'] - N_bar_sym - (1.0 + N_bar_sym) * S_bar / (1.0 + nd['s'] * S_bar)), 0)
        eq2 = Eq(((1.0 + N_bar_sym) / (1.0 + nd['s'] * S_bar) - 1.0) * S_bar + nd['beta_n'] * N_bar_sym + nd['beta_const'], 0)

        solution = solve((eq1, eq2), (N_bar_sym, S_bar), dict=True)

        # Convert symbolic solution to floats. We expect at least one physical root.
        n_bar = float(solution[0][N_bar_sym].evalf())
        S_bar = float(solution[0][S_bar].evalf())


        nd['nbar'] = n_bar
        nd['sbar'] = S_bar
        nd['injection'] = injection
        nd['injection_topology'] = injection_topology
        nd['injected_strength'] = injected_strength
        nd['injected_frequency'] = phys.get('injected_frequency', 0.0)* 1e9*(2*np.pi*tau_p)
        nd['kappa_inj'] = kappa_inj / gamma

        self.nd = nd
        return nd



    def f_nd(self, x, x_tau, x_2tau, j, phi_p, nd=None):
        """
        Fully vectorized N-laser nondimensional VCSEL array rate equations.

        STATES:
            For N lasers, x has shape (n_cases, 3N)
            ordering = [n0, S0, phi0, n1, S1, phi1, ..., n(N-1), S(N-1), phi(N-1)]

        RETURNS:
            out: (n_cases, 3N)
            same ordering for [dn_i, dS_i, dphi_i] for i=0..N-1
        """

        if nd is None:
            nd = self.nd

        n_cases, total = x.shape
        assert total % 3 == 0
        N = total // 3

        # Reshape states for vectorized operations
        X      = x.reshape(n_cases, N, 3)
        X_tau  = x_tau.reshape(n_cases, N, 3)
        X_2tau = x_2tau.reshape(n_cases, N, 3)

        n   = X[:, :, 0]
        S   = X[:, :, 1]
        phi = X[:, :, 2]

        S_t   = X_tau[:, :, 1]
        phi_t = X_tau[:, :, 2]

        S_2t   = X_2tau[:, :, 1]
        phi_2t = X_2tau[:, :, 2]

        # Parameters
        T      = nd["T"]
        s      = nd["s"]
        nbar   = nd["nbar"]
        p      = nd["p"]
        beta_n = nd["beta_n"]
        beta_c = nd["beta_const"]
        alpha  = nd["alpha"]

        delta = np.array(nd["delta_p"]).reshape(1, N)  # shape (1,N) broadcasts across cases

        # φ_p must be (N, N)
        if np.ndim(phi_p) == 0:
            # Account for scalar phi_p
            phi_p = phi_p * np.ones((N, N))[None,:,:]
        elif np.ndim(phi_p) == 2:
            # Account for matrix phi_p (one case)
            phi_p = np.array(phi_p)[None,:,:]
            if phi_p.shape != (1,N, N):
                raise ValueError("phi_p must be scalar or shape (N,N)")
        elif np.ndim(phi_p) == 3:
            # Account for multiple phi_p cases (n_cases, N, N)
            phi_p = np.array(phi_p)
            if phi_p.shape != (n_cases, N, N):
                raise ValueError(f"phi_p must be scalar or shape (N,N) or (n_cases,N,N), got shape {phi_p.shape}")
            
        phi_p_self = phi_p[:, np.arange(N), np.arange(N)]
        phi_p_mutual = phi_p - np.diag(phi_p_self[0,:])[None,:,:]

        # Time-varying coupling
        kappa_val = np.asarray(nd["kappa"])
        if kappa_val.ndim == 3:
            kappa_mat = np.array(kappa_val[j])        # shape (N,N)
        elif kappa_val.ndim == 2:
            kappa_mat = np.array(kappa_val)           # constant (N,N)
        else:
            kappa_mat = np.ones((N, N)) * kappa_val   # scalar
        kappa_diag = np.diag(kappa_mat)             # self-feedback (length N)
        kappa_mat = kappa_mat - np.diag(kappa_diag) # zero diagonal for mutual coupling


        # Keep photon numbers positive
        eps = 1e-11
        S   = np.maximum(S, eps)
        S_t = np.maximum(S_t, eps)
        S_2t = np.maximum(S_2t, eps)

        sqrtS   = np.sqrt(S)
        sqrtS_t = np.sqrt(S_t)
        sqrtS_2t = np.sqrt(S_2t)

        # Carrier dynamics
        denom = 1 + s*S
        dn = (1/T)*(p - n - (1+n)*S/denom)
        dS = ((1+n)/denom - 1)*S + beta_n*n + beta_c
        dphi = (alpha*0.5)*(n - nbar)/denom + delta

        # ----------------- MUTUAL COUPLING -----------------
        # shape: (n_cases, N, N)
        if nd['coupling'] > 0.0:
            mutual_amp = sqrtS[:, :, None] * sqrtS_t[:, None, :]
            phi_diff_mutual = phi_t[:, None, :] - phi[:, :, None] - phi_p_mutual
            cos_mutual = np.cos(phi_diff_mutual)
            sin_mutual = np.sin(phi_diff_mutual)

            # intensity dynamics
            dS += nd['coupling'] * 2 * np.sum(kappa_mat[None, :, :] * mutual_amp * cos_mutual, axis=2)

            # phase dynamics
            ratio = sqrtS_t[:, None, :] / sqrtS[:, :, None]
            ratio = np.clip(ratio, 0, 10)
            dphi +=nd['coupling'] *  np.sum(kappa_mat[None, :, :] * (ratio) * sin_mutual, axis=2)

        # ----------------- SELF-FEEDBACK -----------------

        if nd["self_feedback"] > 0.0:
            self_amp = sqrtS[:, :] * sqrtS_2t[:, :]
            phi_diff_self = phi_2t - phi - 2*phi_p_self
            cos_self = np.cos(phi_diff_self)
            sin_self = np.sin(phi_diff_self)
            dS += nd["self_feedback"] * 2 * kappa_diag[None, :] * self_amp * cos_self
            dphi += nd["self_feedback"] * kappa_diag[None, :] * ((sqrtS_2t / sqrtS) * sin_self)

        # ----------------- INJECTION (optional) -----------------
        injection_topology = nd.get("injection_topology", None)
        if nd.get("injection", False) and injection_topology is not None:

            assert injection_topology.shape == (N,), "injection_topology must be shape (N,) indicating which lasers receive injection."

            inj_strength = nd["injected_strength"]
            inj_phase    = nd["injected_phase_diff"]
            omega_src = nd["injected_frequency"]
            if isinstance(omega_src, np.ndarray):
                omega_inj = omega_src[:, j] if omega_src.ndim == 2 else omega_src[j]
            else:
                omega_inj = omega_src
            kappa_inj    = nd["kappa_inj"].T[j]
            t = j*nd["dt"]

            for p in range(N):

                if not injection_topology[p]:
                    continue

                S0   = S[:, p]
                phi0 = phi[:, p]

                inj_cos = np.cos(omega_inj*t + inj_phase - phi0)
                inj_sin = np.sin(omega_inj*t + inj_phase - phi0)
                dS[:, p]  += 2*kappa_inj*np.sqrt(S0*inj_strength)*inj_cos
                dphi[:, p] +=     kappa_inj*np.sqrt(inj_strength/S0)*inj_sin
        elif nd.get("injection", False) and injection_topology is None:
            raise ValueError("Injection topology must be provided when injection is enabled.")

        dphi[S < 1e-8] = 0
        # ----------------- PACK OUTPUT -----------------
        out = np.empty((n_cases, 3*N))
        out[:, 0::3] = dn
        out[:, 1::3] = dS
        out[:, 2::3] = dphi



        return out


    def compute_noise_sample(self, y_c, noise_amplitude, dt, nd):
        """
        Generate Langevin noise samples for N coupled VCSELs.
        
        State ordering:
            [n1, S1, φ1,  n2, S2, φ2, ...,  nN, SN, φN]
            shape = (n_cases, 3*N)

        Parameters
        ----------
        y_c : np.ndarray, shape (n_cases, 3*N) or (3*N,)
        noise_amplitude : float
        dt : float
        nd : dict
            Must contain keys: n0, T, beta_n, Gs

        Returns
        -------
        noise : np.ndarray, same shape as y_c
        """
        if not noise_amplitude:
            return np.zeros_like(y_c)

        # Ensure 2D
        y_c = np.atleast_2d(y_c)
        n_cases, total_dim = y_c.shape
        N = total_dim // 3   # number of lasers

        # Extract parameters
        n0 = nd['n0']
        T = nd['T']
        beta = nd['beta_n']
        Gs = nd['Gs']

        eps = 1e-12

        # --- Extract state components ---
        n = y_c[:, 0::3]                    # (n_cases, N)
        S = np.maximum(y_c[:, 1::3], eps)   # (n_cases, N)

        # --- Gaussian random numbers ---
        # For each laser: (Fn, Fs, Fphi)
        Z = np.random.randn(n_cases, 3*N)

        # Split into per-laser blocks
        Z_fn  = Z[:, 0::3]    # (n_cases, N)
        Z_fs  = Z[:, 1::3]
        Z_fphi = Z[:, 2::3]

        # --- Prefactors ---
        scale = noise_amplitude * np.sqrt(dt)

        pref_Fn  = np.sqrt(2.0 * Gs * (n + n0 + 1.0) / T)            # Fn
        pref_FsN = np.sqrt(2.0 * beta * (n + n0 + 1.0) * S / T**2)   # Fs correlated contribution
        pref_Fs  = np.sqrt(2.0 * beta * (n + n0 + 1.0) * S)          # Fs
        pref_Fp  = np.sqrt(beta * (n + n0 + 1.0) / (2.0 * S))        # Fphi

        # --- Construct correlated increments ---
        Fn_inc  = (pref_Fn * Z_fn - pref_FsN * Z_fs) * scale
        Fs_inc  = (pref_Fs * Z_fs) * scale
        Fp_inc  = (pref_Fp * Z_fphi) * scale

        # --- Interleave back to (n_cases, 3*N) ---
        noise = np.zeros_like(y_c)
        noise[:, 0::3] = Fn_inc
        noise[:, 1::3] = Fs_inc
        noise[:, 2::3] = Fp_inc

        # Return row if input was 1D
        return noise[0] if y_c.shape[0] == 1 else noise
    

    def integrate(self, history, nd=None, progress=False, theta=0.5, max_iter=5, smooth_freqs=True):
        """
        Integrate the nondimensional DDE system with a trapezoidal predictor-corrector.

        Parameters
        ----------
        history : np.ndarray, shape (n_cases, 3*N_lasers, 2*delay_steps)
            Initial history over [-2*tau, 0] in nondimensional units.
        nd : dict or None
            Nondimensional parameters (if None uses self.nd).
        progress : bool
            If True, show a tqdm progress bar.
        theta : float
            Trapezoidal blend parameter (0.5 is Crank-Nicolson).
        max_iter : int
            Maximum fixed-point iterations per step.

        Returns
        -------
        t_dim : np.ndarray
            Dimensional time array (seconds).
        y : np.ndarray
            State time series, shape (n_cases, 3*N_lasers, steps).
        freqs : np.ndarray
            Instantaneous phase derivatives, shape (n_cases, N_lasers, steps).
        smooth_freqs : bool
            If True (default), apply a moving-average window over one delay to
            smooth the returned frequency estimates.
        """

        if nd is None:
            nd = self.nd

        n_cases = history.shape[0]
        N_lasers = nd['kappa'].shape[-1]
        dt = nd['dt']
        steps = nd['steps']
        delay_steps = nd['delay_steps']
        noise_amplitude = nd['noise_amplitude']
        phi_p = nd['phi_p']
        save_every = int(max(1, nd.get('save_every', 1)))
        smooth_window_delays = nd.get("smooth_window_delays", 1.0)
        delay_interp = nd.get('delay_interp', None)
        max_output_gb = nd.get('max_output_gb', None)
        if max_output_gb is not None:
            bytes_per_step = n_cases * (4 * N_lasers) * 8
            est_bytes = bytes_per_step * steps
            max_bytes = max_output_gb * 1e9
            save_every_req = int(np.ceil(est_bytes / max_bytes)) if est_bytes > max_bytes else 1
            save_every = max(save_every, save_every_req)
            nd['save_every'] = save_every
            if save_every_req > 1:
                est_gb = est_bytes / 1e9
                print(
                    f"Estimated output size ~{est_gb:.2f} GB; "
                    f"using save_every={save_every} to cap output."
                )

        # Require full delay history
        if history.shape[2] < 2 * delay_steps:
            raise ValueError("history too short for configured delay_steps")

        use_stream = save_every > 1
        if use_stream:
            buffer_len = 2 * delay_steps + 1
            y_buf = np.zeros((n_cases, 3 * N_lasers, buffer_len))
            freqs_buf = np.zeros((n_cases, N_lasers, buffer_len))
            y_buf[:, :, :2 * delay_steps] = history[:, :, :2 * delay_steps]
            keep_idx = []
            y_keep = []
            f_keep = []

            def should_store(idx):
                return idx % save_every == 0

            for idx in range(min(2 * delay_steps, steps)):
                if should_store(idx):
                    keep_idx.append(idx)
                    y_keep.append(history[:, :, idx].copy())
                    f_keep.append(np.zeros((n_cases, N_lasers)))
            keep_pos = {idx: pos for pos, idx in enumerate(keep_idx)}
        else:
            # ----------------- allocate outputs -----------------
            y = np.zeros((n_cases, 3*N_lasers, steps))
            freqs = np.zeros((n_cases, N_lasers, steps))
            y[:, :, :2*delay_steps] = history[:, :, :2*delay_steps]

        start_idx = 2 * delay_steps - 1
        if start_idx >= steps:
            t_dim = np.arange(steps) * dt * nd['tau_p']
            if use_stream:
                return t_dim, history[:, :, :steps], np.zeros((n_cases, N_lasers, steps))
            return t_dim, y, freqs

        # ----------------- precompute indices -----------------
        idx_tau_arr = np.arange(steps) - delay_steps
        idx_2tau_arr = np.arange(steps) - 2 * delay_steps
        idx_tau_arr[idx_tau_arr < 0] = 0
        idx_2tau_arr[idx_2tau_arr < 0] = 0
        if delay_interp in ("linear", "cubic"):
            tau_steps = nd["tau"] / dt
            k0 = int(np.floor(tau_steps))
            frac = tau_steps - k0
            tau2_steps = 2.0 * nd["tau"] / dt
            k0_2 = int(np.floor(tau2_steps))
            frac2 = tau2_steps - k0_2

        S_idx = np.arange(N_lasers) * 3 + 1
        phi_idx = np.arange(N_lasers) * 3 + 2

        # ----------------- initialize frequency history -----------------
        hist_len = min(2 * delay_steps, steps)
        for n in range(hist_len):
            if use_stream:
                y_n = y_buf[:, :, n % buffer_len]
                y_tau = y_buf[:, :, idx_tau_arr[n] % buffer_len]
                y_2tau = y_buf[:, :, idx_2tau_arr[n] % buffer_len]
            else:
                y_n = y[:, :, n]
                y_tau = y[:, :, idx_tau_arr[n]]
                y_2tau = y[:, :, idx_2tau_arr[n]]

            f_n = self.f_nd(
                y_n, y_tau, y_2tau, n, phi_p, nd
            )[:, :3*N_lasers]

            deriv_n = f_n
            if use_stream:
                freqs_buf[:, :, n % buffer_len] = deriv_n[:, phi_idx]
                pos = keep_pos.get(n)
                if pos is not None:
                    f_keep[pos] = deriv_n[:, phi_idx].copy()
            else:
                freqs[:, :, n] = deriv_n[:, phi_idx]

        # ----------------- noise normalization -----------------
        if noise_amplitude is None:
            noise_arr = None
            noise_mode = "none"
        elif callable(noise_amplitude):
            noise_arr = noise_amplitude
            noise_mode = "callable"
        elif np.isscalar(noise_amplitude):
            noise_arr = noise_amplitude
            noise_mode = "scalar"
        else:
            noise_arr = np.asarray(noise_amplitude)
            noise_mode = "array"

        # ----------------- work buffers -----------------
        y_guess = np.empty((n_cases, 3*N_lasers))
        y_new   = np.empty_like(y_guess)
        tmp_noise = np.empty_like(y_guess)
        noise_substeps = int(max(1, nd.get("noise_substeps", 1)))

        # ----------------- main loop -----------------
        with tqdm(total=steps - 1 - start_idx,
                desc="Integrating",
                disable=not progress) as pbar:

            for n in range(start_idx, steps - 1):

                if use_stream:
                    y_n = y_buf[:, :, n % buffer_len]
                    if delay_interp in ("linear", "cubic"):
                        idx_hi = max(n - k0, 0)
                        idx_lo = max(n - k0 - 1, 0)
                        y_tau_hi = y_buf[:, :, idx_hi % buffer_len]
                        y_tau_lo = y_buf[:, :, idx_lo % buffer_len]
                        if delay_interp == "cubic":
                            idx_p0 = max(idx_lo - 1, 0)
                            idx_p3 = max(idx_hi + 1, 0)
                            p0 = y_buf[:, :, idx_p0 % buffer_len]
                            p1 = y_tau_lo
                            p2 = y_tau_hi
                            p3 = y_buf[:, :, idx_p3 % buffer_len]
                            t = 1.0 - frac
                            t2 = t * t
                            t3 = t2 * t
                            y_tau = 0.5 * (
                                (2.0 * p1)
                                + (-p0 + p2) * t
                                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                            )
                        else:
                            y_tau = (1.0 - frac) * y_tau_hi + frac * y_tau_lo

                        idx_hi2 = max(n - k0_2, 0)
                        idx_lo2 = max(n - k0_2 - 1, 0)
                        y_2tau_hi = y_buf[:, :, idx_hi2 % buffer_len]
                        y_2tau_lo = y_buf[:, :, idx_lo2 % buffer_len]
                        if delay_interp == "cubic":
                            idx_p0_2 = max(idx_lo2 - 1, 0)
                            idx_p3_2 = max(idx_hi2 + 1, 0)
                            p0 = y_buf[:, :, idx_p0_2 % buffer_len]
                            p1 = y_2tau_lo
                            p2 = y_2tau_hi
                            p3 = y_buf[:, :, idx_p3_2 % buffer_len]
                            t = 1.0 - frac2
                            t2 = t * t
                            t3 = t2 * t
                            y_2tau = 0.5 * (
                                (2.0 * p1)
                                + (-p0 + p2) * t
                                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                            )
                        else:
                            y_2tau = (1.0 - frac2) * y_2tau_hi + frac2 * y_2tau_lo
                    else:
                        idx_tau = idx_tau_arr[n + 1]
                        idx_2tau = idx_2tau_arr[n + 1]
                        y_tau = y_buf[:, :, max(idx_tau, 0) % buffer_len]
                        y_2tau = y_buf[:, :, max(idx_2tau, 0) % buffer_len]
                else:
                    y_n = y[:, :, n]
                    if delay_interp in ("linear", "cubic"):
                        idx_hi = max(n - k0, 0)
                        idx_lo = max(n - k0 - 1, 0)
                        y_tau_hi = y[:, :, idx_hi]
                        y_tau_lo = y[:, :, idx_lo]
                        if delay_interp == "cubic":
                            idx_p0 = max(idx_lo - 1, 0)
                            idx_p3 = max(idx_hi + 1, 0)
                            p0 = y[:, :, idx_p0]
                            p1 = y_tau_lo
                            p2 = y_tau_hi
                            p3 = y[:, :, idx_p3]
                            t = 1.0 - frac
                            t2 = t * t
                            t3 = t2 * t
                            y_tau = 0.5 * (
                                (2.0 * p1)
                                + (-p0 + p2) * t
                                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                            )
                        else:
                            y_tau = (1.0 - frac) * y_tau_hi + frac * y_tau_lo
                        idx_hi2 = max(n - k0_2, 0)
                        idx_lo2 = max(n - k0_2 - 1, 0)
                        y_2tau_hi = y[:, :, idx_hi2]
                        y_2tau_lo = y[:, :, idx_lo2]
                        if delay_interp == "cubic":
                            idx_p0_2 = max(idx_lo2 - 1, 0)
                            idx_p3_2 = max(idx_hi2 + 1, 0)
                            p0 = y[:, :, idx_p0_2]
                            p1 = y_2tau_lo
                            p2 = y_2tau_hi
                            p3 = y[:, :, idx_p3_2]
                            t = 1.0 - frac2
                            t2 = t * t
                            t3 = t2 * t
                            y_2tau = 0.5 * (
                                (2.0 * p1)
                                + (-p0 + p2) * t
                                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                            )
                        else:
                            y_2tau = (1.0 - frac2) * y_2tau_hi + frac2 * y_2tau_lo
                    else:
                        y_tau = y[:, :, idx_tau_arr[n + 1]]
                        y_2tau = y[:, :, idx_2tau_arr[n + 1]]

                # RHS at current step
                f_n = self.f_nd(
                    y_n, y_tau, y_2tau, n, phi_p, nd
                )[:, :3*N_lasers]

                # initial guess (explicit Euler)
                y_guess[:] = y_n
                y_guess += dt * f_n

                # fixed-point iteration
                for _ in range(max_iter):

                    f_guess = self.f_nd(y_guess, y_tau, y_2tau, n + 1, phi_p, nd)[:, :3*N_lasers]

                    y_new[:] = y_n
                    y_new += dt * ((1.0 - theta) * f_n + theta * f_guess)

                    if np.max(np.abs(y_new - y_guess)) < 1e-10:
                        break

                    y_guess[:] = y_new

                y_c = y_guess

                # derivative used for frequency
                deriv_n = (1.0 - theta) * f_n + theta * f_guess
                if use_stream:
                    freqs_buf[:, :, (n + 1) % buffer_len] = deriv_n[:, phi_idx]
                else:
                    freqs[:, :, n + 1] = deriv_n[:, phi_idx]

                # Euler–Maruyama noise (optionally substepped)
                if noise_mode == "none":
                    y_next = y_c
                else:
                    if noise_mode == "scalar":
                        na = noise_arr
                    elif noise_mode == "callable":
                        na = noise_arr(n * dt * nd["tau_p"])
                    else:
                        na = noise_arr[n]

                    if noise_substeps == 1:
                        tmp_noise[:] = self.compute_noise_sample(
                            y_c, na, dt, nd
                        )
                        y_next = y_c + tmp_noise
                        y_next[:, S_idx] = np.maximum(y_next[:, S_idx], 1e-11)
                    else:
                        y_next = y_c.copy()
                        dt_sub = dt / noise_substeps
                        for _ in range(noise_substeps):
                            tmp_noise[:] = self.compute_noise_sample(
                                y_next, na, dt_sub, nd
                            )
                            y_next += tmp_noise
                            y_next[:, S_idx] = np.maximum(y_next[:, S_idx], 1e-11)

                if use_stream:
                    y_buf[:, :, (n + 1) % buffer_len] = y_next
                    if should_store(n + 1):
                        keep_idx.append(n + 1)
                        y_keep.append(y_next.copy())
                        f_keep.append(freqs_buf[:, :, (n + 1) % buffer_len].copy())
                else:
                    y[:, :, n + 1] = y_next

                if progress:
                    pbar.update(1)

        if use_stream:
            keep_idx = np.array(keep_idx, dtype=int)
            order = np.argsort(keep_idx)
            keep_idx = keep_idx[order]
            y_out = np.stack([y_keep[i] for i in order], axis=2)
            freqs_out = np.stack([f_keep[i] for i in order], axis=2)
            t_dim = keep_idx * dt * nd['tau_p']

            window = max(1, int(round(delay_steps * smooth_window_delays / save_every)))
            window = min(window, freqs_out.shape[2])
            if smooth_freqs and window > 1:
                kernel = np.ones(window) / window
                pad = window // 2
                freqs_sm = np.empty_like(freqs_out)
                for i in range(n_cases):
                    for j in range(N_lasers):
                        series = freqs_out[i, j, :]
                        if pad > 0:
                            series = np.pad(series, (pad, pad), mode="edge")
                        smoothed = np.convolve(series, kernel, mode="valid")
                        freqs_sm[i, j, :] = smoothed[:freqs_out.shape[2]]
                        if pad > 0:
                            freqs_sm[i, j, -pad:] = freqs_sm[i, j, -pad - 1]
                freqs_out = freqs_sm

            return t_dim, y_out, freqs_out

        if smooth_freqs:
            window = max(1, int(round(delay_steps * smooth_window_delays)))
            window = min(window, steps)
            kernel = np.ones(window) / window
            freqs_sm = np.empty_like(freqs)
            pad = window // 2
            for i in range(n_cases):
                for j in range(N_lasers):
                    series = freqs[i, j, :]
                    if pad > 0:
                        series = np.pad(series, (pad, pad), mode="edge")
                    smoothed = np.convolve(series, kernel, mode="valid")
                    freqs_sm[i, j, :] = smoothed[:steps]
                    if pad > 0:
                        freqs_sm[i, j, -pad:] = freqs_sm[i, j, -pad - 1]
            freqs = freqs_sm

        t_dim = np.arange(steps) * dt * nd['tau_p']
        return t_dim, y, freqs


    
    


    

    # -------------------------------------------------------------------------
    # Static Utility: Cosine Ramp
    # -------------------------------------------------------------------------

    @staticmethod
    def cosine_ramp(t, t_start, rise_10_90, kappa_initial=0.0, kappa_final=1.0):
        """
        Half-cosine ramp utility with zero slope at start and end.

        This constructs a smooth ramp that transitions from kappa_initial to
        kappa_final. The supplied rise_10_90 is interpreted as the duration
        between the 10% and 90% points of the ramp; internally the function
        uses total_duration = rise_10_90 / 0.8 so the full ramp goes from 0%
        to 100% over this total duration.

        Parameters
        ----------
        t : array_like
            Time array (same units as t_start and rise_10_90).
        t_start : float
            Time where the ramp begins.
        rise_10_90 : float
            Duration between the 10% and 90% points of the ramp.
        kappa_initial : float or np.ndarray, optional
            Starting value(s) of the ramp. Can be a scalar or an array
            (e.g., matrix) (default 0.0).
        kappa_final : float or np.ndarray, optional
            Final value(s) of the ramp. Can be a scalar or an array
            (e.g., matrix) (default 1.0).

        Returns
        -------
        ramped : np.ndarray
            Array containing the ramped values. The shape of the output is
            determined by NumPy's broadcasting rules for the shapes of t,
            kappa_initial, and kappa_final.
        """
        total_duration = rise_10_90 / 0.8
        # u is the normalized time (0.0 to 1.0)
        u = np.clip((np.asarray(t) - t_start) / total_duration, 0.0, 1.0)
        
        # ramp is the normalized ramp value (0.0 to 1.0)
        ramp = 0.5 - 0.5 * np.cos(np.pi * u)
        
        return kappa_initial + (kappa_final - kappa_initial) * ramp


    def generate_history(self, nd=None, shape='FR', n_cases=1, counts=None, guesses=None):
        """
        Produce an initial history array required by the delay integrator.

        The integrator expects inputs for times t in [-2*tau, 0]. This routine
        returns arrays shaped (n_cases, 3*N_lasers, 2*delay_steps) in
        nondimensional units.

        Parameters
        ----------
        nd : dict or None
            Nondimensional parameter dictionary (if None uses self.nd).
        shape : str
            One of:
              - 'FR' : free-running steady-state history (steady carriers and
                       steady photon numbers, linear phase ramp from detuning)
              - 'ZF' : zero-field history (all fields zero except phase ramp)
              - 'EQ' : equilibrium histories for all steady-state solutions
        n_cases : int
            Number of independent cases to tile the history for.
        counts : dict or None
            Optional override for phase and frequency seed counts when
            shape='EQ'.
        guesses : list of np.ndarray or None
            Optional equilibrium guesses passed to solve_equilibria when
            shape='EQ'.

        Returns
        -------
        history : np.ndarray, shape (n_cases, 3*N_lasers, 2*delay_steps)
            History suitable as input to integrate().
        freq_hist : np.ndarray, shape (n_cases, N_lasers, 2*delay_steps)
            Frequency history (rad/s) inferred from the detuning or equilibrium.
        eq : np.ndarray or None
            Final equilibrium when shape='EQ', otherwise None.
        results : np.ndarray or None
            All equilibrium solutions (None for non-'EQ' shapes).

        Notes
        -----
        This method always returns four values. For non-'EQ' shapes, results
        is None.
        """
        if nd is None:
            nd = self.nd
        ds = nd['delay_steps']
        length = 2 * ds
        eq = None
        results = None

        N_lasers = nd.get('N_lasers', 2)

        if shape == 'FR':
            # Free-running steady-state: steady carriers and photons; phi2 may
            # accumulate according to detuning.
            hist = np.zeros((3*N_lasers, length))
            hist[0::3, :] = nd['nbar']         # n1
            hist[1::3, :] = nd['sbar']         # S1 (nondimensional)
            freq_hist = np.zeros((N_lasers, 2*nd['delay_steps']))

            if np.shape(nd['delta_p']) == ():
                raise ValueError("nd['delta_p'] must be array-like for multiple lasers")
            elif np.shape(nd['delta_p']) != (N_lasers,):
                raise ValueError(f"nd['delta_p'] must have shape ({N_lasers},) for {N_lasers} lasers")
            else:
                for p, delta_p in enumerate(np.array(nd['delta_p'])):
                    hist[3*p+2, :] = delta_p * np.arange(length) * nd['dt']  
                    freq_hist[p, :] = delta_p * np.ones(length) / (2*np.pi*1e9*nd['tau_p'])  # convert back to rad/s


        elif shape == 'ZF':
            # Zero-field initial condition: zero photons and zero carriers
            hist = np.zeros((3*N_lasers, length))
            hist[0::3, :] = 0.0         # n1
            hist[1::3, :] = 0.0         # S1 (nondimensional)
            freq_hist = np.zeros((N_lasers, 2*nd['delay_steps']))

            if np.shape(nd['delta_p']) == ():
                raise ValueError("nd['delta_p'] must be array-like for multiple lasers")
            elif np.shape(nd['delta_p']) != (N_lasers,):
                raise ValueError(f"nd['delta_p'] must have shape ({N_lasers},) for {N_lasers} lasers")
            else:
                for p, delta_p in enumerate(np.array(nd['delta_p'])):
                    hist[3*p+2, :] = delta_p * np.arange(length) * nd['dt']  
                    freq_hist[p, :] = delta_p * np.ones(length) / (2*np.pi*1e9*nd['tau_p'])  # convert back to rad/s
        elif shape == 'EQ':
            # Steady-state equilibrium histories for all solutions.
            N_lasers = nd.get('N_lasers', 2)
            eq, results, _ = self.solve_equilibria(nd=nd, counts=counts, guesses=guesses)

            if results is None or len(results) == 0:
                hist = np.zeros((0, 3 * N_lasers, length))
                freq_hist = np.zeros((0, N_lasers, length))
                return hist, freq_hist, eq, results

            n_cases_out = results.shape[0]
            kappa = nd.get("kappa", None)
            if isinstance(kappa, (list, tuple, np.ndarray)):
                kappa_arr = np.array(kappa)
                if kappa_arr.ndim == 3:
                    nd["kappa"] = np.repeat(kappa_arr[-1][None, :, :], kappa_arr.shape[0], axis=0)
            phi_p = nd.get("phi_p", 0.0)
            if np.isscalar(phi_p):
                nd["phi_p"] = np.full((n_cases_out, N_lasers, N_lasers), phi_p)
            else:
                phi_p_arr = np.array(phi_p)
                if phi_p_arr.ndim == 2 and phi_p_arr.shape == (N_lasers, N_lasers):
                    nd["phi_p"] = np.tile(phi_p_arr[None, :, :], (n_cases_out, 1, 1))
                elif phi_p_arr.ndim == 3 and phi_p_arr.shape == (1, N_lasers, N_lasers):
                    nd["phi_p"] = np.tile(phi_p_arr, (n_cases_out, 1, 1))
                elif phi_p_arr.ndim == 3 and phi_p_arr.shape == (n_cases_out, N_lasers, N_lasers):
                    nd["phi_p"] = phi_p_arr
                else:
                    warnings.warn(
                        "phi_p has an unexpected shape; leaving nd['phi_p'] unchanged.",
                        RuntimeWarning,
                    )

            hist = np.zeros((n_cases_out, 3 * N_lasers, length))
            freq_hist = np.zeros((n_cases_out, N_lasers, length))
            t_arr = np.arange(length) * nd["dt"]

            for k, root in enumerate(results):
                omega = root[-1]
                phase_diff = np.concatenate([[0.0], root[2 * N_lasers:(3 * N_lasers - 1)]])[:N_lasers]

                hist[k, 0::3, :] = root[0::2][:N_lasers].reshape(-1, 1)
                hist[k, 1::3, :] = root[1::2][:N_lasers].reshape(-1, 1)
                hist[k, 2::3, :] = omega * t_arr
                hist[k, 2::3, :] += phase_diff.reshape(-1, 1)
                freq_hist[k, :, :] = omega / (2 * np.pi * 1e9 * nd["tau_p"])

            return hist, freq_hist, eq, results

        else:
            raise ValueError("unsupported history shape")

        # Tile for multiple cases
        history = np.tile(hist[np.newaxis, :, :], (n_cases, 1, 1))
        freq_hist = np.tile(freq_hist[np.newaxis, :, :], (n_cases, 1, 1))
        return history, freq_hist, eq, results


    def solve_one(self,resid, guess, nd, phi_p=None):

        sol = root(resid, guess, args=(nd,), tol=1e-10, method='hybr',)
        return sol

    def solve_equilibria(self, nd, phi_p=None, guesses=None, counts=None, n_jobs=-1):
        """
        Solve for steady-state roots of the coupled VCSEL system.

        Parameters
        ----------
        nd : dict
            Nondimensional parameters.
        guesses : list of np.ndarray
            Optional initial guesses in the form
            [S1, S2, ..., SN, phi_2, ..., phi_N, omega].
        counts : dict or None
            Optional override for grid and refinement settings:
              - phase_count: number of phase samples
              - freq_count: number of frequency samples
              - adaptive_grid: whether to refine the grid
              - refine_factor: multiplier for grid density per refinement
              - max_refine: number of refinement passes
        phi_p : unused
            Reserved for future phase-dependent seeding.

        Returns
        -------
        final_root : np.ndarray or None
            Selected equilibrium vector [n1, S1, ..., nN, SN, phi_2..phi_N, omega].
        results : np.ndarray
            Candidate equilibria with the same ordering as final_root.
        E_tot : np.ndarray or None
            Total field intensity used to select final_root.
        """
        results = []
        E_tot = None

        if nd is None:
            nd = self.nd

        if guesses is None:
            guesses = []

        N_lasers = nd['N_lasers']
        tau = nd['tau']


        # ----------------------------
        # Generate guesses
        # ----------------------------
        tol = 1e-10

        if counts:
            phase_count = counts.get('phase_count', 30)
            freq_count = counts.get('freq_count', 100)
            adaptive_grid = counts.get('adaptive_grid', True)
            refine_factor = counts.get('refine_factor', 2)
            max_refine = counts.get('max_refine', 3)
        else:
            phase_count = 30
            freq_count = 100
            adaptive_grid = True
            refine_factor = 2
            max_refine = 3

        base_omg = 10 * 2 * np.pi * 1e9 * nd['tau_p']
        if guesses:
            max_guess_omg = max(abs(g[-1]) for g in guesses)
        else:
            max_guess_omg = 0.0
        max_omg = max(base_omg, 2.0 * max_guess_omg)

        # Start from any user-supplied guesses, then add grid samples per refinement.
        seed_guesses = list(guesses)
        prev_count = -1
        final_root = None
        results = []
        E_tot = None
        # Each refinement multiplies the phase and frequency grid density.
        refine_steps = max_refine + 1 if adaptive_grid else 1
        for refine_idx in range(refine_steps):
            # Effective grid size for this refinement level.
            phase_count_eff = int(phase_count * (refine_factor ** refine_idx))
            freq_count_eff = int(freq_count * (refine_factor ** refine_idx))
            guesses_iter = list(seed_guesses)

            # Build the phase/omega grid and append to the current guess set.
            phase_vals = np.linspace(-np.pi, np.pi, phase_count_eff, endpoint=False)
            omega_seeds = np.linspace(-max_omg, max_omg, freq_count_eff)
            for phi in phase_vals:
                for omega_guess in omega_seeds:
                    guesses_iter.append(np.concatenate([
                        np.tile([nd['sbar']], N_lasers),
                        np.full(N_lasers-1, phi),
                        np.array([omega_guess])
                    ]))

            solutions = Parallel(n_jobs=n_jobs)(
                delayed(self.solve_one)(self.residuals, g, nd) for g in guesses_iter
            )

            # Collect candidate equilibria from this grid.
            results_iter = []
            for sol in solutions:
                if not sol.success:
                    continue
                res_norm = np.linalg.norm(sol.fun)
                if res_norm >= tol:
                    continue

                s = nd['s']
                p = nd['p']
                S = sol.x[:N_lasers]
                n = (1 + S/(1 + s * S))**(-1) * (p - S/(1 + s * S))

                solution = np.zeros(3*N_lasers)
                solution[0::2][:N_lasers] = n
                solution[1::2][:N_lasers] = S
                solution[2*N_lasers:3*N_lasers-1] = sol.x[N_lasers:2*N_lasers-1]
                solution[-1] = sol.x[-1]
                results_iter.append(solution)

            # Deduplicate and filter candidates using residuals and heuristics.
            if results_iter:
                unique = []
                results_arr = np.array(results_iter)
                for r in results_arr:
                    r[2*N_lasers:3*N_lasers-1] = r[2*N_lasers:3*N_lasers-1] % (2 * np.pi)
                    formatted_res = np.concatenate([
                        r[1::2][:N_lasers],
                        r[2*N_lasers:3*N_lasers-1],
                        [r[-1]]
                    ])
                    res_norm = np.linalg.norm(self.residuals(formatted_res, nd))
                    if res_norm >= 1e-6:
                        continue
                    if any(np.allclose(r, u, atol=1e-3, rtol=1e-3) for u in unique):
                        continue
                    if not np.all(r[1::2][:N_lasers] > 0.0) or not np.isfinite(r).all():
                        continue
                    photon_nums = r[1::2][:N_lasers]
                    epsilon = 1e-12
                    if photon_nums.max() / (photon_nums.min() + epsilon) > 10:
                        continue
                    unique.append(r)
                results_arr = np.array(unique)
            else:
                results_arr = np.array([])

            # Compute total field to pick a representative equilibrium.
            if results_arr.size > 0:
                results_arr[:, 2*N_lasers : 3*N_lasers - 1] = (
                    results_arr[:, 2*N_lasers : 3*N_lasers - 1] % (2.0 * np.pi)
                )
                omega = results_arr[:, -1]
                S = results_arr[:,1::2][:,:N_lasers]
                valid_indices = np.all(S > 0.0, axis=1)
                S = S[valid_indices]
                omega = omega[valid_indices]
                phi = np.zeros((len(S), N_lasers))
                phi[:, 1:] = results_arr[valid_indices, 2*N_lasers : 3*N_lasers - 1]
                E = np.sqrt(S) * np.exp(1j * (omega[:, None] * tau + phi))
                E_tot_iter = np.abs(np.sum(E, axis=1))**2
                final_root = results_arr[np.argmax(E_tot_iter)] if results_arr.size else None
            else:
                E_tot_iter = None

            # Stop refining if no new equilibria are found.
            if results_arr.shape[0] <= prev_count:
                break
            prev_count = results_arr.shape[0]
            results = results_arr
            E_tot = E_tot_iter

        return final_root, results, E_tot

    def residuals(self, x, nd=None, verbose=False):
        """
        Compute vector residuals for steady-state equations of coupled VCSELs,
        including proper phase wrapping and delay-phase correction.

        x = [S1, S2, ..., SN, phi_2, ..., phi_N, omega]
        returns residuals [dS1..dSN, dphi2-dphi1 .. dphiN-dphi1, mean(dphi)-omega]

        Parameters
        ----------
        x : np.ndarray
            State vector [S1..SN, phi_2..phi_N, omega].
        nd : dict or None
            Nondimensional parameters (if None uses self.nd).
        verbose : bool
            If True, emit warnings when time-varying or stacked parameters are
            reduced to a single matrix.
        """
        N_lasers = nd['N_lasers']

        S = x[:N_lasers]
        n = (1 + S/(1 + nd['s'] * S))**(-1) * (nd['p'] - S/(1 + nd['s'] * S))  # invert photon equation to get n

        # Steady-state phase offsets φ̃_i
        phi_tilde = np.zeros(N_lasers)
        phi_tilde[1:] = x[N_lasers : 2*N_lasers - 1]


        omega = x[-1]
        

        nd = nd if nd is not None else self.nd

        # Unpack parameters
        T = nd['T']
        s = nd['s']
        nbar = nd['nbar']
        p = nd['p']
        delta = nd.get('delta_p', 0.0)
        beta_n = nd.get('beta_n', 0.0)
        beta_const = nd.get('beta_const', 0.0)
        alpha = nd.get('alpha', 0.0)
        tau = nd.get('tau', 0.0)
        coupling = nd.get('coupling', 0.0)
        self_fb = nd.get('self_feedback', 0.0)
        kappa_c = nd.get('kappa', 0.0)
        if isinstance(kappa_c, (list, tuple, np.ndarray)):
            kappa_arr = np.array(kappa_c)
            if kappa_arr.ndim == 3:
                if verbose:
                    warnings.warn(
                        "kappa provided as a time series; using the last entry.",
                        RuntimeWarning,
                    )
                kappa_c = kappa_arr[-1]
            else:
                kappa_c = kappa_arr

        

        phi_p = nd.get('phi_p', 0.0)
        if isinstance(phi_p, (list, tuple)) and len(phi_p) > 0:
            if np.ndim(phi_p[0]) == 2:
                if verbose:
                    warnings.warn(
                        "phi_p provided as a list of matrices; using the first entry.",
                        RuntimeWarning,
                    )
                phi_p = np.array(phi_p[0])
        elif isinstance(phi_p, np.ndarray) and phi_p.ndim == 3:
            if verbose:
                warnings.warn(
                    "phi_p provided as a stack of matrices; using the first entry.",
                    RuntimeWarning,
                )
            phi_p = phi_p[0]

        
        # Safety clip
        S_c = np.clip(S, 1e-11, None)
        sqrt_S = np.sqrt(S_c)

        # Delay-induced phase rotation
        w_tau = omega * tau



        inv_T = 1.0 / T
        dn   = np.zeros(N_lasers)
        dS   = np.zeros(N_lasers)
        dphi = np.zeros(N_lasers)


        for i in range(N_lasers):

            # -----------------------------
            # Carrier equation (same form)
            # -----------------------------
            dn[i] = inv_T * (
                p - n[i] - (1.0 + n[i]) * S_c[i] / (1.0 + s * S_c[i])
            )


            # -----------------------------
            # Photon equation: self-feedback
            # -----------------------------
            
            phi_fb = -2.0 * (w_tau + phi_p[i,i])
        
        
            dS[i] = ((1.0 + n[i]) / (1.0 + s * S_c[i]) - 1.0) * S_c[i] \
                    + beta_n * n[i] + beta_const \
                    + 2.0 * np.diag(kappa_c)[i] * self_fb * S_c[i] * np.cos(phi_fb)

            # -----------------------------
            # Phase equation: self-feedback
            # -----------------------------
            dphi[i] = (alpha * 0.5) * (n[i] - nbar) / (1.0 + s * S_c[i]) \
                    + np.diag(kappa_c)[i] * self_fb * np.sin(phi_fb) + delta[i]

            # -----------------------------
            # Mutual coupling terms
            # -----------------------------
            for j in range(N_lasers):
                if j == i:
                    continue

                phi_ij = phi_tilde[j] - phi_tilde[i] - w_tau - phi_p[i, j]
                
                cos_ij = np.cos(phi_ij)
                sin_ij = np.sin(phi_ij)

                # Photon equation
                dS[i] += 2.0 * kappa_c[i,j] * coupling * sqrt_S[i] * sqrt_S[j] * cos_ij

                # Phase equation
                dphi[i] += kappa_c[i,j] * coupling * sqrt_S[j] / sqrt_S[i] * sin_ij
        
        out = np.zeros(N_lasers + (N_lasers - 1) + 1)
        out[:N_lasers] = dS
        out[N_lasers:2*N_lasers-1] = dphi[1:] - dphi[0]

        out[-1] = np.mean(dphi) - omega

        return out

    def order_parameter(self, y_segment):
        """
        Compute the Kuramoto-like order parameter for N complex fields.

        Generalized N-laser order parameter:
            r(t) = | sum_i E_i(t) |^2 / [ N * sum_i |E_i(t)|^2 ]

        Parameters
        ----------
        y_segment : np.ndarray, shape (n_cases, 3*N_lasers, time_steps)
            State history segment. For each laser i:
                carrier index = 3*i
                photon index  = 3*i + 1
                phase index   = 3*i + 2

        Returns
        -------
        r_mean : np.ndarray, shape (n_cases,)
            Time-averaged order parameter for each case.
        """

        eps = 1e-12
        n_cases, total_states, T = y_segment.shape
        N = total_states // 3

        # Extract S_i and phi_i for all lasers
        S_all = []
        phi_all = []

        for i in range(N):
            S_i = np.clip(y_segment[:, 3*i + 1, :], eps, None)
            phi_i = y_segment[:, 3*i + 2, :]
            S_all.append(S_i)
            phi_all.append(phi_i)

        # Stack into arrays: shape (n_cases, N, time_steps)
        S = np.stack(S_all, axis=1)
        phi = np.stack(phi_all, axis=1)

        # Build complex fields E_i
        E = np.sqrt(S) * (np.cos(phi) + 1j * np.sin(phi))  # shape (n_cases, N, T)

        # Compute numerator and denominator
        E_sum = np.sum(E, axis=1)                          # shape (n_cases, T)
        numerator = np.abs(E_sum)**2                       # shape (n_cases, T)

        denom = N * np.sum(np.abs(E)**2, axis=1)           # shape (n_cases, T)
        denom = np.maximum(denom, eps)

        r_t = numerator / denom                            # (n_cases, T)

        # Time-average
        return np.mean(r_t, axis=1)

    def invert_scaling(self, y, phys):
        """
        Convert nondimensional state y back to physical units (in-place).

        The inverse transforms mirror the transforms performed in scale_params():
          n_physical = y_nondim/(g0*tau_p) + N0 + 1/(g0*tau_p)
          S_physical = y_s / (g0 * tau_n)
          phases remain unchanged.

        Parameters
        ----------
        y : np.ndarray, shape (n_cases, 3*N_lasers, steps)
            Nondimensional state time series for N_lasers.
        phys : dict
            Original physical parameter dictionary; required keys: g0, tau_p, tau_n, N0.

        Returns
        -------
        y_phys : np.ndarray
            The same array y converted in-place to physical units and returned.
        """
        g0 = phys['g0']
        tau_p = phys['tau_p']
        tau_n = phys['tau_n']
        N0 = phys['N0']

        # In-place conversion of carriers and photon numbers back to SI units.
        n_idx = slice(0, None, 3)
        s_idx = slice(1, None, 3)
        y[:, n_idx, :] = y[:, n_idx, :] / (g0 * tau_p) + N0 + 1.0 / (g0 * tau_p)
        y[:, s_idx, :] = y[:, s_idx, :] / (g0 * tau_n)
        # phase components remain unchanged
        return y
    
    @staticmethod
    def build_coupling_matrix(time_arr, kappa_initial, kappa_final, N_lasers, ramp_start, ramp_shape, tau, scheme='ATA', aMAT=None, dx=None, plot=False):
        """
        Build a time-dependent coupling matrix for N_lasers with a cosine ramp.

        Parameters
        ----------
        time_arr : np.ndarray, shape (steps,)
            Time axis (seconds).
        kappa_initial : float
            Initial coupling strength.
        kappa_final : float
            Final coupling strength.
        N_lasers : int
            Number of lasers.
        ramp_start : float
            Ramp start time in units of tau.
        ramp_shape : float
            10-90 rise time in units of tau.
        tau : float
            Delay time (seconds).
        scheme : str
            Coupling scheme ('ATA', 'NN', 'CUSTOM', 'RANDOM', 'DECAYED').
        aMAT : np.ndarray or None
            Adjacency matrix for CUSTOM/RANDOM schemes.
        dx : float or None
            Spatial decay parameter for DECAYED scheme.
        plot : bool
            If True, display a plot of the final coupling matrix.

        Returns
        -------
        kappa_mat : np.ndarray, shape (steps, N_lasers, N_lasers)
            Coupling matrix time series.
        """

        kappa_initial = np.ones(shape=(N_lasers,N_lasers))* kappa_initial
        kappa_final = np.ones(shape=(N_lasers,N_lasers)) * kappa_final
        ramp = VCSEL.cosine_ramp(time_arr, ramp_start*tau, ramp_shape*tau, kappa_initial=0, kappa_final=1)
        kappa_arr = kappa_initial + (ramp[:, None, None] * (kappa_final[None, :, :]-kappa_initial[None, :, :]))

        if scheme == 'ATA':
            # All-to-all coupling: both lasers have the same coupling strength
            kappa_mat = np.copy(kappa_arr)
        elif scheme == 'NN':
            kappa_mat = np.zeros_like(kappa_arr)

            # --- Keep diagonals ---
            diag_idx = np.arange(N_lasers)
            kappa_mat[:, diag_idx, diag_idx] = kappa_arr[:, diag_idx, diag_idx]

            # --- Nearest neighbors: i → i+1 (upper diagonal) ---
            up_i = np.arange(N_lasers - 1)
            up_j = np.arange(1, N_lasers)
            kappa_mat[:, up_i, up_j] = kappa_arr[:, up_i, up_j]

            # --- Nearest neighbors: i → i−1 (lower diagonal) ---
            down_i = np.arange(1, N_lasers)
            down_j = np.arange(N_lasers - 1)
            kappa_mat[:, down_i, down_j] = kappa_arr[:, down_i, down_j]
        elif scheme =='CUSTOM':
            if aMAT is None:
                raise ValueError("Custom scheme requires aMAT (adjacency matrix).")
            kappa_mat = np.copy(kappa_arr)
            # Custom scheme can be implemented here
            kappa_mat = kappa_arr * aMAT[None, :, :]
        elif scheme == 'RANDOM':
            # Global coupling: each laser couples to the mean field of all others
            aMAT = np.random.randint(0, 2, size=(N_lasers, N_lasers))
            kappa_mat = kappa_arr * aMAT[None, :, :]
        elif scheme =='DECAYED':
            if dx is None:
                raise ValueError("Decayed scheme requires dx (distance array).")
            kappa_mat = np.copy(kappa_arr)
            aMAT = np.zeros((N_lasers, N_lasers))
            # Exponential decay based on distance
            for i in range(N_lasers):
                for j in range(N_lasers):
                    aMAT[i, j] = dx**np.abs(i - j)

            kappa_mat = kappa_arr * aMAT[None, :, :]            

            
        else:
            raise ValueError("Unsupported coupling scheme: {}".format(scheme))
            

        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5, 4), dpi=300)
            plt.imshow(kappa_mat[-1,:,:]/np.max(kappa_mat[-1,:,:]), cmap='viridis', aspect='auto')
            cbar = plt.colorbar(label='Normalized Coupling Strength')
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('Normalized Coupling Strength', fontsize=14)
            plt.xlabel('Laser Index', fontsize=14)
            plt.ylabel('Laser Index', fontsize=14)
            plt.title('Normalized Coupling Matrix', fontsize=16)
            plt.clim(0,1)
            # Set tick labels at 0 and N_lasers-1 with custom labels
            plt.xticks([0, N_lasers - 1], ['1', str(N_lasers)], fontsize=14)
            plt.yticks([0, N_lasers - 1], ['1', str(N_lasers)], fontsize=14)
            plt.tight_layout()
            plt.show()

        
        return kappa_mat

    def compute_jacobians(self, x, nd, verbose=False):
        """
        Analytic Jacobians of the nondimensional VCSEL DDE
        evaluated at a ROTATING-FRAME EQUILIBRIUM.

        Parameters
        ----------
        x : (3N,) array
            Steady-state vector ordered as
            [n1, S1, ..., nN, SN, phi_2..phi_N, omega].
        nd : dict
            Parameters
        verbose : bool
            If True, emit warnings when time-varying matrices are reduced to a single matrix.
        verbose : bool
            If True, emit warnings when time-varying parameters are reduced to
            a single matrix.

        Returns
        -------
        A0, A1, A2 : (3N, 3N)
            Instantaneous, tau-delay, 2tau-delay Jacobians
        """

        omega = x[-1]
        eps = 1e-12
        N = len(x)//3
        dim = 3*N

        n = x[0::2][:N]
        S = np.maximum(x[1::2][:N], eps)
        phi = np.zeros(N)
        phi[1:] = x[2*N : 3*N - 1]   
        

        sqrtS = np.sqrt(S)

        sqrtS_tau  = sqrtS  # assume constant S at equilibrium
        sqrtS_2tau = sqrtS

        # =========================================================
        # parameters
        # =========================================================
        T      = nd["T"]
        s      = nd["s"]
        nbar   = nd["nbar"]
        beta_n = nd["beta_n"]
        alpha  = nd["alpha"]

        phi_p = np.array(nd["phi_p"])
        if phi_p.ndim == 3:
            if verbose:
                warnings.warn(
                    "phi_p provided as a time series; using the last entry.",
                    RuntimeWarning,
                )
            phi_p = phi_p[-1]
        kappa_full = np.array(nd["kappa"])
        if kappa_full.ndim == 3:
            if verbose:
                warnings.warn(
                    "kappa provided as a time series; using the last entry.",
                    RuntimeWarning,
                )
            kappa_full = kappa_full[-1]
        kappa_diag = np.diag(kappa_full)
        kappa_mat  = kappa_full - np.diag(kappa_diag)

        coupling = nd["coupling"]
        self_fb  = nd["self_feedback"]

        phi_p_self   = np.diag(phi_p)
        phi_p_mutual = phi_p - np.diag(phi_p_self)

        denom = 1 + s*S

        # =========================================================
        # allocate
        # =========================================================
        A0 = np.zeros((dim,dim))
        A1 = np.zeros((dim,dim))
        A2 = np.zeros((dim,dim))

        def idx(i,v):
            #i = laser index
            #v = variable index (0=n,1=S,2=phi)
            return 3*i + v

        # =========================================================
        # A0 : intrinsic physics (ALL VERIFIED)
        # =========================================================
        for i in range(N):
            A0[idx(i,0),idx(i,0)] = -1/T - S[i]/(T*denom[i]) # dn_i/dn_i
            A0[idx(i,0),idx(i,1)] = -1/T * ((denom[i]*(1+n[i]) - (1+n[i])*S[i]*s)/(denom[i]**2)) # dn_i/dS_i
            A0[idx(i,1),idx(i,0)] = S[i]/denom[i] + beta_n # dS_i/dn_i
            A0[idx(i,1),idx(i,1)] = (denom[i]*(1+n[i]) - (1+n[i])*S[i]*s)/(denom[i]**2) - 1 # dS_i/dS_i
            A0[idx(i,2),idx(i,0)] = alpha*0.5/denom[i] # dphi_i/dn_i
            A0[idx(i,2),idx(i,1)] = -alpha*0.5*(n[i]-nbar)*s/(denom[i]**2) # dphi_i/dS_i

        # =========================================================
        # delayed + instantaneous coupling (ALL VERIFIED)
        # =========================================================
        if coupling > 0.0:
            for i in range(N):
                for j in range(N):
                    kap = kappa_mat[i,j]
                    if i == j:
                        # skip diagonal (self-feedback handled later)
                        continue

                    # rotating-frame phase difference
                    dphi = phi[j] - phi[i] - phi_p_mutual[i,j] - omega * nd["tau"]
                    c = np.cos(dphi)
                    s_ = np.sin(dphi)

                    # A1 (tau-delay)
                    A1[idx(i,1),idx(j,1)] += coupling*kap*c*sqrtS[i]/sqrtS_tau[j] # dS_i/dS_j
                    A1[idx(i,1),idx(j,2)] += -2*coupling*kap*sqrtS[i]*sqrtS_tau[j]*s_ # dS_i/dphi_j
                    A1[idx(i,2),idx(j,1)] += coupling*kap*s_/(2*sqrtS[i]*sqrtS_tau[j]) # dphi_i/dS_j
                    A1[idx(i,2),idx(j,2)] += coupling*kap*(sqrtS_tau[j]/sqrtS[i])*c # dphi_i/dphi_j

                    # A0 instantaneous
                    A0[idx(i,1),idx(i,1)] += coupling*kap*c*sqrtS_tau[j]/sqrtS[i] # dS_i/dS_i
                    A0[idx(i,1),idx(i,2)] += 2*coupling*kap*sqrtS[i]*sqrtS_tau[j]*s_ # dS_i/dphi_i
                    A0[idx(i,2),idx(i,2)] += -coupling*kap*(sqrtS_tau[j]/sqrtS[i])*c # dphi_i/dphi_i
                    A0[idx(i,2),idx(i,1)] += -0.5*coupling*kap*s_ * sqrtS_tau[j]/(sqrtS[i]**3) # dphi_i/dS_i

        # =========================================================
        # A2 : self-feedback (INCOMPLETE)
        # =========================================================
        if self_fb > 0.0:
            for i in range(N):
                kap = kappa_diag[i]
                dphi =  - 2*phi_p_self[i] - 2*omega * nd["tau"]
                c = np.cos(dphi)
                s_ = np.sin(dphi)

                A2[idx(i,1),idx(i,1)] += self_fb*kap*c*sqrtS[i]/sqrtS_2tau[i]
                A2[idx(i,1),idx(i,2)] += -2*self_fb*kap*sqrtS[i]*sqrtS_2tau[i]*s_
                A2[idx(i,2),idx(i,1)] += self_fb*kap*s_/(2*sqrtS[i]*sqrtS_2tau[i])
                A2[idx(i,2),idx(i,2)] += self_fb*kap*(sqrtS_2tau[i]/sqrtS[i])

                # instantaneous contribution to A0
                A0[idx(i,1),idx(i,2)] += 2*self_fb*kap*sqrtS[i]*sqrtS_2tau[i]*s_
                A0[idx(i,2),idx(i,2)] += -self_fb*kap*(sqrtS_2tau[i]/sqrtS[i])

        return A0, A1, A2




    def compute_spectrum(self, A_list, tau_list, N,
                     newton_maxit=20,
                     newton_tol=1e-10,
                     stable_margin=-2.0, sparse=False, spectral_shift=0.0, n_eigenvalues=10,
                     verbose=False):
        """
        Cached dense Chebyshev collocation spectrum solver.
        Reuses Pi, Chebyshev matrix, and static Sigma structure.

        Parameters
        ----------
        A_list : list of np.ndarray
            Jacobian matrices for each delay term.
        tau_list : list of float
            Delay values (same ordering as A_list).
        N : int
            Chebyshev polynomial order.
        newton_maxit : int
            Maximum Newton iterations for eigenvalue refinement.
        newton_tol : float
            Convergence tolerance for Newton refinement.
        stable_margin : float or None
            If set, eigenvalues below this real part are accepted without refinement.
        sparse : bool
            If True, use sparse shift-invert for the generalized eigenproblem.
        spectral_shift : float
            Shift for shift-invert mode.
        n_eigenvalues : int
            Number of eigenvalues to compute in sparse mode.
        verbose : bool
            If True, emit warnings when time-varying matrices are reduced to a single matrix.
        """

        from scipy.linalg import eig
        from scipy.special import chebyt

        # ---------------------------------------------------------
        # Setup
        # ---------------------------------------------------------
        normalized = []
        for A in A_list:
            A_arr = np.array(A)
            if A_arr.ndim == 3:
                if verbose:
                    warnings.warn(
                        "A_list contains time-varying matrices; using the last entry.",
                        RuntimeWarning,
                    )
                A_arr = A_arr[-1]
            normalized.append(A_arr.astype(complex))
        A_list = normalized
        tau = np.array(tau_list, dtype=float)
        tau_max = np.max(tau)

        n = A_list[0].shape[0]
        m = len(A_list)
        size = n * (N + 1)

        I_n = np.eye(n, dtype=complex)

        # ---------------------------------------------------------
        # CACHE CHECK
        # ---------------------------------------------------------
        rebuild_cache = (
            not hasattr(self, "_spec_cache") or
            self._spec_cache["N"] != N or
            self._spec_cache["tau_max"] != tau_max or
            self._spec_cache["n"] != n or
            self._spec_cache["m"] != m
        )

        if rebuild_cache:

            # ----- Build Pi once -----
            b = np.zeros((N+1, N+1), dtype=float)
            b[0, :] = 4.0 / tau_max
            b[1, 0:3] = [2, 0, -1]

            for i in range(2, N):
                b[i, i-1:i+2] = [1/i, 0, -1/i]

            b[N, N-1:N+1] = [1/N, 0]
            b *= tau_max / 4.0

            Pi = np.kron(b, I_n)

            # ----- Precompute Chebyshev matrix -----
            x_vals = -2*(tau[1:] / tau_max) + 1
            T = np.array([[chebyt(k)(x) for k in range(N+1)] for x in x_vals])

            # ----- Prebuild static Sigma structure -----
            Sigma_template = np.eye(size, dtype=complex)
            Sigma_template[0:n, 0:n] = 0.0

            # Cache everything
            self._spec_cache = {
                "N": N,
                "tau_max": tau_max,
                "n": n,
                "m": m,
                "Pi": Pi,
                "T": T,
                "Sigma_template": Sigma_template
            }

        cache = self._spec_cache
        Pi = cache["Pi"]
        T = cache["T"]

        # Copy template (cheap compared to rebuilding)
        Sigma = cache["Sigma_template"].copy()

        # ---------------------------------------------------------
        # Only rebuild FIRST BLOCK ROW
        # ---------------------------------------------------------
        for k in range(N+1):

            block = A_list[0].copy()

            for j in range(1, m):
                block += A_list[j] * T[j-1, k]

            col_slice = slice(k*n, (k+1)*n)
            Sigma[0:n, col_slice] = block

        # ---------------------------------------------------------
        # Dense QZ (robust)
        # ---------------------------------------------------------
        if not sparse:
            eigenvalues = eig(Sigma, Pi, right=False, overwrite_a=True, overwrite_b=True)
        else:
            sigma = spectral_shift  # shift for shift-invert (tune if needed)
            # Convert to sparse if needed
            if not sp.issparse(Sigma):
                Sigma = sp.csc_matrix(Sigma)
            else:
                Sigma = Sigma.tocsc()

            if not sp.issparse(Pi):
                Pi = sp.csc_matrix(Pi)
            else:
                Pi = Pi.tocsc()

            n = Sigma.shape[0]

            M = Sigma - sigma * Pi

            # Sparse LU
            lu = spla.splu(M)

            def shift_invert(v):
                return lu.solve(Pi @ v)

            OP = spla.LinearOperator((n, n), matvec=shift_invert)

            mu = spla.eigs(OP, k=n_eigenvalues, which='LM', return_eigenvectors=False)

            lambdas = sigma + 1.0 / mu

            # Filter infinite/huge eigenvalues
            mask = np.isfinite(lambdas) & (np.abs(lambdas) < 1e8)

            eigenvalues = lambdas[mask]



        # ---------------------------------------------------------
        # Newton correction
        # ---------------------------------------------------------
        corrected = []

        A0 = A_list[0]
        if m > 1:
            A1 = A_list[1]
            tau1 = tau[1]

        for lam in eigenvalues:

            if stable_margin is not None and np.real(lam) < stable_margin:
                corrected.append(lam)
                continue

            lam_k = lam

            for _ in range(newton_maxit):

                if m > 1 and np.real(-lam_k * tau1) > 700:
                    break

                M = lam_k * I_n - A0

                if m > 1:
                    exp_term = np.exp(-lam_k * tau1)
                    M -= A1 * exp_term
                    dM = I_n + tau1 * A1 * exp_term
                else:
                    dM = I_n

                try:
                    X = np.linalg.solve(M, dM)
                    trace_term = np.trace(X)

                    if not np.isfinite(trace_term) or abs(trace_term) < 1e-14:
                        break

                    lam_new = lam_k - 1.0 / trace_term

                    if not np.isfinite(lam_new) or abs(lam_new - lam_k) < newton_tol:
                        lam_k = lam_new
                        break

                    lam_k = lam_new

                except np.linalg.LinAlgError:
                    break

            corrected.append(lam_k)

        return np.array(corrected)


    

    def compute_stability(self, x_eq, nd, N=20, newton_maxit=20, threshold=0.0, sparse=False, spectral_shift=0.0, n_eigenvalues=10, verbose=False):
        """
        Compute stability eigenvalues of a steady-state equilibrium.

        Parameters
        ----------
        x_eq : (3N,) array
            Steady-state vector ordered as
            [n1, S1, ..., nN, SN, phi_2..phi_N, omega].
        nd : dict
            Parameters

        Returns
        -------
        stable : float
            1.0 if stable, 0.0 if unstable.
        eigvals : np.ndarray
            Eigenvalues of the linearized DDE system.
        """

        stable = None

        nd_local = nd
        kappa_val = nd.get("kappa")
        phi_p_val = nd.get("phi_p")
        if isinstance(kappa_val, (list, tuple, np.ndarray)):
            kappa_arr = np.array(kappa_val)
            if kappa_arr.ndim == 3:
                nd_local = dict(nd_local)
                nd_local["kappa"] = kappa_arr[-1]
        if isinstance(phi_p_val, (list, tuple, np.ndarray)):
            phi_p_arr = np.array(phi_p_val)
            if phi_p_arr.ndim == 3:
                if nd_local is nd:
                    nd_local = dict(nd_local)
                nd_local["phi_p"] = phi_p_arr[-1]

        A0, A1, A2 = self.compute_jacobians(x_eq, nd_local, verbose=verbose)
        
        if nd_local['self_feedback'] > 0.0:
            tau_list = [0.0, nd_local['tau'], 2*nd_local['tau']]
            A_list = [A0, A1, A2]
        else:
            tau_list = [0.0, nd_local['tau']]
            A_list = [A0, A1]

        eigvals = self.compute_spectrum(
            A_list,
            tau_list,
            N=N,
            newton_maxit=newton_maxit,
            newton_tol=1e-10,
            sparse=sparse,
            spectral_shift=spectral_shift,
            n_eigenvalues=n_eigenvalues,
            verbose=verbose,
        )

        

        # Filter out eigenvalues near zero (numerical artifacts)
        eigvals = eigvals[np.abs(eigvals) > 1e-10]

        if eigvals.size == 0:
            return 1.0, eigvals


        if np.max(eigvals.real) > 0.0:
            stable = 0.0
        else:
            stable = 1.0
        
        return stable, eigvals
