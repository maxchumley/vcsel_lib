"""
vcsel_lib.py

VCSEL simulation utilities.

Provides a VCSEL class that encapsulates physical parameters, nondimensional
scaling, the nondimensional rate equations (with delay and optional noise),
integration routines (Trapezoidal predictor corrector with Euler-Maruyama noise), utilities to generate histories and compute
equilibria (beta), and small helpers (cosine ramp, order parameter, invert scaling).

This module is intended for research/teaching use and keeps a clear separation
between physical parameters (SI units) and nondimensional parameters used by
the integrator.

Author: Max Chumley with documentation assistance from GitHub Copilot
"""

import numpy as np
from sympy import symbols, Eq, solve
from tqdm import tqdm
from scipy.optimize import root
import time

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
      history = vcsel.generate_history(nd, shape='FR')
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
             n' = g0 * tau_p * (N - N0)  (so nbar ~ O(1) at steady-state)
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
        I = phys.get('I', None)
        q = phys['q']
        coupling = phys.get('coupling', 1.0)
        self_feedback = phys.get('self_feedback', 0.0)
        noise_amplitude = phys.get('noise_amplitude', 0.0)
        dt = phys.get('dt', 1e-12)
        Tmax = phys.get('Tmax', 3e-6)
        tau = phys.get('tau', 1e-9)  # feedback delay in seconds

        # Derived discrete parameters
        steps = int(Tmax / dt)
        delay_steps = int(tau / dt)

        injection = phys.get('injection', False)
        injected_strength = phys.get('injected_strength', 0.0)
        injected_phase_diff = phys.get('injected_phase_diff', 0.0)
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
        p = g0 * tau_p * (I * tau_n / (q) - N0) - 1
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
        nd['injected_strength'] = injected_strength
        nd['injected_frequency'] = phys.get('injected_frequency', 0.0)* 1e9*(2*np.pi*tau_p)
        nd['kappa_inj'] = kappa_inj / gamma

        print(n_bar)

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

        n_t   = X_tau[:, :, 0]
        S_t   = X_tau[:, :, 1]
        phi_t = X_tau[:, :, 2]

        n_2t   = X_2tau[:, :, 0]
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
                raise ValueError("phi_p must be scalar or shape (N,N) or (n_cases,N,N)")
            
        phi_p_self = phi_p[:, np.arange(N), np.arange(N)]
        phi_p_mutual = phi_p - np.diag(phi_p_self[0,:])[None,:,:]

        # Time-varying coupling
        kappa_mat = np.array(nd["kappa"][j])        # shape (N,N)
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
            dphi +=nd['coupling'] *  np.sum(kappa_mat[None, :, :] * (sqrtS_t[:, None, :] / sqrtS[:, :, None]) * sin_mutual, axis=2)

        # ----------------- SELF-FEEDBACK -----------------

        if nd["self_feedback"] > 0.0:
            self_amp = sqrtS[:, :] * sqrtS_2t[:, :]
            phi_diff_self = phi_2t - phi - 2*phi_p_self
            cos_self = np.cos(phi_diff_self)
            sin_self = np.sin(phi_diff_self)
            dS += nd["self_feedback"] * 2 * kappa_diag[None, :] * self_amp * cos_self
            dphi += nd["self_feedback"] * kappa_diag[None, :] * ((sqrtS_2t / sqrtS) * sin_self)

        # ----------------- INJECTION (optional) -----------------
        if nd.get("injection", False):
            inj_strength = nd["injected_strength"]
            inj_phase    = nd["injected_phase_diff"]
            omega_inj    = nd["injected_frequency"][j]
            kappa_inj    = nd["kappa_inj"][j]
            t = j*nd["dt"]

            for p in range(N):

                S0   = S[0, p]
                phi0 = phi[0, p]

                inj_cos = np.cos(omega_inj*t + inj_phase - phi0)
                inj_sin = np.sin(omega_inj*t + inj_phase - phi0)

                dS[0, p]  += 2*kappa_inj*np.sqrt(S0*inj_strength)*inj_cos
                dphi[0, p] +=     kappa_inj*np.sqrt(inj_strength/S0)*inj_sin

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
    
    def integrate(self, history, nd=None, progress=False):
        """
        Integrate the nondimensional VCSEL equations.

        Integration scheme:
        - Single-step trapezoidal (Heun) predictor-corrector for the deterministic dynamics.
            * Predictor:   y* = y_n + dt * f(y_n)
            * Corrector:   y_{n+1} = y_n + (dt/2) * (f(y_n) + f(y*))
        - Euler-Maruyama (EM) treatment of additive noise:
            noise is evaluated at the start of the step (y_n) and added after the trapezoid corrector.
        - Delay terms are obtained directly from the stored history; no multistep bootstrap is required.

        Parameters
        ----------
        history : np.ndarray, shape (n_cases, 6, 2*delay_steps)
            Initial history for t in [-2*tau, 0] in nondimensional units.
            The integrator requires at least 2*delay_steps columns to provide the
            necessary delayed states during the first steps.
        nd : dict or None
            Nondimensional parameter dictionary (if None, uses self.nd).
        progress : bool
            If True, show a progress bar using tqdm.

        Returns
        -------
        t_dim : np.ndarray
            Physical time array (seconds) of shape (steps,).
        y : np.ndarray, shape (n_cases, 6, steps)
            Time series of the nondimensional state for all cases.
        freqs : np.ndarray, shape (n_cases, 2, steps)
            Instantaneous phase time derivatives dphi/dt for both lasers
            (columns correspond to laser 1 and laser 2).
        """

        if nd is None:
            nd = self.nd

        n_cases = history.shape[0]
        N_lasers = nd['kappa'].shape[-1]
        dt = nd['dt']
        steps = nd['steps']
        delay_steps = nd['delay_steps']
        noise_amplitude = nd['noise_amplitude']

        # Prepare output arrays
        y = np.zeros((n_cases, 3*N_lasers, steps))
        derivs = np.zeros((n_cases, 3*N_lasers, steps))
        freqs = np.zeros((n_cases, N_lasers, steps))
        freqs[:,:,:2*delay_steps] = nd['delta_p'][None,:,None] * np.ones((n_cases, N_lasers, 2*delay_steps))
        # phi_p = np.full(n_cases, nd['phi_p'])
        phi_p = nd['phi_p']



        # Require full delay history
        if history.shape[2] < 2 * delay_steps:
            raise ValueError("history too short for configured delay_steps")

        # Put initial history into output buffer
        y[:, :, :2 * delay_steps] = history[:, :, :2 * delay_steps]

        # Start integration at the last history index
        start_idx = 2 * delay_steps - 1
        if start_idx >= steps:
            t_dim = np.arange(steps) * dt * nd['tau_p']
            return t_dim, y

        # f_hist holds the last 4 derivative evaluations (for diagnostics)
        f_hist = np.zeros((n_cases, 3*N_lasers, 4))

        # Precompute delay index arrays for speed
        idx_tau_arr = np.arange(steps) - delay_steps
        idx_2tau_arr = np.arange(steps) - 2 * delay_steps
        idx_tau_arr[idx_tau_arr < 0] = 0
        idx_2tau_arr[idx_2tau_arr < 0] = 0

        # ---------- MAIN TRAPEZOIDAL LOOP ----------
        with tqdm(total=steps - 1 - start_idx, desc="Integrating",
                disable=not progress) as pbar:

            y_guess = np.empty((n_cases, 3*N_lasers))
            y_c = np.empty((n_cases, 3*N_lasers))
            tmp_noise = np.empty((n_cases, 3*N_lasers))

            n = start_idx
            while n < steps - 1:
                y_n = y[:, :, n]

                idx_tau = idx_tau_arr[n + 1]
                idx_2tau = idx_2tau_arr[n + 1]
                y_tau = y[:, :, idx_tau]
                y_2tau = y[:, :, idx_2tau]

                # --- Trapezoid predictor + corrector (deterministic)
                f_n = self.f_nd(y_n, y_tau, y_2tau, n, phi_p, nd)[:, :3*N_lasers]
                y_guess[:] = y_n + dt * f_n
                f_guess = self.f_nd(y_guess, y_tau, y_2tau, n + 1, phi_p, nd)[:, :3*N_lasers]

                y_c[:] = y_n + 0.5 * dt * (f_n + f_guess)
                derivs[:, :, n] = 0.5 * (f_n + f_guess)

                # --- Euler–Maruyama noise at y_n
                if noise_amplitude:
                    tmp_noise[:] = self.compute_noise_sample(y_n, noise_amplitude, dt, nd)
                else:
                    tmp_noise.fill(0.0)

                y_next = y_c + tmp_noise

                # ---- POSITIVITY FIX (minimal change) ----
                eps = 1e-11
                n_idx = np.arange(N_lasers) * 3 + 0
                S_idx = np.arange(N_lasers) * 3 + 1
                # clamp n and S only
                # y_next[:, n_idx] = np.maximum(y_next[:, n_idx], eps)
                y_next[:, S_idx] = np.maximum(y_next[:, S_idx], eps)



                y[:, :, n + 1] = y_next

                # --- Update f_hist and freqs
                f_hist = np.roll(f_hist, shift=1, axis=2)
                f_hist[:, :, 0] = derivs[:, :, n] + tmp_noise[:]/np.sqrt(dt)  # include noise contribution in derivative

                freq_indices = np.arange(N_lasers) * 3 + 2  # indices of phi variables
                freqs[:, :, n + 1] = f_hist[:, :, 0][:, freq_indices]


                n += 1
                if progress:
                    pbar.update(1)

        # Time array in physical units
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
        
        # The final step uses broadcasting:
        # ramp is broadcast against (kappa_final - kappa_initial) and kappa_initial.
        # The result's shape will be the broadcast shape of t and the kappas.
        return kappa_initial + (kappa_final - kappa_initial) * ramp

    def generate_history(self, nd=None, shape='FR', n_cases=1):
        """
        Produce an initial history array required by the delay integrator.

        The integrator expects inputs for times t in [-2*tau, 0]. This routine
        returns an array shaped (n_cases, 6, 2*delay_steps) in nondimensional
        units.

        Parameters
        ----------
        nd : dict or None
            Nondimensional parameter dictionary (if None uses self.nd).
        shape : str
            One of:
              - 'FR' : free-running steady-state history (steady carriers and
                       steady photon numbers, linear phase ramp from detuning)
              - 'ZF' : zero-field history (all fields zero except phase ramp)
        n_cases : int
            Number of independent cases to tile the history for.

        Returns
        -------
        history : np.ndarray, shape (n_cases, 6, 2*delay_steps)
            History suitable as input to integrate().
        """
        if nd is None:
            nd = self.nd
        ds = nd['delay_steps']
        length = 2 * ds
        eq = None

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
            # Steady-state equilibrium
            N_lasers = nd.get('N_lasers', 2)
            hist = np.zeros((n_cases, 3*N_lasers, length))
            freq_hist = np.zeros((n_cases, N_lasers, 2*nd['delay_steps']))
            for k, phi_p in enumerate(nd['phi_p']):
                eq, _ = self.solve_equilibria(nd=nd, phi_p=phi_p)

                if eq is not None:
                    omega = eq[-1]
                    phase_diff = np.concatenate([[0.0], eq[2*N_lasers:(3*N_lasers -1)]])[:N_lasers]#np.concatenate([[0.0], eq[2*N_lasers:3*N_lasers-1]])

                    hist[k, 0::3, :] = eq[0::2][:N_lasers].reshape(-1,1)*np.ones(shape=(N_lasers,length))         # n1
                    hist[k, 1::3, :] = eq[1::2][:N_lasers].reshape(-1,1)*np.ones(shape=(N_lasers,length))            # S1 (nondimensional)
                    hist[k, 2::3, :] = omega * nd['dt'] * np.arange(length)*np.ones(shape=(N_lasers,length))       # phi1
                    hist[k, 2::3, :] += phase_diff.reshape(-1,1)  
                    freq_hist[k, :, :] = omega * np.ones((N_lasers, length)) / (2*np.pi*1e9*nd['tau_p'])  # convert back to rad/s
                else:
                    hist[k, 0, :] = nd['nbar']         # n1
                    hist[k, 1, :] = nd['sbar']         # S1 (nondimensional)
                    hist[k, 2, :] = 0.0       # phi1
                    hist[k, 3, :] = nd['nbar']         # n2
                    hist[k, 4, :] = nd['sbar']         # S2
                    # phi2: linear ramp consistent with delta (nondimensional)
                    hist[k, 5, :] = nd['delta_p'] * np.arange(length) * nd['dt']



            # history = np.tile(hist[np.newaxis, :, :], (n_cases, 1, 1))
            return hist, freq_hist, eq

        else:
            raise ValueError("unsupported history shape")

        # Tile for multiple cases
        history = np.tile(hist[np.newaxis, :, :], (n_cases, 1, 1))
        freq_hist = np.tile(freq_hist[np.newaxis, :, :], (n_cases, 1, 1))
        return history, freq_hist, eq
    
    def solve_equilibria(self, nd, phi_p=None, guesses=None):
        """
        Solve for steady-state roots of the coupled VCSEL system.

        Parameters
        ----------
        nd : dict
            Nondimensional parameters.
        residuals_func : callable
            Function of the form residuals(x, nd=...).
        guesses : list of np.ndarray
            List of initial guesses, each shape (6,).
        tol : float
            Tolerance for solver convergence.

        Returns
        -------
        results : list of dict
            Each dict contains {'x': root_vector, 'success': bool, 'res_norm': float}.
        """
        results = []

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

        phase_count = 20
        max_omg = 10 * 2 * np.pi * 1e9 * nd['tau_p']#np.max(np.abs(nd['delta_p']))
        phase_vals = np.linspace(-2*np.pi, 2*np.pi, phase_count, endpoint=False)
        omega_seeds = np.linspace(-max_omg, max_omg, 200)#np.linspace(-1*nd['delta_p'], 1*nd['delta_p'], 50)  # try zero and the detuning as initial omega guesses

        
        for phi in phase_vals:
            for omega_guess in omega_seeds:
                guesses.append(np.concatenate([
                    np.tile([nd['nbar'], nd['sbar']], N_lasers),  # n1, S1, n2, S2, ...
                    np.full(N_lasers-1, phi),                      # φ1, φ2, ...
                    np.array([omega_guess])                      # ω
                ]))

        

        for i, guess in enumerate(guesses):
            sol = root(self.residuals, guess, args=(nd,phi_p,), tol=tol, method='hybr',)
            res_norm = np.linalg.norm(sol.fun)
            # from scipy.optimize import least_squares
            if res_norm < tol and sol.success:
                results.append(sol.x)


        if results:
            results = np.array(results)

            results[:, 2*N_lasers : 3*N_lasers - 1] = results[:, 2*N_lasers : 3*N_lasers - 1] % (2.0 * np.pi)  # wrap phases to [0, 2pi)
            

            omega = results[:, -1]

            S = results[:,1::2][:,:N_lasers]                # (n_roots, N)
            valid_indices = np.all(S > 0.0, axis=1)   # only keep roots with all S_i > 0
            S = S[valid_indices]
            omega = omega[valid_indices]
            phi = np.zeros((len(S), N_lasers))
            phi[:, 1:] = results[valid_indices, 2*N_lasers : 3*N_lasers - 1] # phi1 = 0

            
            E = np.sqrt(S) * np.exp(1j * (omega[:, None] * tau + phi + nd['phi_p']))
            E_tot = np.abs(np.sum(E, axis=1))**2

            final_root = results[np.argmax(E_tot)]
        else:
            final_root = None

        return final_root, results

    def residuals(self, x, nd=None, phi_p_override=None):
        """
        Compute vector residuals for steady-state equations of coupled VCSELs,
        including proper phase wrapping and delay-phase correction.

        x = [n1, S1, n2, S2, phase_diff, omega]
        returns residuals [dn1, dS1, dn2, dS2, dphi2 - dphi1, dphi1 - omega]
        """
        import numpy as np

        N_lasers = nd['N_lasers']

        n = x[0:2*N_lasers][::2]
        S = x[1:2*N_lasers][::2]
        # phase_diff = np.zeros((N_lasers, N_lasers))
        # phase_diff[np.triu_indices(N_lasers, k=1)] = x[2*N_lasers:(2*N_lasers + np.sum(np.arange(N_lasers)))]
        # phase_diff = phase_diff + phase_diff.T

        # Steady-state phase offsets φ̃_i
        phi_tilde = np.zeros(N_lasers)
        phi_tilde[1:] = x[2*N_lasers : 3*N_lasers - 1]


        omega = x[-1]
        

        nd = nd if nd is not None else self.nd

        # Unpack parameters
        T = nd['T']; s = nd['s']; nbar = nd['nbar']; p = nd['p']
        delta = nd.get('delta_p', 0.0)
        beta_n = nd.get('beta_n', 0.0)
        beta_const = nd.get('beta_const', 0.0)
        alpha = nd.get('alpha', 0.0)
        tau = nd.get('tau', 0.0)
        coupling = nd.get('coupling', 0.0)
        self_fb = nd.get('self_feedback', 0.0)
        kappa_c = nd['kappa'] if isinstance(nd.get('kappa'), (list, tuple, np.ndarray)) else nd.get('kappa', 0.0)

        

        # Choose phi_p
        if phi_p_override is not None:
            phi_p = phi_p_override
        else:
            phi_p = nd.get('phi_p', 0.0)
            #float(phi_p_val[-1]) if isinstance(phi_p_val, (list, tuple, np.ndarray)) else float(phi_p_val)

        
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
        
        out = np.zeros(2*N_lasers + (N_lasers - 1) + 1)  # dn1, dS1, dn2, dS2, dphi2- dphi1, dphi1 - omega
        out[0:2*N_lasers][::2] = dn
        out[0:2*N_lasers][1::2] = dS
        out[2*N_lasers:3*N_lasers-1] = dphi[1:] - dphi[0]

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
        y : np.ndarray, shape (n_cases, 6, steps)
            Nondimensional state time series to be converted (converted in-place).
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

        # In-place conversion of carriers and photon numbers back to SI units
        y[:, 0, :] = y[:, 0, :] / (g0 * tau_p) + N0 + 1.0 / (g0 * tau_p)  # n1 -> N1
        y[:, 1, :] = y[:, 1, :] / (g0 * tau_n)  # S1 -> physical photon number
        # phi1 unchanged
        y[:, 3, :] = y[:, 3, :] / (g0 * tau_p) + N0 + 1.0 / (g0 * tau_p)  # n2 -> N2
        y[:, 4, :] = y[:, 4, :] / (g0 * tau_n)  # S2 -> physical photon number
        # phi2 unchanged
        return y
    
    @staticmethod
    def build_coupling_matrix(time_arr, kappa_initial, kappa_final, N_lasers, ramp_start, ramp_shape, tau, scheme='ATA', aMAT=None, dx=None, plot=False):
        """
        Build the coupling matrix for two VCSELs from time-dependent coupling array.

        Parameters
        ----------
        kappa_arr : np.ndarray, shape (steps,)
            Time-dependent coupling strength array.

        Returns
        -------
        kappa_mat : np.ndarray, shape (steps, 2)
            Coupling matrix for the two VCSELs.
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
            # plt.savefig(f'decayed_coupling/coupling_dx_{dx:.2f}.png', dpi=300)
            plt.show()

        
        return kappa_mat


# If executed as a script, run a short example demonstrating the usage of VCSEL
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from sympy import symbols, Eq, solve
    from itertools import combinations

    alpha = 2
    tau_p = 5.4e-12
    tau_n = 0.25e-9
    g0 = 8.75e-4 * 1e9
    N0 = 2.86e5
    s = 4e-6 
    q = 1.602e-19
    beta = 1.e-3
    # kappa_c = 12e9
    tau = 1e-9  # delay (s)
    eta = 0.9
    current_threshold = 3
    I = eta*current_threshold * q/ tau_n * (N0 + 1/(g0*tau_p))



    self_feedback = 0.0
    coupling = 1.0
    noise_amplitude = 0.0



        
    detuning = 0.5 # detuning (GHz)
    delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s

    phi_p = 0.0

    dt = 0.5*tau_p # 1 ps
    Tmax = 2e-7
    steps = int(Tmax / dt)
    time_arr = np.linspace(0, Tmax, steps)
    delay_steps = int(tau / dt)
    segment_len = int(steps/2)
    segment_start = int(steps/2)

    # Kappa ramp
    kappa_c = 5e9


    ramp_start = 2
    ramp_shape = 100


    N_lasers = 2


    n_iterations = 1

    kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=kappa_c, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme='ATA')




    phi_p_vals = np.array([np.pi])

    phys = {
        'tau_p': tau_p,
        'tau_n': tau_n,
        'g0': g0,
        'N0': N0,
        'N_bar': N0 + 1/(g0*tau_p),
        's': s,
        'beta': beta,
        'kappa_c_mat': kappa_arr,
        'phi_p_mat': np.ones(shape=(n_iterations,N_lasers,N_lasers))*phi_p_vals[:,None,None],
        'I': I,
        'q': q,
        'alpha': alpha,
        'delta': np.sort(np.concatenate([[0.0], [delta]])),
        'coupling': coupling,
        'self_feedback': self_feedback,
        'noise_amplitude': noise_amplitude,
        'dt': dt,
        'Tmax': Tmax,
        'tau': tau,
        'N_lasers': N_lasers
    }

    N_bar_sym, S_bar = symbols('N_bar S_bar')




    kappa_max = 20e9

    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    n_cases = len(nd['phi_p'])
    nd['N_lasers'] = N_lasers

    nd['kappa'] = nd['kappa'][-1]

    history, freq_hist, _ = vcsel.generate_history(nd, shape='FR', n_cases=n_iterations)

    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()


    t, y, freqs = vcsel.integrate(history, nd=nd, progress=True)
    E_tot = np.abs(np.sum(np.sqrt(y[:, 1::3, :]) * (np.cos(y[:, 2::3, :]) + 1j * np.sin(y[:, 2::3, :])), axis=1))**2
    plt.figure(figsize=(10, 6), dpi=90)
    plt.plot(t * 1e9, E_tot[0,:], label=r'$|E_1 + E_2|^2$', linewidth=2)
    plt.xlabel('Time (ns)')
    plt.ylabel('Scaled $|E_{tot}|^2$')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()
