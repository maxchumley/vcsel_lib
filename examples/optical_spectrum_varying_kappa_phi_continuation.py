#%%
# Example usage
import numpy as np
import multiprocessing as mp
from vcsel_lib import VCSEL
from sympy import symbols, Eq, solve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from IPython.display import clear_output
import gc
from scipy.ndimage import uniform_filter1d
from IPython.display import clear_output



rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')




for p in range(1):
    # --- Parameters ---
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

    # print(f"p={g0*tau_p * (I*tau_n/(q) - N0) - 1:.3f}")


    self_feedback = 0.0
    coupling = 1.0
    noise_amplitude = 1.0

    detuning = 4.0  # detuning (GHz)
    delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s


    dt = 0.5*tau_p # 1 ps
    Tmax = 2e-7 


    steps = int(Tmax / dt)

    time_arr = np.linspace(0, Tmax, steps)
    delay_steps = int(tau / dt)
    segment_len = int(steps/2)
    segment_start = int(steps/2)

    resolution = 100
    N_lasers = 2
    coupling_scheme = 'ATA'  # 'ATA', 'NN' or 'RANDOM'
    ramp_start = 2

    dx=1.0

    kappa_c = np.linspace(0e9,20e9,resolution)



    phi_p_vals = np.array([1*np.pi])#np.linspace(0,2*np.pi,resolution)

    n_iterations = 100

    phys = {
        'tau_p': tau_p,
        'tau_n': tau_n,
        'g0': g0,
        'N0': N0,
        'N_bar': N0 + 1/(g0*tau_p),
        's': s,
        'beta': beta,
        'kappa_c_mat': None,
        'phi_p_mat': np.ones(shape=(n_iterations,N_lasers,N_lasers))*phi_p_vals[:,None,None],
        'I': I,
        'q': q,
        'alpha': alpha,
        'delta': np.sort(np.concatenate([[0], [delta]])),
        'coupling': coupling,     
        'self_feedback': self_feedback, 
        'noise_amplitude': noise_amplitude,
        'dt': dt,
        'Tmax': Tmax,
        'tau': tau,
        'N_lasers': N_lasers
    }



    # t, y, freqs = vcsel.integrate(history, nd=nd, progress=True)



    ramp_start = 10
    ramp_shape = 100


    kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=kappa_c[1], N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, dx=dx)

    phys['kappa_c_mat'] = kappa_arr

    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()



    n_cases = len(nd['phi_p'])


    history, _, _ = vcsel.generate_history(nd, shape='FR', n_cases=n_cases)



    %matplotlib inline
    from scipy.signal import butter, filtfilt, decimate, welch
    import warnings
    import psutil
    import matplotlib 
    from tqdm import tqdm
    # matplotlib.use('Agg')  

    # Calculate phi_p based on free running wavelength lambda_0 and delay tau
    lambda_0 = 910e-9  # meters (example value, update as needed)
    c = 3e8  # speed of light (m/s)
    nu_0 = c / lambda_0  # optical frequency (Hz)
    phi_p = phys['phi_p_mat'][0]#np.pi#(2 * np.pi * nu_0 * tau) % (2 * np.pi)  # phase shift due to delay, wrapped to [0, 2π]

    # print(f"Calculated phi_p: {phi_p:.4f} radians ({phi_p/np.pi:.4f} π)")





    y = np.zeros((6, steps))
    kappa_prev = 0.0e9 
    dtype = np.float32  # cut memory in half
    spectrum_db_norm_list = np.zeros((resolution, segment_len), dtype=dtype)
    spectrum_E1_db_norm_list = np.zeros((resolution, segment_len), dtype=dtype)
    spectrum_E2_db_norm_list = np.zeros((resolution, segment_len), dtype=dtype)
    cos_phase_diff_time = np.zeros((len(kappa_c), segment_len), dtype=dtype)
    order_param = np.zeros(len(kappa_c), dtype=dtype)

    import sys



    pbar = tqdm(kappa_c, desc="Simulating", unit="step")
    for k, kappa in enumerate(pbar):
        # Slice the ramp for this segment
        if k > 0:
            kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=kappa_c[k-1], kappa_final=kappa_c[k], N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, dx=dx)
        phys['kappa_c_mat'] = kappa_arr
        vcsel = VCSEL(phys)
        nd = vcsel.scale_params()
        t, y_scaled, freqs = vcsel.integrate(history, nd=nd, progress=False)

        y = y_scaled.copy()#vcsel.invert_scaling(y_scaled.copy(), phys)
        S = []
        phi = []
        dphi = []

        for i in range(N_lasers):
            # Your original indexing used:
            #   Laser 1 → y[:, 1, :]
            #   Laser 2 → y[:, 4, :]
            #   Laser 1 phase → y[:, 2, :]
            #   Laser 2 phase → y[:, 5, :]
            #
            # Pattern is (per laser): intensity index = 3*i + 1 , phase index = 3*i + 2
            S.append(np.abs(y[:, 3*i + 1, :]))
            phi.append(y[:, 3*i + 2, :])
            dphi.append(freqs[:, i, :] * 1e-9/(2*np.pi*tau_p))

        S = np.stack(S, axis=1)        # shape: (time, N, traj)
        phi = np.stack(phi, axis=1)
        dphi = np.stack(dphi, axis=1)

        # Pairwise phase differences (using laser 0 as reference)
        phase_diff = np.unwrap(phi - phi[:, 0:1, :], axis=1)

        # Order parameter now works for N lasers
        order_param[k] = np.mean(vcsel.order_parameter(y[:,:,-int(len(t)/2):]))

        # Build fields E_i
        E_list = []
        for i in range(N_lasers):
            Ei = np.sqrt(S[:, i, :]) * (np.cos(phi[:, i, :]) + 1j*np.sin(phi[:, i, :]))
            E_list.append(Ei)

        E_all = np.stack(E_list, axis=1)     # shape: (time, N, traj)

        # Total intensity across lasers
        E_tot = np.sum(E_all, axis=1)

        E_1 = E_all[:, 0, :]
        E_2 = E_all[:, 1, :]    

        avg_cos_pd = np.mean(np.mean(np.cos(phase_diff), axis=0)[1:], axis=0)

        


        # ###############################################################################
        # ###############        SPECTRUM COMPUTATION (N LASERS)        #################
        # ###############################################################################

        h = 6.626e-34
        conversion = h * nu_0 / tau_p

        fs = 1.0 / nd['dt']
        desired_df = 1e6 * tau_p
        nperseg = np.shape(E_all)[-1]
        noverlap = nperseg // 2
        N_fft = max(int(np.ceil(fs / desired_df)), nperseg)

        


        # ---------- TOTAL FIELD SPECTRUM ----------
        f, psd_tot = welch(
            E_all.sum(axis=1),
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=N_fft,
            return_onesided=False,
            scaling="density"
        )
        psd_tot_watts = np.abs(psd_tot)**2 * conversion
        spectrum_db = np.mean(10*np.log10(psd_tot_watts/1e-3 + 1e-20), axis=0)

        # ---------- PER-LASER SPECTRA ----------
        # num_lasers = E_all.shape[1]
        spectra_db = np.zeros((N_lasers, len(f)), dtype=np.float32)
        cos_phase_diff_time[k,:] = avg_cos_pd[-segment_len:]#np.cos(phase_diff_time)


        for i in range(N_lasers):
            _, psd_i = welch(
                E_all[:, i, :],
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=N_fft,
                return_onesided=False,
                scaling="density"
            )
            psd_i_watts = np.abs(psd_i)**2 * conversion
            spectra_db[i] = np.mean(10*np.log10(psd_i_watts/1e-3 + 1e-20), axis=0)

        # ---------- Frequency window ----------
        idx_sort = np.argsort(f)
        f_sorted = f[idx_sort]

        f_plot_min = -5.0
        f_plot_max = 5.0
        mask = (f_sorted >= f_plot_min*1e9*tau_p) & (f_sorted <= f_plot_max*1e9*tau_p)

        f_window = f_sorted[mask]
        spectrum_db = spectrum_db[idx_sort][mask]
        spectra_db = spectra_db[:, idx_sort][:, mask]

        if k == 0:
            n_freqs = len(f_window)
            spectrum_db_norm_list = np.zeros((resolution, n_freqs), dtype=dtype)
            spectrum_laser_db_norm_list = np.zeros((resolution, N_lasers, n_freqs), dtype=dtype)
            spectrum_E1_db_norm_list = np.zeros((resolution, n_freqs), dtype=dtype)
            spectrum_E2_db_norm_list = np.zeros((resolution, n_freqs), dtype=dtype)

        spectrum_db_norm_list[k] = spectrum_db - np.max(spectrum_db)
        spectrum_laser_db_norm_list[k] = spectra_db - np.max(spectra_db, axis=1, keepdims=True)

        spectrum_E1_db_norm_list[k] = spectra_db[0] - np.max(spectra_db[0])
        spectrum_E2_db_norm_list[k] = spectra_db[1] - np.max(spectra_db[1])


        # ###############################################################################
        # ###############################   PLOTTING   #################################
        # ###############################################################################

            # ###############################################################################
        # ###############################   PLOTTING   #################################
        # ###############################################################################

        font_size = 22
        f_plot = f_window/tau_p

        if k%1 == 0 or k == len(kappa_c)-1:
            # clear_output(wait=True)
            fig = plt.figure(figsize=(18,10), dpi=300)
            gs = fig.add_gridspec(20, 30, height_ratios=[1]*20, width_ratios=[1]*30, hspace=0.3)

            # --- Cosine of Phase Difference (top row, spans all columns) ---
            ax0 = fig.add_subplot(gs[1:8, 0:-3])

            time_window = np.arange(int(0), int(np.ceil(segment_len)/3))
            im0 = ax0.imshow(
                cos_phase_diff_time[:, time_window],
                aspect='auto',
                extent=[time_arr[time_window[0]]*1e9, time_arr[time_window[-1]]*1e9, kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
                origin='lower',
                cmap='jet',
                vmin=-1, vmax=1, rasterized=True
            )
            cbar0 = fig.colorbar(im0, ax=ax0, pad=0.02)
            cbar0.set_label(r'$\cos(\Delta\phi)$', fontsize=font_size, labelpad=0 )
            cbar0.ax.tick_params(labelsize=font_size)
            ax0.set_ylabel(r'$\kappa_c~(\mathrm{ns}^{-1})$', fontsize=font_size)
            ax0.set_xlabel(r'Time (ns)', fontsize=font_size)
            ax0.set_title(r'$\phi_p={:+.2f}\pi$'.format(phi_p[0,0]/np.pi), fontsize=font_size, pad=16)
            ax0.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
            ax0.tick_params(axis='both', labelsize=font_size)

            # --- Order Parameter Plot (to the right of ax0) ---
            ax_order = fig.add_subplot(gs[1:8, -2:])
            ax_order.plot(order_param, kappa_c * 1e-9, color='black', linewidth=2)
            ax_order.set_title('Order Parameter', fontsize=font_size, pad=16)
            ax_order.set_ylim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
            ax_order.tick_params(axis='both', labelsize=font_size)
            ax_order.set_yticks([])  # Remove y axis ticks
            ax_order.set_xticks(np.linspace(0, 1, 2))
            ax_order.set_xlim(0, 1)
            
            # --- Optical Spectrum (Total) ---
            ax1 = fig.add_subplot(gs[11:, 1:8])
            im = ax1.imshow(
                spectrum_db_norm_list,
                aspect='auto',
                extent=[f_plot[0]*1e-9, f_plot[-1]*1e-9, kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
                origin='lower',
                cmap='jet', rasterized=True
            )
            im.set_clim(-100,0) 
            ax1.set_xlabel("Frequency (GHz)", fontsize=font_size, labelpad=10)
            ax1.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
            ax1.set_title(r"$|\mathcal{F}\left(E_{tot}\right)|^2$", fontsize=font_size, pad=16)
            ax1.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
            ax1.set_xticks(np.linspace(f_plot_min, f_plot_max, 3))
            ax1.set_xlim(f_plot_min, f_plot_max)
            ax1.set_ylim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
            ax1.tick_params(axis='both', labelsize=font_size)
            ax1.axvline(detuning, color='black', linestyle='--', linewidth=2, label=rf"$\delta$", alpha=0.5)

            # --- E_1 Spectrum ---
            ax2 = fig.add_subplot(gs[11:, 11:18])
            im2 = ax2.imshow(
                spectrum_E1_db_norm_list,
                aspect='auto',
                extent=[f_plot[0]*1e-9, f_plot[-1]*1e-9, kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
                origin='lower',
                cmap='jet', rasterized=True
            )
            im2.set_clim(-100, 0)
            ax2.set_xlabel("Frequency (GHz)", fontsize=font_size, labelpad=10)
            ax2.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
            # ax2.set_title(r"$\mathcal{F}\left(|E_1|^2\right)$", fontsize=font_size, pad=16)
            ax2.set_title(r"$|\mathcal{F}\left(E_1\right)|^2$", fontsize=font_size, pad=16)
            ax2.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
            ax2.set_xticks(np.linspace(f_plot_min, f_plot_max, 3))
            ax2.set_xlim(f_plot_min, f_plot_max)
            ax2.set_ylim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
            ax2.tick_params(axis='both', labelsize=font_size)
            ax2.axvline(detuning, color='black', linestyle='--', linewidth=2, label=rf"$\delta$", alpha=0.5)

            # --- E_2 Spectrum ---
            ax3 = fig.add_subplot(gs[11:, 21:29])

            im3 = ax3.imshow(
                spectrum_E2_db_norm_list,
                aspect='auto',
                extent=[f_plot[0]*1e-9, f_plot[-1]*1e-9, kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
                origin='lower',
                cmap='jet', rasterized=True
            )
            cbar3 = fig.colorbar(im3, ax=ax3, pad=0.05)
            cbar3.set_label(r'Power (dBm)', fontsize=font_size, labelpad=12)
            cbar3.ax.tick_params(labelsize=font_size)
            im3.set_clim(-100, 0)
            ax3.set_xlabel("Frequency (GHz)", fontsize=font_size, labelpad=10)
            ax3.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
            ax3.set_title(r"$|\mathcal{F}\left(E_2\right)|^2$", fontsize=font_size, pad=16)
            ax3.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
            ax3.set_xticks(np.linspace(f_plot_min, f_plot_max, 3)) 
            ax3.set_xlim(f_plot_min, f_plot_max)
            ax3.set_ylim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
            ax3.tick_params(axis='both', labelsize=font_size)
            ax3.axvline(detuning, color='black', linestyle='--', linewidth=2, label=rf"$\delta$", alpha=0.5)
            ax3.legend(fontsize=font_size, loc='upper right')

            filename = f"./3_laser_random_detuning/2LASER_{p}.png"
            
            # phi_p{phi_p[0,0]/np.pi:.2f}pi_continuation_noise_detuning{detuning}_alpha{alpha}_noise_{n_cases}_{n_iterations}avg_self{self_feedback:.2f}.png"
            # plt.savefig(filename, bbox_inches='tight')
            plt.show()
            plt.close(fig)
            plt.cla(); plt.clf()
            plt.close('all')
            clear_output(wait=True)



#%%

np.save(f"./linewidth_estimation/numpy_arrays/spectrum_db_norm_list_self{self_feedback:.2f}_200_avg_spectra_detuning{detuning}.npy", spectrum_db_norm_list)
np.save(f"./linewidth_estimation/numpy_arrays/spectrum_E1_db_norm_list_self{self_feedback:.2f}_200_avg_spectra_detuning{detuning}.npy", spectrum_E1_db_norm_list)
np.save(f"./linewidth_estimation/numpy_arrays/spectrum_E2_db_norm_list_self{self_feedback:.2f}_200_avg_spectra_detuning{detuning}.npy", spectrum_E2_db_norm_list)
np.save(f"./linewidth_estimation/numpy_arrays/f_sorted_self{self_feedback:.2f}_200_avg_spectra_detuning{detuning}.npy", f_sorted)
np.save(f"./linewidth_estimation/numpy_arrays/cos_phase_diff_time_self{self_feedback:.2f}_200_avg_spectra_detuning{detuning}.npy", cos_phase_diff_time)
# %%

# --- Load saved data at desired detuning frequency---

detuning = 0.0

import numpy as np
spectrum_db_norm_list = np.load(f"./linewidth_estimation/numpy_arrays/spectrum_db_norm_list_no_self_200_avg_spectra_detuning{detuning}.npy")
spectrum_E1_db_norm_list = np.load(f"./linewidth_estimation/numpy_arrays/spectrum_E1_db_norm_list_no_self_200_avg_spectra_detuning{detuning}.npy")
spectrum_E2_db_norm_list= np.load(f"./linewidth_estimation/numpy_arrays/spectrum_E2_db_norm_list_no_self_200_avg_spectra_detuning{detuning}.npy")
cos_phase_diff_time = np.load(f"./linewidth_estimation/numpy_arrays/cos_phase_diff_time_no_self_200_avg_spectra_detuning{detuning}.npy")
f_sorted = np.load(f"./linewidth_estimation/numpy_arrays/f_sorted_no_self_200_avg_spectra_detuning{detuning}.npy")
f_plot_min = -15   # -2 GHz
f_plot_max = 15   # +15 GHz
mask = (f_sorted >= f_plot_min*1e9*tau_p) & (f_sorted <= f_plot_max*1e9*tau_p)

# --- Apply mask & normalize ---
f_window = f_sorted[mask]/tau_p
f_plot = f_window
n_cases = 200
kappa_arr = np.concatenate([np.linspace(0, kappa_max, n_cases)])


#%%
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from IPython.display import clear_output
from scipy.signal import find_peaks
from scipy.ndimage import median_filter


from scipy.signal import savgol_filter

# Example parameters (tune for your data)
window_length = 5   # must be odd
polyorder = 3


num_curves = 200
colors = [cm.Blues(0.3 + 0.7 * (i) / (num_curves - 1)) for i in range(num_curves)]


fields = {'E1': spectrum_E1_db_norm_list, 'E2': spectrum_E2_db_norm_list, 'Etot': spectrum_db_norm_list}


plot_field = 'E2'

spec_smooth_reflected = np.zeros((fields[plot_field].shape[0], 1*fields[plot_field].shape[1]), dtype=fields[plot_field].dtype)

fwhm = []

peak_pts = []
fig = plt.figure(figsize=(8, 5), dpi=300)

for i in range(0,num_curves):
    # Find the index of the maximum for this spectrum
    # i = 0
    if kappa_arr[i]*1e-9 > -1:# and kappa_arr[i]*1e-9 < 11:
        

        clear_output(wait=True)

        # Find FWHM of the peak centered at freq_shift
        # peak_power = spectrum_db_norm_list[i, max_idx]
        half_max =  - 3  # 3 dB down for FWHM


        spec_smooth = fields[plot_field][i, :]#savgol_filter(fields[plot_field][i, :], window_length, polyorder)
        # Reflect spec_smooth about the peak (index max_idx)
        spec_smooth_reflected[i,:] = spec_smooth#np.concatenate([spec_smooth[::-1], spec_smooth])
        f_plot_reflected = f_plot#np.concatenate([-f_plot[::-1], f_plot])



        # max_idx = np.argmax(spec_smooth)
        # spec_smooth = spec_smooth - np.max(spec_smooth)
        # spec_smooth[f_plot<=0] = -1000
        # spec_smooth = spec_smooth - np.max(spec_smooth)
        max_idx = np.argmax(spec_smooth_reflected[i,:])
        freq_shift = np.abs(f_plot_reflected[max_idx])
        shifted_freq = (f_plot_reflected - freq_shift) * 1e-6

        
        

        # plt.plot(-f_plot[::-1]*1e-6,  spectrum_db_norm_list[i, ::-1], color=colors[i], label=f'$\kappa_c$={kappa_arr[i]*1e-9:.2f} ns$^{{-1}}$', alpha=0.5)
        # plt.plot(f_plot*1e-6,  spectrum_db_norm_list[i, :], color=colors[i], label=f'$\kappa_c$={kappa_arr[i]*1e-9:.2f} ns$^{{-1}}$', alpha=0.5)



        # plt.plot(f_plot_reflected*1e-6 - np.abs(f_plot_reflected[max_idx]*1e-6),  spec_smooth_reflected[i,:], color=colors[i], label=f'$\kappa_c$={kappa_arr[i]*1e-9:.2f} ns$^{{-1}}$')

        plt.plot(f_plot_reflected*1e-6 ,  spec_smooth_reflected[i,:], color=colors[i], label=f'$\kappa_c$={kappa_arr[i]*1e-9:.2f} ns$^{{-1}}$')

        peak_pts.append((np.abs(f_plot_reflected[max_idx]*1e-6), spec_smooth_reflected[i,max_idx*1]))

        

        # plt.plot(shifted_freq,  spectrum_db_norm_list[i, :], color=colors[0], label=f'$\kappa_c$={kappa_arr[i]*1e-9:.2f} ns$^{{-1}}$')


plt.axhline(half_max, color='red', linestyle='--', linewidth=2, label='Half max (FWHM)')
peak_pts = np.array(peak_pts)
plt.plot(peak_pts[:,0], peak_pts[:,1], 'gx', markersize=10, label='Peak Points')

plt.xlabel("Frequency offset from peak (MHz)", fontsize=18)
plt.ylabel("Power (dBm)", fontsize=18)
# plt.title(rf"Normalized Optical Spectrum $\kappa={kappa_c[i]*1e-9:.4f}~\mathrm{{ns}}^{{-1}}$", fontsize=18) 
plt.xlim(-15000, 15000)
plt.ylim(-200, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Add colorbar for kappa_c
norm = mpl.colors.Normalize(vmin=kappa_arr[0]*1e-9, vmax=kappa_arr[num_curves-1]*1e-9)
sm = plt.cm.ScalarMappable(cmap=cm.Blues, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, pad=0.02, ax=plt.gca())
cbar.set_label(r'$\kappa_c~(\mathrm{ns}^{-1})$', fontsize=18)
cbar.ax.tick_params(labelsize=24)
titles = {'E1': r"$E_{1}$", 'E2': r"$E_{2}$", 'Etot': r"$E_{tot}$"}

plt.title(titles[plot_field],fontsize =24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()

# plt.savefig(f"./linewidth_estimation/3db_spectra_{plot_field}.png")
plt.show()
plt.close(fig)
        # break
        


#%%

import numpy as np
from scipy.signal import find_peaks
from astropy.modeling import models
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Arrays to store FWHM for each field
fwhm_E1_list = np.zeros(len(kappa_arr))
fwhm_E2_list = np.zeros(len(kappa_arr))
fwhm_Etot_list = np.zeros(len(kappa_arr))

# Initial guess for FWHM in Hz
fwhm0_E1 = 0e6
fwhm0_E2 = 0e6
fwhm0_Etot = 0e6

plot_freq = 300e6

window_length = 1001    # must be odd
polyorder = 3



detuning = 0.0

for detuning in np.linspace(-4,4,11):
    detuning = np.round(detuning,2)
    spectrum_db_norm_list = np.load(f"./linewidth_estimation/numpy_arrays/spectrum_db_norm_list_no_self_200_avg_spectra_detuning{detuning}.npy")
    spectrum_E1_db_norm_list = np.load(f"./linewidth_estimation/numpy_arrays/spectrum_E1_db_norm_list_no_self_200_avg_spectra_detuning{detuning}.npy")
    spectrum_E2_db_norm_list= np.load(f"./linewidth_estimation/numpy_arrays/spectrum_E2_db_norm_list_no_self_200_avg_spectra_detuning{detuning}.npy")
    cos_phase_diff_time = np.load(f"./linewidth_estimation/numpy_arrays/cos_phase_diff_time_no_self_200_avg_spectra_detuning{detuning}.npy")
    f_sorted = np.load(f"./linewidth_estimation/numpy_arrays/f_sorted_no_self_200_avg_spectra_detuning{detuning}.npy")
    f_plot_min = -15   # -2 GHz
    f_plot_max = 15   # +15 GHz
    mask = (f_sorted >= f_plot_min*1e9*tau_p) & (f_sorted <= f_plot_max*1e9*tau_p)

    # --- Apply mask & normalize ---
    f_window = f_sorted[mask]/tau_p
    f_plot = f_window
    n_cases = 200
    kappa_arr = np.concatenate([np.linspace(0, kappa_max, n_cases)])



    def compute_linewidth(
        spectrum_db_norm_list, field_name, fwhm_list_db3, fwhm0, folder, plot=False
    ):
        """
        Fits Lorentzian to spectra and extracts linewidths.
        
        Parameters
        ----------
        spectrum_db_norm_list : 2D array [cases, freqs]
            Normalized spectra in dB (peak = 0 dB).
        field_name : str
            Name used for saving plots.
        fwhm_list_db3 : array
            Output array for fitted -3 dB linewidths.
        fwhm_list_db20 : array
            Output array for computed -20 dB linewidths.
        fwhm0 : float
            Initial guess for FWHM (Hz). If <= 0, estimated from data.
        folder : str
            Output folder name.
        """
        import os
        from scipy.optimize import least_squares
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        import time
        from scipy.signal import find_peaks, peak_widths

        spec_smooth_reflected = np.zeros((spectrum_db_norm_list.shape[0], 1*spectrum_db_norm_list.shape[1]), dtype=spectrum_db_norm_list.dtype)

        if plot:
            os.makedirs(f'./linewidth_estimation/{folder}', exist_ok=True)

        for k in range(len(spectrum_db_norm_list)):
            
            if kappa_arr[k]*1e-9 > -.01 and k>=0:#and kappa_arr[k]*1e-9 < 11:
                
                spec_smooth = spectrum_db_norm_list[k, :]#savgol_filter(spectrum_db_norm_list[k, :], window_length, polyorder)
                # Reflect spec_smooth about the peak (index max_idx)
                spec_smooth_reflected[k,:] = spec_smooth#np.concatenate([spec_smooth[::-1], spec_smooth])
                f_plot_reflected = f_plot#np.concatenate([-f_plot[::-1], f_plot])



                # max_idx = np.argmax(spec_smooth)
                # spec_smooth = spec_smooth - np.max(spec_smooth)
                # spec_smooth[f_plot<=0] = -1000
                # spec_smooth = spec_smooth - np.max(spec_smooth)
                max_idx = np.argmax(spec_smooth_reflected[k,:])
                freq_shift = (f_plot_reflected[max_idx])
                shifted_freq = (f_plot_reflected - freq_shift) * 1e-6
                # shifted_freq = (f_plot_reflected) * 1e-6




                # print(peak_freqs, peak_heights_dB)

                
                

                # crossings on finer grid
                # crossings = np.where(np.diff(np.sign(spec_smooth_reflected[k,:] + 3)))[0]
                # # Filter crossings by proximity to peak
                # crossings = crossings[np.abs(shifted_freq[crossings]) < plot_freq*1e-6]
                # if len(crossings) >= 1:
                #     left_group = shifted_freq[crossings][shifted_freq[crossings] < 0]
                #     right_group = shifted_freq[crossings][shifted_freq[crossings] > 0]
                #     if len(left_group) > 0 and len(right_group) > 0:
                #         left_cross = np.mean(np.abs(left_group))
                #         right_cross = np.mean(np.abs(right_group))
                #         linewidth_20db = left_cross + right_cross
                #     elif len(right_group) > 0:
                #         linewidth_20db = 2*np.mean(np.abs(right_group))
                #         left_cross = np.nan
                #         right_cross = np.mean(np.abs(right_group))
                # else:
                #     linewidth_20db = np.nan

                

                # fwhm_list_db20[k] = linewidth_20db
                
                


                # fit_curve_db = spectrum_window_db

                clear_output(wait=True)
                # --- Find peaks with prominence ---
                # Get peaks and their heights
                peaks, props = find_peaks(spec_smooth_reflected[k, :], height=0)

                if len(peaks) > 0:
                    # Get highest peak
                    highest_peak_idx = np.argmax(props["peak_heights"])
                    peak_pos = peaks[highest_peak_idx]

                    spectrum = spec_smooth_reflected[k, :]
                    freqs = shifted_freq
                    target_level = spectrum[peak_pos] - 20.0  # -3 dB from peak

                    # Left side: find first index below -3 dB
                    left_idx = np.where(spectrum[:peak_pos] <= target_level)[0]
                    if len(left_idx) > 0:
                        i = left_idx[-1]
                        freq_left = np.interp(target_level,
                                            [spectrum[i], spectrum[i+1]],
                                            [freqs[i], freqs[i+1]])
                    else:
                        freq_left = freqs[0]

                    # Right side: find first index below -3 dB
                    right_idx = np.where(spectrum[peak_pos:] <= target_level)[0]
                    if len(right_idx) > 0:
                        i = right_idx[0] + peak_pos
                        freq_right = np.interp(target_level,
                                            [spectrum[i-1], spectrum[i]],
                                            [freqs[i-1], freqs[i]])
                    else:
                        freq_right = freqs[-1]

                    fwhm_list_db3[k] = freq_right - freq_left

                else:
                    peak_pos = np.argmax(spec_smooth_reflected[k, :])
                    freq_left = np.nan
                    freq_right = np.nan
                    fwhm_list_db3[k] = np.nan

                # -----------------------------------------
                # Plotting
                # -----------------------------------------
                if plot:
                    fig = plt.figure(figsize=(8,5), dpi=150)
                    plt.plot(shifted_freq, spec_smooth_reflected[k, :], label="Spectrum")

                    # Red -20 dB line spanning peak width
                    plt.plot([freq_left, freq_right], [target_level, target_level],
                            color='red', linewidth=3,
                            label='-20 dB width')

                    # Mark peak
                    peak_freq = shifted_freq[peak_pos]
                    plt.plot(peak_freq, spec_smooth_reflected[k, peak_pos], 'ko')

                    plt.xlabel('Frequency offset (MHz)', fontsize=24)
                    plt.ylabel('Power (dBm)', fontsize=24)
                    plt.xlim(-300, 300)
                    plt.ylim(-50, 1)
                    plt.xticks(fontsize=22)
                    plt.yticks(fontsize=22)
                    plt.title(f"$\\kappa_c$={kappa_arr[k]*1e-9:.2f} ns$^{{-1}}$", fontsize=28)
                    plt.legend(fontsize=16, loc='upper right')
                    plt.tight_layout()
                    plt.savefig(f'./linewidth_estimation/{folder}/kappa_{kappa_arr[k]*1e-9:.2f}.png', dpi=150)
                    plt.show()
                    plt.close(fig)

            # break

        return fwhm_list_db3

    plot = False

    # # # Fit and plot for E_1
    fwhm_E1_list = compute_linewidth(spectrum_E1_db_norm_list, 'E_1', fwhm_E1_list, fwhm0_E1, 'E_1', plot=plot)

    # # # Fit and plot for E_2
    fwhm_E2_list = compute_linewidth(spectrum_E2_db_norm_list, 'E_2', fwhm_E2_list, fwhm0_E2, 'E_2', plot=plot)

    # Fit and plot for E_tot
    fwhm_Etot_list = compute_linewidth(spectrum_db_norm_list, 'E_tot', fwhm_Etot_list, fwhm0_Etot, 'E_tot', plot=plot)

    # Convert to MHz for easier plotting
    fwhm_E1_list_MHz = fwhm_E1_list #/ 1e6
    fwhm_E2_list_MHz = fwhm_E2_list #/ 1e6
    fwhm_Etot_list_MHz = fwhm_Etot_list #/ 1e6

    

    num_curves=200
    plt.figure(figsize=(9, 7), dpi=200)
    # Moving-average plots (NaN-safe)
    def moving_average_nan_safe(x, window):
        x = np.asarray(x, dtype=float)
        mask = np.isfinite(x).astype(float)
        x_filled = np.where(np.isfinite(x), x, 0.0)
        kernel = np.ones(int(window), dtype=float)
        num = np.convolve(x_filled, kernel, mode='same')
        den = np.convolve(mask, kernel, mode='same')
        return num / np.where(den == 0, np.nan, den)

    ma_window = 1  # adjust as needed
    E_tot_ma = moving_average_nan_safe(fwhm_Etot_list_MHz, ma_window)
    E1_ma    = moving_average_nan_safe(fwhm_E1_list_MHz,   ma_window)
    E2_ma    = moving_average_nan_safe(fwhm_E2_list_MHz,   ma_window)

    n = min(num_curves, len(kappa_arr), len(E_tot_ma), len(E1_ma), len(E2_ma))

    plt.plot(
        kappa_arr[:n]*1e-9, E_tot_ma[:n],
        linewidth=2, color='blue', label='E_tot', linestyle='-'
    )
    plt.plot(
        kappa_arr[:n]*1e-9, E1_ma[:n],
        linewidth=2, color='green', label='E_1', linestyle='--'
    )
    plt.plot(
        kappa_arr[:n]*1e-9, E2_ma[:n],
        linewidth=2, color='red', label='E_2', linestyle='-', marker='.', markersize=8, markerfacecolor='r', alpha=0.4
    )
    plt.xlabel(r'$\kappa_c~(\mathrm{ns}^{-1})$', fontsize=28)
    plt.ylabel('20 dB Linewidth (MHz)', fontsize=28)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5, which='both')


    plt.legend(loc='upper right', fontsize=20)

    # plt.xlim(7.25,11)
    plt.ylim(1e0 , 5e2) 
    plt.xticks(np.linspace(0,20,6), fontsize=24)
    plt.yticks(fontsize=24)
    plt.title("No Self Feedback", fontsize=32)
    plt.tight_layout()
    plt.savefig(f"./linewidth_estimation/fwhm_linewidth_all_fields_detuning{detuning:.2f}.png")
    plt.show()


#%%

E_list = []  # Clear E_list at the start of each outer loop iteration
E1_list = []
E2_list = []
spectrum_db_norm_list = []

h = 6.626e-34  # J·s
nu = 3e8 / 900e-9  # Hz
conversion = h * nu / tau_p  # still available if you want absolute scaling

for idx in range(50):
    # --- Compute fields ---
    E_1 = np.sqrt(y[idx, 1, :]) * (np.cos(y[idx, 2, :]) + 1j * np.sin(y[idx, 2, :]))
    E_2 = np.sqrt(y[idx, 4, :]) * (np.cos(y[idx, 5, :]) + 1j * np.sin(y[idx, 5, :]))
    E_tot = E_1 + E_2
    E_tot = np.nan_to_num(E_tot, nan=0.0)

    E_list.append(E_tot)
    E1_list.append(E_1)
    E2_list.append(E_2)

    # --- Take second half of signal to avoid transients ---
    E_seg = E_tot[int(time_pts // 2):]

    # --- Optical spectrum: FFT of field ---
    N = E_seg.size
    fft_vals = np.fft.fftshift(np.fft.fft(E_seg))
    fft_freqs = np.fft.fftshift(np.fft.fftfreq(N, dt))

    # Power spectral density in "photon units"
    psd = np.abs(fft_vals)**2 / N

    # Convert to Watts using photon energy per photon lifetime
    psd_watts = psd * conversion

    # Convert to dBm (reference = 1 mW)
    spectrum_dbm = 10 * np.log10(psd_watts / 1e-3 + 1e-20)
    spectrum_dbm = spectrum_dbm - np.max(spectrum_dbm)  # Normalize so max is 0 dB

    # --- Center the most prominent peak at 0 for each row ---
    peak_idx = np.argmax(spectrum_dbm)
    freq_shift = fft_freqs[peak_idx]
    fft_freqs_centered = fft_freqs - freq_shift

    spectrum_db_norm_list.append((fft_freqs_centered, spectrum_dbm))
    if idx == 0:
        fft_freqs_plot = fft_freqs_centered  # Save for plotting

# --- Plot ---
plt.figure(figsize=(10, 6), dpi=150)
for i in range(len(spectrum_db_norm_list)):
    freqs, spectrum = spectrum_db_norm_list[i]
    plt.plot(freqs * 1e-6, spectrum, label=f'Case {i+1}')

plt.xlabel('Frequency offset from peak (MHz)', fontsize=16)
plt.ylabel('Power (dB, normalized)', fontsize=16)
plt.title('Optical Spectrum (Centered)', fontsize=18)
# plt.legend()
# plt.ylim(-100,0)
plt.xlim(-100,100)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



#%%

font_size = 22
fig = plt.figure(figsize=(18, 10), dpi=300)
gs = fig.add_gridspec(20, 30, height_ratios=[1]*20, width_ratios=[1]*30, hspace=0.3)

# --- Precompute peak trajectories ---
n_cases = len(kappa_c)
peak_freq_tot = np.full(n_cases, np.nan)
peak_freq_E1  = np.full(n_cases, np.nan)
peak_freq_E2  = np.full(n_cases, np.nan)

for k in range(n_cases):
    # Total field
    row = spectrum_db_norm_list[k]
    peaks, _ = find_peaks(row)
    if len(peaks):
        p = peaks[np.argmax(row[peaks])]
        peak_freq_tot[k] = f_plot[p]*1e-9  # GHz
    # E1 field
    row1 = spectrum_E1_db_norm_list[k]
    peaks1, _ = find_peaks(row1)
    if len(peaks1):
        p1 = peaks1[np.argmax(row1[peaks1])]
        peak_freq_E1[k] = f_plot[p1]*1e-9
    # E2 field
    row2 = spectrum_E2_db_norm_list[k]
    peaks2, _ = find_peaks(row2)
    if len(peaks2):
        p2 = peaks2[np.argmax(row2[peaks2])]
        peak_freq_E2[k] = f_plot[p2]*1e-9

# --- Total spectrum ---
ax1 = fig.add_subplot(gs[11:, 1:8])
im = ax1.imshow(
    spectrum_db_norm_list,
    aspect='auto',
    extent=[f_plot[0]*1e-9, f_plot[-1]*1e-9, kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
    origin='lower',
    cmap='viridis',
    rasterized=True
)
im.set_clim(-200, 0)
ax1.set_xlabel("Frequency (GHz)", fontsize=font_size, labelpad=10)
ax1.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
ax1.set_title(r"$|\mathcal{F}\left(E_1+E_2\right)|^2$", fontsize=font_size, pad=16)
ax1.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
ax1.set_xticks(np.linspace(f_plot_min, f_plot_max, 3))
ax1.set_xlim(f_plot_min, f_plot_max)
ax1.set_ylim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
ax1.tick_params(axis='both', labelsize=font_size)
ax1.axvline(detuning, color='black', linestyle='--', linewidth=2, label=rf"$\delta$", alpha=0.5)
# Peak path
valid_tot = ~np.isnan(peak_freq_tot)
ax1.plot(peak_freq_tot[valid_tot], kappa_c[valid_tot]*1e-9, color='red', lw=2, alpha=0.5)
# ax1.scatter(peak_freq_tot[valid_tot], kappa_c[valid_tot]*1e-9, color='red', s=10)

# --- E1 spectrum ---
ax2 = fig.add_subplot(gs[11:, 11:18])
im2 = ax2.imshow(
    spectrum_E1_db_norm_list,
    aspect='auto',
    extent=[f_plot[0]*1e-9, f_plot[-1]*1e-9, kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
    origin='lower',
    cmap='viridis',
    rasterized=True
)
im2.set_clim(-200, 0)
ax2.set_xlabel("Frequency (GHz)", fontsize=font_size, labelpad=10)
ax2.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
ax2.set_title(r"$|\mathcal{F}\left(E_1\right)|^2$", fontsize=font_size, pad=16)
ax2.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
ax2.set_xticks(np.linspace(f_plot_min, f_plot_max, 3))
ax2.set_xlim(f_plot_min, f_plot_max)
ax2.set_ylim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
ax2.tick_params(axis='both', labelsize=font_size)
ax2.axvline(detuning, color='black', linestyle='--', linewidth=2, label=rf"$\delta$", alpha=0.5)
valid_E1 = ~np.isnan(peak_freq_E1)
ax2.plot(peak_freq_E1[valid_E1], kappa_c[valid_E1]*1e-9, color='red', lw=2, alpha=0.5)
# ax2.scatter(peak_freq_E1[valid_E1], kappa_c[valid_E1]*1e-9, color='red', s=10)

# --- E2 spectrum ---
ax3 = fig.add_subplot(gs[11:, 21:29])
im3 = ax3.imshow(
    (spectrum_E2_db_norm_list),
    aspect='auto',
    extent=[f_plot[0]*1e-9, f_plot[-1]*1e-9, kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
    origin='lower',
    cmap='viridis',
    rasterized=True
)
cbar3 = fig.colorbar(im3, ax=ax3, pad=0.05)
cbar3.set_label(r'Power (dBm)', fontsize=font_size, labelpad=12)
cbar3.ax.tick_params(labelsize=font_size)
im3.set_clim(-200, 0)
ax3.set_xlabel("Frequency (GHz)", fontsize=font_size, labelpad=10)
ax3.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
ax3.set_title(r"$|\mathcal{F}\left(E_2\right)|^2$", fontsize=font_size, pad=16)
ax3.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
ax3.set_xticks(np.linspace(f_plot_min, f_plot_max, 3))
ax3.set_xlim(f_plot_min, f_plot_max)
ax3.set_ylim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
ax3.tick_params(axis='both', labelsize=font_size)
ax3.axvline(detuning, color='black', linestyle='--', linewidth=2, label=rf"$\delta$", alpha=0.5)
valid_E2 = ~np.isnan(peak_freq_E2)
ax3.plot(peak_freq_E2[valid_E2], kappa_c[valid_E2]*1e-9, color='red', lw=2, alpha=0.5)
# ax3.scatter(peak_freq_E2[valid_E2], kappa_c[valid_E2]*1e-9, color='red', s=10)
ax3.legend(fontsize=font_size, loc='upper right')

# Optional legend on first two
ax1.legend(fontsize=font_size, loc='upper right')
ax2.legend(fontsize=font_size, loc='upper right')

plt.show()
plt.close(fig)
