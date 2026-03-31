#%%
# Time series plotting for a VCSEL model

import numpy as np
from IPython.display import clear_output
from vcsel_lib import VCSEL
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from tqdm import tqdm
import time
from joblib import Parallel, delayed
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('font', family='serif')




import matplotlib
# matplotlib.use("Agg")  # disable GUI backend
# %matplotlib inline
import matplotlib.pyplot as plt
plt.ioff()
cmap = plt.colormaps['jet']

detuning = 4.0
N_lasers = 2
delta = detuning * 2 * np.pi * 1e9
delta = np.sort(np.concatenate([delta/2*np.linspace(-1,1,N_lasers)]))
# print(delta/(2*np.pi*1e9))

all_data = []
phi_p_arr = np.linspace(0.0,2,20)



for phi_p_loop in phi_p_arr:
# phi_p_loop = 1.0


    all_eqs = []
    all_order_params = []
    sorted_indices_arr = []
    max_num_eqs = 0
    kappa_vals = np.linspace(0.0e9, 10e9, 10)

    

    folder_name = f"./all_eq_2laser_symmetric_detuning_noise"
    # os.makedirs(f"{folder_name}/detuning_{detuning:.2f}_0self_phi_p{phi_p_loop:.2f}pi_ramp_noise_injection", exist_ok=True)
    eqs = None
    stable_arr = []
    avg_time = 0
    for kappa_ind, kappa_c in enumerate(tqdm(kappa_vals)):

        
        # Physical parameters
        alpha = 2
        tau_p = 5.4e-12
        tau_n = 0.25e-9
        g0 = 8.75e-4 * 1e9
        N0 = 2.86e5
        s = 4e-6
        q = 1.602e-19
        beta = 1.e-3
        tau = 1e-9
        eta = 0.9
        current_threshold = 3

        I = eta * current_threshold * q / tau_n * (N0 + 1/(g0*tau_p))

        # Coupling / feedback / detuning
        # self_feedback = 0.5
        coupling = 1.0

        

        # Time discretization
        dt = 0.5 * tau_p#.1e-11
        Tmax = 3e-7
        steps = int(Tmax / dt)
        time_arr = np.linspace(0, Tmax, steps)
        delay_steps = int(tau / dt)

        noise_amplitude = 0.0

        
        coupling_scheme = 'ATA'
        ramp_start = 0
        ramp_shape = 0.00000001
        dx = 1.0

        kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=kappa_c, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx)

        self_feedback = 0.00

        phys = {
            'tau_p': tau_p,
            'tau_n': tau_n,
            'g0': g0,
            'N0': N0,
            'N_bar': N0 + 1/(g0*tau_p),
            's': s,
            'beta': beta,
            'kappa_c_mat': kappa_arr[-1,:,:],
            'phi_p_mat': np.ones(shape=(N_lasers,N_lasers))*phi_p_loop*np.pi,
            'I': I,
            'q': q,
            'alpha': alpha,
            'delta': delta,  # detuning for each laser
            'coupling': coupling,
            'self_feedback': self_feedback,
            'noise_amplitude': noise_amplitude,
            'dt': dt,
            'Tmax': Tmax,
            'tau': tau,
            'N_lasers': N_lasers,
            'sparse': True
        }


        vcsel = VCSEL(phys)
        nd = vcsel.scale_params()
        n_cases = len(nd['phi_p'])





        if eqs is not None:
            for eq_pt in eqs:
                if np.any(np.isnan(eq_pt)):
                    continue
                guesses.append(np.concatenate([
                    eq_pt[1::2][:N_lasers],  # S1, S2, ...
                    eq_pt[2*N_lasers:3*N_lasers-1],                      # φ1, φ2, ...
                    np.array([eq_pt[-2]])                      # ω
                ]))
        else:
            guesses = []

        # print("Additional guesses...", len(guesses))
        counts = {'phase_count': 5, 'freq_count': 50, 'max_refine':2, 'refine_factor':2}

        eq_max, eqs, _ = vcsel.solve_equilibria(nd, guesses=guesses, counts=counts)
        guesses = []

        if len(eqs) > max_num_eqs:
            max_num_eqs = len(eqs)

        tmp_stable = []

        start = time.time()
        N = 30
        n_eigenvalues = N*3*N_lasers - 1
        # print(len(eqs), n_eigenvalues)
        tmp_stable = Parallel(n_jobs=-1)(
            delayed(vcsel.compute_stability)(eq_pt, nd, N=N, newton_maxit=10000, threshold=1e-10, sparse=phys['sparse'], spectral_shift=0.01+0.01j, n_eigenvalues=n_eigenvalues)
            for eq_pt in eqs
        )
        end = time.time() - start
        avg_time += end
        tmp_stable = [result[0] for result in tmp_stable]

        # tmp_stable = [1 for eq in eqs]
        

        eqs = np.column_stack([eqs, np.array(tmp_stable)])  # add stability as last column
        stable_arr.append(np.array(tmp_stable))
        all_eqs.append(eqs)
        
    for k, eqs in enumerate(all_eqs):

        expected_cols = 2*N_lasers + (N_lasers - 1) + 2

        # Ensure eqs is a 2D numpy array with correct column size
        eqs = np.asarray(eqs)

        if eqs.size == 0:
            # Completely empty → create full NaN block
            eqs = np.nan * np.ones((max_num_eqs, expected_cols))
        
        else:
            # Ensure 2D
            if eqs.ndim == 1:
                eqs = eqs.reshape(1, -1)

            # Pad if needed
            if eqs.shape[0] < max_num_eqs:
                eq_diff = max_num_eqs - eqs.shape[0]
                pad_rows = np.nan * np.ones((eq_diff, expected_cols))
                eqs = np.vstack([eqs, pad_rows])

        all_eqs[k] = eqs

        

    all_eqs = np.array(all_eqs)

    # n1, S1, n2, S2, phase_diff, omega
    S = all_eqs[:,:,1::2][:,:,:N_lasers]

    omega = all_eqs[:,:,-2]

    phi = omega[:,:,None]*tau + np.concatenate((np.zeros(shape=(len(kappa_vals), max_num_eqs, 1)), all_eqs[:,:,2*N_lasers:3*N_lasers-1]), axis=2)


    E = np.sqrt(S) * (np.cos(phi) + 1j*np.sin(phi))



    E_tot = np.abs(np.sum(E, axis=2))**2



    import matplotlib.pyplot as plt
    import matplotlib as mpl


    n_kappa = all_eqs.shape[0]
    n_branches = all_eqs.shape[1]

    # reconstruct kappa values used in the loop (in Hz) and convert to ns^-1 for plotting
    # kappa_vals = np.linspace(1e9, 5e9, n_kappa)        # Hz
    kappa_plot = np.array(kappa_vals) * 1e-9                     # ns^-1

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Create discrete colormap for stable/unstable
    colors_discrete = ['red', 'blue']  # unstable, stable
    cmap_discrete = mpl.colors.ListedColormap(colors_discrete)
    norm_discrete = mpl.colors.BoundaryNorm([0, 0.5, 1.0], cmap_discrete.N)

    # Plot all points colored by stability (0 or 1)
    for b in range(len(kappa_plot)):
        Et = omega[b,:]/(2*np.pi*1e9*tau_p)
        E_mask = np.isfinite(Et)
        stable_mask = np.isfinite(stable_arr[b])
        if not np.any(E_mask):
            continue
        if not np.any(stable_mask):
            continue
        Et = Et[E_mask]
        stable = stable_arr[b][stable_mask]
        
        # Plot all points with discrete stable/unstable coloring
        scatter = ax.scatter(kappa_plot[b]*np.ones_like(Et), Et,
                c=stable, cmap=cmap_discrete, norm=norm_discrete,
                s=5, edgecolors='none')

    ax.set_xlabel(r'$\kappa_c$ (ns$^{-1}$)', fontsize=20)
    ax.set_ylabel(r'$\omega$ (GHz)', fontsize=20)
    ax.set_title(rf'$\delta = {detuning:.2f}$ GHz    $~~~\phi_p = {phi_p_loop:.2f}\pi$', fontsize=22, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(alpha=0.25)
    # ax.set_ylim(-10,10)

    # Discrete colorbar for stability
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02, boundaries=[0, 0.5, 1.0], ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Unstable', 'Stable'], fontsize=16, rotation=90)
    
    
    # plt.xlim(0.0,20.5)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    all_data.append(all_eqs)









#%%
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

# WARNING - this will overwrite existing data. Make sure to back up any important data before running.
# Prompt user for confirmation before running
user_input = input("Are you sure you want to overwrite the data? (yes/no): ")
if user_input.lower() != 'yes':
    print("Cell execution cancelled.")
    raise SystemExit("User cancelled execution")

for p, item in enumerate(all_data):
    item_with_phi = np.pad(item, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=np.nan)
    item_with_phi[:, :, -1] = phi_p_arr[p]
    all_data[p] = item_with_phi


all_data_stacked = np.concatenate(all_data, axis=1)

# Save as dictionary with requested keys
data_dict = {
    'all_data': all_data_stacked,
    'kappa_vals': kappa_vals,
    'detuningGHZ': delta/2 * np.pi * 1e9,
    'params': phys
}

np.save(f'../data/all_equilibria_data_{N_lasers}laser_{detuning:.2f}Ghz.npy', data_dict)

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################


#%%
# Load dictionary and extract data
N_lasers = 2
detuning = 4.0
save = False
data_dict = np.load(f'../data/all_equilibria_data_{N_lasers}laser_{detuning:.2f}Ghz.npy', allow_pickle=True).item()
all_data_stacked = data_dict['all_data']
kappa_vals = data_dict['kappa_vals']
kappa_plot = np.array(kappa_vals) * 1e-9
detuning_arr = data_dict['detuningGHZ']

stable = all_data_stacked[:,:,-2]
omega = all_data_stacked[:,:,-3]/(2*np.pi*1e9*tau_p)
phi_p = all_data_stacked[:,:,-1]


s_all = all_data_stacked[:,:,1::2][:,:,:N_lasers]
n_all = all_data_stacked[:,:,0::2][:,:,:N_lasers]
phi_all = all_data_stacked[:,:,2*N_lasers:3*N_lasers-1]
phi_all = phi_all % (2 * np.pi)


phi = omega[:,:,None]*tau + np.concatenate((np.zeros(shape=(len(kappa_vals), all_data_stacked.shape[1], 1)), all_data_stacked[:,:,2*N_lasers:3*N_lasers-1]), axis=2)


E = np.sqrt(s_all) * (np.cos(phi) + 1j*np.sin(phi))

E_tot = np.abs(np.sum(E, axis=2))**2

N_tot = n_all[:,:,0]





#%%

import plotly.graph_objects as go
import numpy as np

desired_phi_p = 0
i=-1
valid_mask = (
            np.isfinite(omega[i,:]) &
            np.isfinite(phi_p[i,:]) &
            np.isfinite(stable[i,:]) &
            (phi_p[i,:] == phi_p_arr[desired_phi_p])
        )


flat_omega = omega.flatten()
flat_omega = flat_omega[np.isfinite(flat_omega)]
freq_range = [np.floor(np.min(flat_omega)), np.ceil(np.max(flat_omega))]



# Prepare data for all phi_p values
all_frames_data = []

for desired_phi_p in range(len(phi_p_arr)):
    x_all = []
    y_all = []
    z_all = []
    c_all = []
    
    for i, kap in enumerate(kappa_plot):
        valid = (
            np.isfinite(omega[i,:]) &
            np.isfinite(phi_p[i,:]) &
            np.isfinite(stable[i,:]) &
            np.isin(phi_p[i,:], phi_p_arr[desired_phi_p])
        )

        if not np.any(valid):
            continue
                
        x_all.append(np.full(np.sum(valid), kap))
        y_all.append(omega[i, valid])
        z_all.append(E_tot[i, valid]*1)
        c_all.append(stable[i, valid])

    if x_all:
        x = np.concatenate(x_all)
        y = np.concatenate(y_all)
        z = np.concatenate(z_all)
        c = np.concatenate(c_all)
    else:
        x = y = z = c = np.array([])
    
    all_frames_data.append({'x': x, 'y': y, 'z': z, 'c': c})

# Create initial trace
fig = go.Figure(
    data=[go.Scatter3d(
        x=all_frames_data[0]['x'],
        y=all_frames_data[0]['y'],
        z=all_frames_data[0]['z'],
        mode='markers',
        marker=dict(
            size=1,
            color=all_frames_data[0]['c'],
            colorscale=[[0,'red'], [1,'blue']],
            opacity=1,
            colorbar=dict(title='Stability', tickvals=[0, 1], ticktext=['Unstable', 'Stable'])
        )
    )]
)

# Create frames for slider
frames = [
    go.Frame(
        data=[go.Scatter3d(
            x=all_frames_data[k]['x'],
            y=all_frames_data[k]['y'],
            z=all_frames_data[k]['z'],
            marker=dict(
                size=1,
                color=all_frames_data[k]['c'],
                colorscale=[[0,'red'], [1,'blue']],
                opacity=1
            )
        )],
        name=str(k)
    )
    for k in range(len(phi_p_arr))
]

fig.frames = frames
# np.nanmax(E_tot)
# kappa_vals[-1]*1e-9
# freq_range[0], freq_range[1]
# Add slider
fig.update_layout(
    scene=dict(
        xaxis=dict(title='κc (ns⁻¹)', range=[0, 10]),
        yaxis=dict(title='ω (GHz)', range=[freq_range[0], freq_range[1]]),
        zaxis=dict(title='E_tot', range=[np.nanmin(E_tot), 20]),
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, b=40, t=40),
    title=f'3D Equilibrium Branches (φ_p = {phi_p_arr[0]:.3f}π)',
    updatemenus=[],
    sliders=[dict(
        active=0,
        yanchor='top',
        y=0,
        xanchor='left',
        currentvalue=dict(
            prefix='φ_p index: ',
            visible=True,
            xanchor='right'
        ),
        steps=[dict(
            args=[[f.name], dict(
                frame=dict(duration=0, redraw=True),
                mode='immediate',
                transition=dict(duration=0)
            )],
            label=f'{k}',
            method='animate'
        ) for k, f in enumerate(fig.frames)]
    )]
)

fig.show()
# fig.update_layout(
#     sliders=[dict(
#         active=0,
#         yanchor='top',
#         y=0,
#         xanchor='left',
#         currentvalue=dict(
#             prefix='φ_p index: ',
#             visible=True,
#             xanchor='right'
#         ),
#         transition=dict(duration=0),
#         steps=[dict(
#             args=[[f.name], dict(
#                 frame=dict(duration=0, redraw=True),
#                 mode='immediate',
#                 transition=dict(duration=0)
#             )],
#             label=f'{k}',
#             method='animate'
#         ) for k, f in enumerate(fig.frames)]
#     )]
# )

if data_dict['params']['sparse']:
    computation_type = 'sparse'
else:
    computation_type = 'dense'


save = False
if save:
    fig.write_html(f"../HTML_files/branches_{N_lasers}laser_{detuning:.0f}Ghz_{computation_type}.html", auto_play=False)



#%%

import plotly.graph_objects as go
import numpy as np

# Subsample parameter (e.g., keep every nth point along kappa axis)
subsample = 1  # adjust this value to control density

# Plot only a subset of kappa values (in ns^-1)
kappa_plot_min = 0.0
kappa_plot_max = 10.0

# Prepare data for all phi_p values
x_all = []
y_all = []
z_all = []
c_all = []
phi_p_colors = []

z_var = E_tot

valid_kappa_idx = np.where((kappa_plot >= kappa_plot_min) & (kappa_plot <= kappa_plot_max))[0]
valid_kappa_idx = valid_kappa_idx[::subsample]

for desired_phi_p in range(len(phi_p_arr)):
    for original_i in valid_kappa_idx:
        kap = kappa_plot[original_i]
        valid = (
            np.isfinite(omega[original_i,:]) &
            np.isfinite(phi_p[original_i,:]) &
            np.isfinite(stable[original_i,:]) &
            np.isin(phi_p[original_i,:], phi_p_arr[desired_phi_p])
        )

        if not np.any(valid):
            continue

        x_all.append(np.full(np.sum(valid), kap))
        y_all.append(omega[original_i, valid])
        z_all.append(z_var[original_i, valid])
        c_all.append(stable[original_i, valid])
        phi_p_colors.append(np.full(np.sum(valid), phi_p_arr[desired_phi_p]))

if x_all:
    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    z = np.concatenate(z_all)
    c = np.concatenate(c_all)
    phi_p_color = np.concatenate(phi_p_colors)
else:
    x = y = z = c = phi_p_color = np.array([])

# Create figure with discrete colormap for stability
fig = go.Figure(
    data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=1,
            color=c,
            colorscale=[[0, 'red'], [1, 'blue']],
            opacity=1,
            cmin=0,
            cmax=1,
            colorbar=dict(
                title='Stability',
                tickvals=[0.25, 0.75],
                ticktext=['Unstable', 'Stable']
            )
        )
    )]
)



# dict(
#         xaxis=dict(title='κc (ns⁻¹)', range=[0, kappa_vals[-1]*1e-9]),
#         yaxis=dict(title='ω (GHz)', range=[freq_range[0], freq_range[1]]),
#         zaxis=dict(title=r'|E_tot|^2', range=[np.nanmin(z_var), np.nanmax(z_var)*1.1]),
#         aspectmode='cube'
#     )
fig.update_layout(
    scene=dict(
        xaxis=dict(title='κc (ns⁻¹)', range=[kappa_plot_min, kappa_plot_max]),
        yaxis=dict(title='ω (GHz)', range=[-2,2]),
        zaxis=dict(title='|E_tot|^2', range=[np.nanmin(E_tot), 20]),
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, b=40, t=40),
    title='3D Equilibrium Branches'
)

fig.show()
# fig.write_html(f"branches_all_phi_p_{N_lasers}laser_{detuning:.0f}Ghz_carrier.html", auto_play=False)
if data_dict['params']['sparse']:
    computation_type = 'sparse'
else:
    computation_type = 'dense'


if save:
    fig.write_html(f"../HTML_files/manifold_{N_lasers}laser_{detuning:.2f}Ghz_{computation_type}.html", auto_play=False)


#%%
import matplotlib as mpl
desired_phi_p = 0
i=-1
valid_mask = (
            np.isfinite(omega[i,:]) &
            np.isfinite(phi_p[i,:]) &
            np.isfinite(stable[i,:]) &
            (phi_p[i,:] == phi_p_arr[desired_phi_p])
        )

freq_range = [np.floor(np.min(omega[i, valid_mask])),np.ceil(np.max(omega[i, valid_mask]))]


# for desired_phi_p in range(50):

#     test_data = all_data_stacked[all_data_stacked[:,:,-1]==phi_p_arr[desired_phi_p]].reshape((len(kappa_vals), -1, all_data_stacked.shape[2]))[:,:,-3]

# Create discrete colormap for stable/unstable
colors_discrete = ['red', 'blue']  # unstable, stable
cmap_discrete = mpl.colors.ListedColormap(colors_discrete)
norm_discrete = mpl.colors.BoundaryNorm([0, 0.5, 1.0], cmap_discrete.N)

fig = plt.figure(figsize=(9, 6), dpi=300)
for i, kap in enumerate(kappa_vals):
    valid_mask = (
            np.isfinite(omega[i,:]) &
            np.isfinite(phi_p[i,:]) &
            np.isfinite(stable[i,:]) &
            (phi_p[i,:] == phi_p_arr[desired_phi_p])
        )
    x = kap*np.ones(np.shape(omega[i,:][valid_mask]))*1e-9
    y = omega[i, valid_mask]
    c = stable[i, valid_mask]
    plt.scatter(x, y, s=5, c=c, cmap=cmap_discrete, norm=norm_discrete, edgecolors='none')

plt.xlim(0,20)
# plt.ylim(1.5,2.6)
cbar = plt.colorbar(boundaries=[0, 0.5, 1.0], ticks=[0.25, 0.75])
cbar.ax.set_yticklabels(['Unstable', 'Stable'], fontsize=14, rotation=90)
plt.show()
plt.close(fig)
clear_output(wait=True)



# #%%

# import numpy as np

# x = np.sort(omega[i, valid_mask])
# e = E_tot[i, valid_mask]

# # Sort E_tot in the same order as omega
# sort_idx = np.argsort(omega[i, valid_mask])
# x_sorted = omega[i, valid_mask][sort_idx]
# e_sorted = e[sort_idx]

# # Filter where E_tot > 15
# mask = e_sorted > 16
# x_filtered = x_sorted[mask]
# omega_filtered = x_sorted[mask]

# # Compute first difference
# dx = np.diff(x_filtered)

# fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
# ax.plot(omega_filtered[:-1], dx, '.-', linewidth=2, markersize=8)
# ax.set_ylim(0, 2)
# ax.set_xlabel(r'$\omega$ (GHz)', fontsize=20)
# ax.set_ylabel('First Difference', fontsize=20)
# ax.set_title('Frequency Difference', fontsize=22, pad=20)
# ax.tick_params(axis='both', which='major', labelsize=18)
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()
# plt.close(fig)




#%%
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

n_frames = 20


# Leg 1: Ramp E_tot 
anim_coefs = np.array([np.ones(n_frames),np.linspace(0,1,n_frames)])

# Leg 2: All omega -> 8.1
anim_coefs = np.hstack([anim_coefs,[np.linspace(1,0,n_frames),np.ones(n_frames)]])

# Leg 3: Bring back omega
anim_coefs = np.hstack([anim_coefs,[np.linspace(0,1,n_frames),np.ones(n_frames)]])

# Leg 4: Decrease E_tot
anim_coefs = np.hstack([anim_coefs,[np.ones(n_frames),np.linspace(1,0,n_frames)]])

frame_num = 0

for a,b in zip(anim_coefs[0], anim_coefs[1]):
    fig = plt.figure(figsize=(10, 8), dpi=300)

    # 2 columns: big plot + skinny colorbar
    gs = GridSpec(
        1, 2,
        width_ratios=[30, 1],   # control colorbar width
        wspace=0.05,            # gap between them
        left=0.12, right=0.92,  # border
        bottom=0.12, top=0.9
    )

    ax = fig.add_subplot(gs[0], projection='3d')
    cax = fig.add_subplot(gs[1])  # dedicated colorbar axis

    # Create discrete colormap for stable/unstable
    colors_discrete = ['red', 'blue']  # unstable, stable
    cmap_discrete = mpl.colors.ListedColormap(colors_discrete)
    norm_discrete = mpl.colors.BoundaryNorm([0, 0.5, 1.0], cmap_discrete.N)

    desired_phi_p = 0
    # for desired_phi_p in range(50):




    for i, kap in enumerate(kappa_plot):

        valid_mask = (
            np.isfinite(omega[i,:]) &
            np.isfinite(phi_p[i,:]) &
            np.isfinite(stable[i,:]) &
            (phi_p[i,:] == phi_p_arr[desired_phi_p])
        )

        if not np.any(valid_mask):
            continue

        scatter = ax.scatter(
            np.full_like(omega[i, valid_mask], kap),
            np.full_like(omega[i, valid_mask], freq_range[1])*(1-a) + a*omega[i, valid_mask],
            E_tot[i, valid_mask]*b,
            c=stable[i, valid_mask],
            cmap=cmap_discrete,
            norm=norm_discrete,
            s=2,
            alpha=0.5,
            edgecolors='none'
        )
        
        # Project onto xy plane (z=0)
        ax.scatter(
            np.full_like(omega[i, valid_mask], kap),
            omega[i, valid_mask],
            np.zeros_like(E_tot[i, valid_mask]),
            c='k',
            s=1,
            alpha=0.1,
            edgecolors='none'
        )

        ax.scatter(
            np.full_like(omega[i, valid_mask], kap),
            np.zeros_like(omega[i, valid_mask])+freq_range[1],
            E_tot[i, valid_mask],
            c='k',
            s=1,
            alpha=0.1,
            edgecolors='none'
        )

    ax.set_xlabel(r'$\kappa_c$ (ns$^{-1}$)', labelpad=8, fontsize=18)
    ax.set_ylabel(r'$\omega$ (GHz)', labelpad=8, fontsize=18)
    ax.set_zlabel(r'$|E_{tot}|^2$', labelpad=12, fontsize=18)

    ax.set_title(f'Equilibrium Branches ($\phi_p={phi_p_arr[desired_phi_p]:.2f}\pi$)', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.view_init(elev=20, azim=250)

    ax.set_xlim(0,20)
    ax.set_ylim(freq_range[0], freq_range[1])
    ax.set_zlim(0, np.nanmax(E_tot)*1.1)

    # Set grid opacity lower
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0, 0, 0, 0.05)  # RGBA with alpha

    # colorbar in its own slot (does NOT resize the 3D axes)
    cbar = fig.colorbar(scatter, cax=cax, boundaries=[0, 0.5, 1.0], ticks=[0.25, 0.75])
    cbar.set_label('Stability', fontsize=16)
    cbar.ax.set_yticklabels(['Unstable', 'Stable'], rotation=90, fontsize=14)

    # plt.savefig(f'./3d_branch_anim/{frame_num}.png', bbox_inches='tight', transparent=False)
    plt.show()
    clear_output(wait=True)
    frame_num += 1




#%%
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from IPython.display import clear_output
n_frames = 200

# Leg 1: Ramp E_tot 
anim_coefs = np.array([np.ones(n_frames),np.linspace(0,1,n_frames)])

# Leg 2: All omega -> 8.1
anim_coefs = np.hstack([anim_coefs,[np.linspace(1,0,n_frames),np.ones(n_frames)]])

# Leg 3: Bring back omega
anim_coefs = np.hstack([anim_coefs,[np.linspace(0,1,n_frames),np.ones(n_frames)]])

# Leg 4: Decrease E_tot
anim_coefs = np.hstack([anim_coefs,[np.ones(n_frames),np.linspace(1,0,n_frames)]])

frame_num = 0

# for desired_phi_p in range(1,50):
desired_phi_p = -1
for plane_kappa in kappa_plot:
    plane_kappa = kappa_plot[-1]
    fig = plt.figure(figsize=(10, 8), dpi=300)

    # 2 columns: big plot + skinny colorbar
    gs = GridSpec(
        1, 2,
        width_ratios=[30, 1],   # control colorbar width
        wspace=0.05,            # gap between them
        left=0.12, right=0.92,  # border
        bottom=0.12, top=0.9
    )

    ax = fig.add_subplot(gs[0], projection='3d',computed_zorder=False)
    cax = fig.add_subplot(gs[1])  # dedicated colorbar axis

    # Create discrete colormap for stable/unstable
    colors_discrete = ['red', 'blue']  # unstable, stable
    cmap_discrete = mpl.colors.ListedColormap(colors_discrete)
    norm_discrete = mpl.colors.BoundaryNorm([0, 0.5, 1.0], cmap_discrete.N)

    # Plot gray plane at kappa=plane_kappa
    omega_grid = np.linspace(-3, 8.1, 50)
    E_tot_grid = np.linspace(0, np.nanmax(E_tot)*1.1, 50)
    Omega_mesh, E_mesh = np.meshgrid(omega_grid, E_tot_grid)
    Kappa_mesh = np.full_like(Omega_mesh, plane_kappa)
    
    # ax.plot_surface(
    #     Kappa_mesh, Omega_mesh, E_mesh,
    #     color='gray',
    #     alpha=1,
    #     shade=False,
    #     edgecolor='none',
    #     zorder=0
    # )
    # Vectorize the loop by processing all kappa values at once
    valid_mask = (
        np.isfinite(omega) &
        np.isfinite(phi_p) &
        np.isfinite(stable) &
        np.isin(phi_p, phi_p_arr[:desired_phi_p])
    )

    # Split data into before and after plane_kappa
    kappa_mesh = kappa_plot[:, np.newaxis]
    before_plane = kappa_mesh < plane_kappa
    after_plane = ~before_plane

    # Mask for points before plane
    mask_before = valid_mask & before_plane
    if np.any(mask_before):
        i_before, j_before = np.where(mask_before)
        scatter = ax.scatter(
            kappa_plot[i_before],
            omega[mask_before],
            E_tot[mask_before],
            c=stable[mask_before],
            cmap=cmap_discrete,
            norm=norm_discrete,
            s=2,
            alpha=1,
            edgecolors='none',
            depthshade=False,
            zorder=1
        )

    # Mask for points after plane
    mask_after = valid_mask & after_plane
    if np.any(mask_after):
        i_after, j_after = np.where(mask_after)
        scatter = ax.scatter(
            kappa_plot[i_after],
            omega[mask_after],
            E_tot[mask_after],
            c=stable[mask_after],
            cmap=cmap_discrete,
            norm=norm_discrete,
            s=2,
            alpha=1,
            edgecolors='none',
            depthshade=False,
            zorder=-1
        )
        
        # # Project onto xy plane (z=0)
        # ax.scatter(
        #     np.full_like(omega[i, valid_mask], kap),
        #     omega[i, valid_mask],
        #     np.zeros_like(E_tot[i, valid_mask]),
        #     c='k',
        #     s=1,
        #     alpha=0.1,
        #     edgecolors='none'
        # )

        # ax.scatter(
        #     np.full_like(omega[i, valid_mask], kap),
        #     np.zeros_like(omega[i, valid_mask])+8.1,
        #     E_tot[i, valid_mask],
        #     c='k',
        #     s=1,
        #     alpha=0.1,
        #     edgecolors='none'
        # )

    ax.set_xlabel(r'$\kappa_c$ (ns$^{-1}$)', labelpad=8, fontsize=18)
    ax.set_ylabel(r'$\omega$ (GHz)', labelpad=8, fontsize=18)
    ax.set_zlabel(r'$|E_{tot}|^2$', labelpad=12, fontsize=18)

    ax.set_title(f'Equilibrium Branches ($\phi_p\in[0, {phi_p_arr[desired_phi_p]:.2f}\pi]$)', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.view_init(elev=20, azim=250)

    ax.set_xlim(0,20)
    ax.set_ylim(freq_range[0], freq_range[1])
    ax.set_zlim(0, np.nanmax(E_tot)*1.1)

    # Set grid opacity lower
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0, 0, 0, 0.05)  # RGBA with alpha

    # colorbar in its own slot (does NOT resize the 3D axes)
    cbar = fig.colorbar(scatter, cax=cax, boundaries=[0, 0.5, 1.0], ticks=[0.25, 0.75])
    cbar.set_label('Stability', fontsize=16)
    cbar.ax.set_yticklabels(['Unstable', 'Stable'], rotation=90, fontsize=14)

    # plt.savefig(f'./3d_branch_anim_sweeping_plane/{frame_num}.png', bbox_inches='tight', transparent=False)
    plt.show()
    clear_output(wait=True)
    frame_num += 1
    break









#%%
from IPython.display import clear_output
import matplotlib.pyplot as plt

# Optional: fix axis limits so the plot doesn't jump
omega_min, omega_max = np.nanmin(omega), np.nanmax(omega)
E_min, E_max       = np.nanmin(E_tot), np.nanmax(E_tot)


for i in range(len(kappa_plot)):
    
    fig, ax = plt.subplots(figsize=(9,6), dpi=300)
    
    # Create discrete colormap for stable/unstable
    colors_discrete = ['red', 'blue']  # unstable, stable
    cmap_discrete = mpl.colors.ListedColormap(colors_discrete)
    norm_discrete = mpl.colors.BoundaryNorm([0, 0.5, 1.0], cmap_discrete.N)
    
    scatter = ax.scatter(
        omega[i,:],
        E_tot[i,:],
        c=stable[i,:],
        cmap=cmap_discrete,
        norm=norm_discrete,
        s=1,
        alpha=1
    )

    ax.set_xlabel(r"$\omega$ (GHz)", fontsize=20)
    ax.set_ylabel(r"$|E_{\mathrm{tot}}|^2$", fontsize=20)
    ax.set_title(rf"$\kappa_c = {kappa_vals[i]*1e-9:.2f}$ $ns^{{-1}}$", fontsize=22)

    ax.set_xlim(omega_min, omega_max)
    ax.set_ylim(E_min, E_max)

    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=18)

    # Add colorbar with discrete labels
    cbar = fig.colorbar(scatter, ax=ax, boundaries=[0, 0.5, 1.0], ticks=[0.25, 0.75])
    cbar.set_label('Stability', fontsize=16)
    cbar.ax.set_yticklabels(['Unstable', 'Stable'], fontsize=14, rotation=90)

    fig.tight_layout()

    # plt.savefig(f'./2d_branch_anim/{i}.png', bbox_inches='tight', transparent=False)
    plt.show()
    plt.close(fig)
    clear_output(wait=True)


