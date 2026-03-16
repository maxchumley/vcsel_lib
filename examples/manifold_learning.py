#%%
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from vcsel_lib import VCSEL
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from tqdm import tqdm
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('font', family='serif')

N_lasers = 2
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

phi_p_arr = np.linspace(0.0,2,50)


folder_name = f"./all_eq_2laser_symmetric_detuning_noise"
kappa_vals = np.linspace(0e9, 20e9, 400)
kappa_plot = np.array(kappa_vals) * 1e-9

all_data_stacked = np.load(f'{folder_name}/branches/all_equilibria_data.npy') 
stable = all_data_stacked[:,:,-2]
omega = all_data_stacked[:,:,-3]/(2*np.pi*1e9*tau_p)
phi_p = all_data_stacked[:,:,-1]


s_all = all_data_stacked[:,:,1::2][:,:,:N_lasers]
phi_all = all_data_stacked[:,:,2::2][:,:,:N_lasers]
n_all = all_data_stacked[:,:,0::2][:,:,:N_lasers]


phi = omega[:,:,None]*tau + np.concatenate((np.zeros(shape=(len(kappa_vals), 3706, 1)), all_data_stacked[:,:,2*N_lasers:3*N_lasers-1]), axis=2)


E = np.sqrt(s_all) * (np.cos(phi) + 1j*np.sin(phi))

E_tot = np.abs(np.sum(E, axis=2))**2

N_tot = np.sum(n_all, axis=2)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from IPython.display import clear_output


# for desired_phi_p in range(1,50):
desired_phi_p = -1

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

# Vectorize the loop by processing all kappa values at once
valid_mask = (
    np.isfinite(omega) &
    np.isfinite(phi_p) &
    np.isfinite(stable) &
    np.isin(phi_p, phi_p_arr[:desired_phi_p])
)

# Split data into before and after plane_kappa
kappa_mesh = kappa_plot[:, np.newaxis]


# Find maximum E_tot for each kappa value where stable==1
max_E_per_kappa = []
max_omega_per_kappa = []

for i in range(len(kappa_plot)):
    # Get valid points for this kappa that are stable
    valid_stable = valid_mask[i, :] & (stable[i, :] == 1)
    
    if np.any(valid_stable):
        # Find index of maximum E_tot among stable points
        max_idx = np.argmax(E_tot[i, valid_stable])
        # Get the actual indices in the original array
        valid_indices = np.where(valid_stable)[0]
        actual_max_idx = valid_indices[max_idx]
        
        max_E_per_kappa.append(E_tot[i, actual_max_idx])
        max_omega_per_kappa.append(omega[i, actual_max_idx])
    else:
        max_E_per_kappa.append(np.nan)
        max_omega_per_kappa.append(np.nan)

max_E_per_kappa = np.array(max_E_per_kappa)
max_omega_per_kappa = np.array(max_omega_per_kappa)

# Get phi_p values for maximum E_tot at each kappa
max_phi_p_per_kappa = []
for i in range(len(kappa_plot)):
    valid_stable = valid_mask[i, :] & (stable[i, :] == 1)
    if np.any(valid_stable):
        max_idx = np.argmax(E_tot[i, valid_stable])
        valid_indices = np.where(valid_stable)[0]
        actual_max_idx = valid_indices[max_idx]
        max_phi_p_per_kappa.append(phi_p[i, actual_max_idx])
    else:
        max_phi_p_per_kappa.append(np.nan)

max_phi_p_per_kappa = np.array(max_phi_p_per_kappa)

# Plot curve colored by phi_p value
valid_max = np.isfinite(max_E_per_kappa) & np.isfinite(max_omega_per_kappa) & np.isfinite(max_phi_p_per_kappa)
curve_scatter = ax.scatter(
    kappa_plot[valid_max],
    max_omega_per_kappa[valid_max],
    max_E_per_kappa[valid_max],
    c=max_phi_p_per_kappa[valid_max],
    cmap='cool',
    s=50,
    label='Max $|E_{tot}|^2$ (stable)',
    zorder=10,
    edgecolors='None',
    linewidths=0.5
)

# Add colorbar for the curve
curve_cbar = plt.colorbar(curve_scatter, ax=ax, pad=0.1, shrink=0.6)
curve_cbar.set_label(r'$\phi_p$ ($\pi$)', fontsize=14)


scatter = ax.scatter(
    kappa_plot[np.where(valid_mask)[0]],
    omega[valid_mask],
    E_tot[valid_mask],
    c=stable[valid_mask],
    cmap=cmap_discrete,
    norm=norm_discrete,
    s=2,
    alpha=.1,
    edgecolors='none',
    depthshade=False,
    zorder=1
)


ax.set_xlabel(r'$\kappa_c$ (ns$^{-1}$)', labelpad=8, fontsize=18)
ax.set_ylabel(r'$\omega$ (GHz)', labelpad=8, fontsize=18)
ax.set_zlabel(r'$|E_{tot}|^2$', labelpad=12, fontsize=18)

ax.set_title(f'Equilibrium Branches ($\phi_p\in[0, {phi_p_arr[desired_phi_p]:.2f}\pi]$)', fontsize=20)

ax.tick_params(axis='both', which='major', labelsize=14)

ax.view_init(elev=20, azim=230)

ax.set_xlim(0,20)
ax.set_ylim(-3,8.1)
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


#%%

import plotly.graph_objects as go
import numpy as np

# Prepare data for all phi_p values
all_frames_data = []

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

detuning = 4.0
delta = detuning * 2 * np.pi * 1e9

# Time discretization
dt = 0.5 * tau_p#.1e-11
Tmax = 3e-7
steps = int(Tmax / dt)
time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)

noise_amplitude = 0.0#VCSEL.cosine_ramp(time_arr, 50*tau, 50*tau, kappa_initial=0, kappa_final=1)

N_lasers = 2
coupling_scheme = 'ATA'
ramp_start = 0
ramp_shape = 0.00000001
dx = 1.0
n_iterations = 1

kappa_c = 20e9
phi_p_arr = np.linspace(0.0,2,50)
# phi_p = 0.0

kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=kappa_c, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx)

self_feedback = 0.00

def compute_potential(params, Omg):

    tau = params['tau']
    alpha = params['alpha']
    delta = params['delta']
    kappa_c_mat = params['kappa_c_mat']
    phi_p_mat = params['phi_p_mat']

    theta = np.arctan(alpha)

    # Omg is (1000,)
    Omg = Omg*(2 * np.pi * 1e9)   # (1000,1)
    Omega_exp = Omg[:, None, None]   # (1000,1,1)

    # First term
    V1 = 1/(2*tau) * (delta - Omg[:,None])**2   # (1000,2)

    # Coupling term
    cos_term = np.cos(Omega_exp*tau + theta + phi_p_mat)  # (1000,2,2)

    V2 = -np.sqrt(1+alpha**2)/(tau**2) * np.sum(
            kappa_c_mat * cos_term,
            axis=2
        )   # (1000,2)

    V = V1 + V2

    return V

for desired_phi_p in range(50):
    x_all = []
    y_all = []
    z_all = []
    c_all = []
    
    
    for i, kap in enumerate(kappa_plot):
        phys = {
            'tau_p': tau_p,
            'tau_n': tau_n,
            'g0': g0,
            'N0': N0,
            'N_bar': N0 + 1/(g0*tau_p),
            's': s,
            'beta': beta,
            'kappa_c_mat': 1e9*(np.ones_like(kappa_arr[-1,:,:])*kap - kap * np.eye(N_lasers)),
            'phi_p_mat': np.ones(shape=(N_lasers,N_lasers))*phi_p_arr[desired_phi_p]*np.pi,
            'I': I,
            'q': q,
            'alpha': alpha,
            'delta': np.sort(np.concatenate([delta*np.linspace(0,1,N_lasers)])),  # detuning for each laser
            'coupling': coupling,
            'self_feedback': self_feedback,
            'noise_amplitude': noise_amplitude,
            'dt': dt,
            'Tmax': Tmax,
            'tau': tau,
            'N_lasers': N_lasers
        }
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
        z_all.append(compute_potential(phys,omega[i, valid]))
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
        z=compute_potential(phys,y)[:,0],
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
frames = []
for k in range(50):
    phys['phi_p_mat'] = np.ones(shape=(N_lasers,N_lasers))*phi_p_arr[k]*np.pi
    frames.append(
        go.Frame(
            data=[go.Scatter3d(
                x=all_frames_data[k]['x'],
                y=all_frames_data[k]['y'],
                z=compute_potential(phys, all_frames_data[k]['y'])[:,0],
                marker=dict(
                    size=1,
                    color=all_frames_data[k]['c'],
                    colorscale=[[0,'red'], [1,'blue']],
                    opacity=1
                )
            )],
            name=str(k)
        )
    )

fig.frames = frames

# Add slider
fig.update_layout(
    scene=dict(
        xaxis=dict(title='κc (ns⁻¹)', range=[0, 20]),
        yaxis=dict(title='ω (GHz)', range=[-3, 8.1]),
        zaxis=dict(title='V', range=[-0.2e30, 1.6e30]),
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

fig.write_html("potential.html", auto_play=False)



#%%

import numpy as np
import matplotlib.pyplot as plt
from vcsel_lib import VCSEL

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

detuning = 4.0
delta = detuning * 2 * np.pi * 1e9

# Time discretization
dt = 0.5 * tau_p#.1e-11
Tmax = 3e-7
steps = int(Tmax / dt)
time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)

noise_amplitude = 0.0#VCSEL.cosine_ramp(time_arr, 50*tau, 50*tau, kappa_initial=0, kappa_final=1)

N_lasers = 2
coupling_scheme = 'ATA'
ramp_start = 0
ramp_shape = 0.00000001
dx = 1.0
n_iterations = 1

kappa_c = 10e9
phi_p = 0.0

self_feedback = 0.00

phi_p = all_data_stacked[:,:,-1]
phi_p_arr = np.linspace(0.0,2,50)

for desired_phi_p in range(50):
    # phi_p = 0.0
    
    for kappa_ind, kappa_c in enumerate(kappa_vals):

        fig = plt.figure(figsize=(8, 6), dpi=200)

        kappa_ind = 250
        kappa_c = kappa_vals[kappa_ind]
        
        

        phys = {
                    'tau_p': tau_p,
                    'tau_n': tau_n,
                    'g0': g0,
                    'N0': N0,
                    'N_bar': N0 + 1/(g0*tau_p),
                    's': s,
                    'beta': beta,
                    'kappa_c_mat': kappa_c*np.ones((N_lasers, N_lasers)) - kappa_c * np.eye(N_lasers),
                    'phi_p_mat': np.ones(shape=(N_lasers,N_lasers))*phi_p_arr[desired_phi_p]*np.pi,
                    'I': I,
                    'q': q,
                    'alpha': alpha,
                    'delta': np.sort(np.concatenate([delta*np.linspace(0,1,N_lasers)])),  # detuning for each laser
                    'coupling': coupling,
                    'self_feedback': self_feedback,
                    'noise_amplitude': noise_amplitude,
                    'dt': dt,
                    'Tmax': Tmax,
                    'tau': tau,
                    'N_lasers': N_lasers
                }



        def compute_potential(params, Omg):

            tau = params['tau']
            alpha = params['alpha']
            delta = params['delta']
            kappa_c_mat = params['kappa_c_mat']
            phi_p_mat = params['phi_p_mat']

            theta = np.arctan(alpha)

            # Omg is (1000,)
            Omg = Omg*(2 * np.pi * 1e9)   # (1000,1)
            Omega_exp = Omg[:, None, None]   # (1000,1,1)

            # First term
            V1 = 1/(2*tau) * (delta - Omg[:,None])**2   # (1000,2)

            # Coupling term
            cos_term = np.cos(Omega_exp*tau + theta + phi_p_mat)  # (1000,2,2)

            V2 = -np.sqrt(1+alpha**2)/(tau**2) * np.sum(
                    kappa_c_mat * cos_term,
                    axis=2
                )   # (1000,2)

            V = V1 + V2

            return V


        omg = np.linspace(-2.5, 8, 1000)
        V = compute_potential(phys, omg)
        valid = (
                np.isfinite(omega[kappa_ind,:]) &
                np.isfinite(stable[kappa_ind,:]) &
                np.isin(phi_p[kappa_ind,:], phi_p_arr[desired_phi_p])
            )

    
    
        # Create color array based on stability
        stable_mask = valid & (stable[kappa_ind, :] == 1)
        unstable_mask = valid & (stable[kappa_ind, :] == 0)
        
        # # Plot stable points (blue)
        # if np.any(stable_mask):
        #     plt.scatter(omega[kappa_ind, stable_mask], compute_potential(phys, omega[kappa_ind, stable_mask])[:, 0], c='blue', s=30, label='Stable', zorder=2)
        #     plt.scatter(omega[kappa_ind, stable_mask], compute_potential(phys, omega[kappa_ind, stable_mask])[:, 1], c='blue', s=30, zorder=2)
        
        # # Plot unstable points (red)
        # if np.any(unstable_mask):
        #     plt.scatter(omega[kappa_ind, unstable_mask], compute_potential(phys, omega[kappa_ind, unstable_mask])[:, 0], c='red', s=10, label='Unstable', zorder=1)
        #     plt.scatter(omega[kappa_ind, unstable_mask], compute_potential(phys, omega[kappa_ind, unstable_mask])[:, 1], c='red', s=10, zorder=1)

    
        plt.plot(omg, V, linewidth=2, zorder=0, label='Potential V($\Omega$)')
        plt.xlabel('Omega (GHz)', fontsize=12)
        plt.ylabel('Potential V', fontsize=12)
        plt.title('Potential Function V(Omega)', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.ylim(-0.2e30,1.6e30)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f'./potential_function_plots/potential_{desired_phi_p}_no_eqs.png', bbox_inches='tight', transparent=False)
        plt.show()
        plt.close(fig)
        clear_output(wait=True)
        break

#%%

