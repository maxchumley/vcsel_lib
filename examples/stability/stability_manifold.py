#%%
# Time series plotting for a VCSEL model

import numpy as np
from IPython.display import clear_output
from vcsel_lib import VCSEL
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from pathlib import Path
from tqdm import tqdm
import time
from joblib import Parallel, delayed

try:
    from examples._paths import STABILITY_DATA_DIR, STABILITY_RESULTS_DIR
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples._paths import STABILITY_DATA_DIR, STABILITY_RESULTS_DIR

DATA_DIR = STABILITY_DATA_DIR
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
delta_span = detuning * 2 * np.pi * 1e9
# print(delta/(2*np.pi*1e9))

all_data = []
alpha_arr = np.array([2], dtype=int)
a_arr = np.array([0.0])#np.linspace(0.0, 1.0, 20)  # GHz common detuning offset
phi_p_arr = np.array([0.0])
sweep_params = [
    (alpha_loop, a_loop, phi_p_loop)
    for alpha_loop in alpha_arr
    for a_loop in a_arr
    for phi_p_loop in phi_p_arr
]


for alpha_loop, a_loop, phi_p_loop in sweep_params:
# phi_p_loop = 1.0


    all_eqs = []
    all_order_params = []
    sorted_indices_arr = []
    max_num_eqs = 0
    kappa_vals = np.linspace(0.0e9, 40e9, 50)

    

    folder_name = STABILITY_RESULTS_DIR / "all_eq_2laser_symmetric_detuning_noise"
    # os.makedirs(f"{folder_name}/detuning_{detuning:.2f}_0self_phi_p{phi_p_loop:.2f}pi_ramp_noise_injection", exist_ok=True)
    eqs = None
    stable_arr = []
    avg_time = 0
    for kappa_ind, kappa_c in enumerate(tqdm(kappa_vals)):

        
        # Physical parameters
        alpha = int(alpha_loop)
        a = float(a_loop) * 2 * np.pi * 1e9
        delta = np.sort((delta_span/2*np.linspace(-1,1,N_lasers)) + a)
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

        
        coupling_scheme = 'CUSTOM'
        ramp_start = 0
        ramp_shape = 0.00000001
        dx = 1.0

        aMAT = np.ones(shape=(N_lasers,N_lasers)) - np.eye(N_lasers)
        kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=kappa_c, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx, aMAT=aMAT)

        self_feedback = 0.00

        # np.ones(shape=(N_lasers,N_lasers))*phi_p_loop*np.pi

        phys = {
            'tau_p': tau_p,
            'tau_n': tau_n,
            'g0': g0,
            'N0': N0,
            'N_bar': N0 + 1/(g0*tau_p),
            's': s,
            'beta': beta,
            'kappa_c_mat': kappa_arr[-1,:,:],
            'phi_p_mat': np.array([[0,0],[np.pi,0]]),
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
    ax.set_title(
        rf'$\delta = {detuning:.2f}$ GHz    $a = {a_loop:.2f}$ GHz    $\phi_p = {phi_p_loop:.2f}\pi$',
        fontsize=22,
        pad=20,
    )
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
    alpha_loop, a_loop, phi_p_loop = sweep_params[p]
    item_with_meta = np.pad(item, ((0, 0), (0, 0), (0, 3)), mode='constant', constant_values=np.nan)
    item_with_meta[:, :, -3] = alpha_loop
    item_with_meta[:, :, -2] = a_loop
    item_with_meta[:, :, -1] = phi_p_loop
    all_data[p] = item_with_meta


all_data_stacked = np.concatenate(all_data, axis=1)

# Save as dictionary with requested keys
data_dict = {
    'all_data': all_data_stacked,
    'kappa_vals': kappa_vals,
    'detuning_GHz': np.asarray(delta) / (2 * np.pi * 1e9),
    'alpha_arr': alpha_arr,
    'a_arr': a_arr,
    'phi_p_arr': phi_p_arr,
    'sweep_params': np.asarray(sweep_params, dtype=float),
    'metadata_columns': np.array(['alpha', 'a', 'phi_p'], dtype=object),
    'params': phys,
    'metadata': {
        'N_lasers': N_lasers,
        'detuning_GHz_input': detuning,
        'alpha_arr': alpha_arr,
        'a_arr': a_arr,
        'tau_p': phys['tau_p'],
        'tau': phys['tau'],
        'kappa_budget_convention': 'sum_ij kappa_ij = kappa_c',
    },
}

DATA_DIR.mkdir(parents=True, exist_ok=True)
stability_data_filename = DATA_DIR / f"all_equilibria_data_{N_lasers}laser_{detuning:.2f}Ghz_a0-1_phi0.npy"
np.save(stability_data_filename, data_dict)

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
stability_data_filename = DATA_DIR / f"all_equilibria_data_{N_lasers}laser_{detuning:.2f}Ghz_a0-1_phi0.npy"
data_dict = np.load(stability_data_filename, allow_pickle=True).item()
all_data_stacked = data_dict['all_data']
kappa_vals = data_dict['kappa_vals']
kappa_plot = np.array(kappa_vals) * 1e-9
params = data_dict.get('params', {})
metadata = data_dict.get('metadata', {})
N_lasers = int(metadata.get('N_lasers', params.get('N_lasers', N_lasers)))
detuning = float(metadata.get('detuning_GHz_input', detuning))
tau_p = float(metadata.get('tau_p', params.get('tau_p', tau_p if 'tau_p' in globals() else np.nan)))
tau = float(metadata.get('tau', params.get('tau', tau if 'tau' in globals() else np.nan)))
phi_p_arr = np.asarray(data_dict.get('phi_p_arr', phi_p_arr if 'phi_p_arr' in globals() else [0.0]))
alpha_arr = np.asarray(data_dict.get('alpha_arr', metadata.get('alpha_arr', alpha_arr if 'alpha_arr' in globals() else [])))
a_arr = np.asarray(data_dict.get('a_arr', metadata.get('a_arr', a_arr if 'a_arr' in globals() else [])))
detuning_arr = np.asarray(
    data_dict.get(
        'detuning_GHz',
        (
            np.asarray(params['delta']) / (2 * np.pi * 1e9)
            if 'delta' in params
            else data_dict.get('detuningGHZ', np.array([]))
        ),
    )
)

if not np.isfinite(tau_p) or not np.isfinite(tau):
    raise KeyError("Saved data is missing tau_p/tau metadata; rerun the save cell with the updated script.")

metadata_columns = list(data_dict.get('metadata_columns', []))
has_alpha_column = (
    'alpha_arr' in data_dict
    or 'alpha_arr' in metadata
    or ('sweep_params' in data_dict and np.asarray(data_dict['sweep_params']).ndim == 2)
)

if metadata_columns:
    meta_count = len(metadata_columns)
    stable = all_data_stacked[:,:,-(meta_count + 1)]
    omega = all_data_stacked[:,:,-(meta_count + 2)]/(2*np.pi*1e9*tau_p)
    meta_values = {
        name: all_data_stacked[:,:, -meta_count + col_idx]
        for col_idx, name in enumerate(metadata_columns)
    }
    alpha_saved = meta_values.get('alpha', np.full_like(stable, params.get('alpha', np.nan), dtype=float))
    a_saved = meta_values.get('a', np.full_like(stable, np.nan, dtype=float))
    phi_p = meta_values.get('phi_p', np.full_like(stable, np.nan, dtype=float))
elif has_alpha_column:
    stable = all_data_stacked[:,:,-3]
    omega = all_data_stacked[:,:,-4]/(2*np.pi*1e9*tau_p)
    alpha_saved = all_data_stacked[:,:,-2]
    a_saved = np.full_like(stable, np.nan, dtype=float)
    phi_p = all_data_stacked[:,:,-1]
else:
    stable = all_data_stacked[:,:,-2]
    omega = all_data_stacked[:,:,-3]/(2*np.pi*1e9*tau_p)
    alpha_saved = np.full_like(stable, params.get('alpha', np.nan), dtype=float)
    a_saved = np.full_like(stable, np.nan, dtype=float)
    phi_p = all_data_stacked[:,:,-1]

saved_alpha_values = np.unique(alpha_saved[np.isfinite(alpha_saved)])
if saved_alpha_values.size > 0:
    alpha_arr = saved_alpha_values
saved_a_values = np.unique(a_saved[np.isfinite(a_saved)])
if saved_a_values.size > 0:
    a_arr = saved_a_values
saved_phi_p_values = np.unique(phi_p[np.isfinite(phi_p)])
if saved_phi_p_values.size > 0:
    phi_p_arr = saved_phi_p_values


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

flat_omega = omega.flatten()
flat_omega = flat_omega[np.isfinite(flat_omega)]
freq_range = [np.floor(np.min(flat_omega)), np.ceil(np.max(flat_omega))]
omega_pad = max(1.0, 0.10 * (freq_range[1] - freq_range[0]))
freq_range = [freq_range[0] - omega_pad, freq_range[1] + omega_pad]



def representative_column_values(values):
    return np.array([
        vals[0] if (vals := values[np.isfinite(values[:, j]), j]).size else np.nan
        for j in range(values.shape[1])
    ])


def phi_index_blocks_for_alpha(phi_values, alpha_values, phi_values_unique, alpha_value):
    col_phi = representative_column_values(phi_values)
    col_alpha = representative_column_values(alpha_values)
    return [
        np.where(np.isclose(col_alpha, alpha_value) & np.isclose(col_phi, phi_value))[0]
        for phi_value in phi_values_unique
    ]


def a_index_blocks_for_alpha_phi(a_values, alpha_values, phi_values, a_values_unique, alpha_value, phi_value):
    col_a = representative_column_values(a_values)
    col_alpha = representative_column_values(alpha_values)
    col_phi = representative_column_values(phi_values)
    return [
        np.where(
            np.isclose(col_alpha, alpha_value)
            & np.isclose(col_phi, phi_value)
            & np.isclose(col_a, a_value)
        )[0]
        for a_value in a_values_unique
    ]


# Prepare data by saved a index for one alpha value and phi_p = 0.
alpha_index_for_slider = 0
alpha_value_for_slider = alpha_arr[alpha_index_for_slider] if len(alpha_arr) else np.nan
phi_value_for_slider = 0.0
if len(phi_p_arr):
    phi_value_for_slider = phi_p_arr[np.argmin(np.abs(phi_p_arr - 0.0))]
a_blocks = a_index_blocks_for_alpha_phi(
    a_saved,
    alpha_saved,
    phi_p,
    a_arr,
    alpha_value_for_slider,
    phi_value_for_slider,
)

all_frames_data = []

for desired_a_idx in range(len(a_blocks)):
    x_all = []
    y_all = []
    z_all = []
    c_all = []
    cols = a_blocks[desired_a_idx]
    
    for i, kap in enumerate(kappa_plot):
        valid = (
            np.isfinite(omega[i, cols]) &
            np.isfinite(stable[i, cols])
        )

        if not np.any(valid):
            continue
                
        x_all.append(np.full(np.sum(valid), kap))
        y_all.append(omega[i, cols][valid])
        z_all.append(E_tot[i, cols][valid]*1)
        c_all.append(stable[i, cols][valid])

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
    for k in range(len(a_arr))
]

fig.frames = frames
# np.nanmax(E_tot)
# kappa_vals[-1]*1e-9
# freq_range[0], freq_range[1]
# Add slider
fig.update_layout(
    scene=dict(
        xaxis=dict(title='κc (ns⁻¹)', range=[0, 40]),
        yaxis=dict(title='ω (GHz)', range=[freq_range[0], freq_range[1]]),
        zaxis=dict(title='E_tot', range=[np.nanmin(E_tot), 20]),
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, b=40, t=40),
    title=(
        f'3D Equilibrium Branches '
        f'(alpha = {alpha_value_for_slider:g}, phi_p = {phi_value_for_slider:.2f} pi, a index = 0)'
    ),
    updatemenus=[],
    sliders=[dict(
        active=0,
        yanchor='top',
        y=0,
        xanchor='left',
        currentvalue=dict(
            prefix='a index: ',
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
    html_dir = STABILITY_RESULTS_DIR / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_dir / f"branches_{N_lasers}laser_{detuning:.0f}Ghz_{computation_type}.html", auto_play=False)



#%%

import plotly.graph_objects as go
import numpy as np

# Subsample parameter (e.g., keep every nth point along kappa axis)
subsample = 1  # adjust this value to control density

# Plot only a subset of kappa values (in ns^-1)
kappa_plot_min = 0.0
kappa_plot_max = 40.0

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
        yaxis=dict(title='ω (GHz)', range=[freq_range[0], freq_range[1]]),
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
    html_dir = STABILITY_RESULTS_DIR / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_dir / f"manifold_{N_lasers}laser_{detuning:.2f}Ghz_{computation_type}.html", auto_play=False)


#%%
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

desired_phi_p = 0

# Build a global mask for this phi_p across all kappa (for robust y-limits)
global_mask = (
    np.isfinite(omega) &
    np.isfinite(phi_p) &
    np.isfinite(stable) &
    np.isclose(phi_p, phi_p_arr[desired_phi_p])
)

if np.any(global_mask):
    omega_valid = omega[global_mask]
    freq_range = [np.floor(np.min(omega_valid)), np.ceil(np.max(omega_valid))]
else:
    freq_range = [np.nanmin(omega), np.nanmax(omega)]
omega_pad = max(1.0, 0.10 * (freq_range[1] - freq_range[0]))
freq_range = [freq_range[0] - omega_pad, freq_range[1] + omega_pad]

# Discrete colormap for stable/unstable
colors_discrete = ['red', 'blue']  # unstable, stable
cmap_discrete = mpl.colors.ListedColormap(colors_discrete)
norm_discrete = mpl.colors.BoundaryNorm([0, 0.5, 1.0], cmap_discrete.N)

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

for i, kap in enumerate(kappa_vals):
    valid_mask = (
        np.isfinite(omega[i, :]) &
        np.isfinite(phi_p[i, :]) &
        np.isfinite(stable[i, :]) &
        np.isclose(phi_p[i, :], phi_p_arr[desired_phi_p])
    )
    if not np.any(valid_mask):
        continue

    x = np.full(np.sum(valid_mask), kap * 1e-9)
    y = omega[i, valid_mask]
    c = stable[i, valid_mask]

    # Plot unstable (red) first
    unstable = c < 0.5
    if np.any(unstable):
        ax.scatter(
            x[unstable], y[unstable],
            s=8, c=np.zeros(np.sum(unstable)),
            cmap=cmap_discrete, norm=norm_discrete,
            edgecolors='none', alpha=0.9, zorder=2
        )

    # Plot stable (blue) second and on top
    stable_mask = c >= 0.5
    if np.any(stable_mask):
        ax.scatter(
            x[stable_mask], y[stable_mask],
            s=8, c=np.ones(np.sum(stable_mask)),
            cmap=cmap_discrete, norm=norm_discrete,
            edgecolors='none', alpha=0.9, zorder=3
        )

ax.set_xlim(0, 40)
ax.set_ylim(freq_range[0], freq_range[1])
ax.set_xlabel(r'$\kappa_c$ (ns$^{-1}$)', fontsize=24)
ax.set_ylabel(r'$\omega$ (GHz)', fontsize=24)
ax.set_title(rf'Frequency branches ($\phi_p={phi_p_arr[desired_phi_p]:.2f}\pi$)', fontsize=28, pad=14)
ax.tick_params(axis='both', labelsize=20)
ax.grid(alpha=0.25)

# Independent mappable for consistent discrete colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap_discrete, norm=norm_discrete)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, boundaries=[0, 0.5, 1.0], ticks=[0.25, 0.75], pad=0.02)
cbar.ax.set_yticklabels(['Unstable', 'Stable'], fontsize=24, rotation=90)
cbar.set_label('Stability', fontsize=24)

fig.tight_layout()
plt.show()
plt.close(fig)
clear_output(wait=True)
#%%
import matplotlib.pyplot as plt

desired_phi_p = 0

# Build global mask for robust y-limits
global_mask = (
    np.isfinite(E_tot) &
    np.isfinite(phi_p) &
    np.isfinite(stable) &
    np.isclose(phi_p, phi_p_arr[desired_phi_p])
)

if np.any(global_mask):
    etot_valid = E_tot[global_mask]
    etot_range = [np.floor(np.min(etot_valid)), np.ceil(np.max(etot_valid))]
else:
    etot_range = [np.nanmin(E_tot), np.nanmax(E_tot)]

# Discrete colormap: unstable/red, stable/blue
cmap_discrete = mpl.colors.ListedColormap(['red', 'blue'])
norm_discrete = mpl.colors.BoundaryNorm([0, 0.5, 1.0], cmap_discrete.N)

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

for i, kap in enumerate(kappa_vals):
    valid_mask = (
        np.isfinite(E_tot[i, :]) &
        np.isfinite(phi_p[i, :]) &
        np.isfinite(stable[i, :]) &
        np.isclose(phi_p[i, :], phi_p_arr[desired_phi_p])
    )
    if not np.any(valid_mask):
        continue

    x = np.full(np.sum(valid_mask), kap * 1e-9)
    y = E_tot[i, valid_mask]
    c = stable[i, valid_mask]

    unstable = c < 0.5
    if np.any(unstable):
        ax.scatter(
            x[unstable], y[unstable],
            s=8, c=np.zeros(np.sum(unstable)),
            cmap=cmap_discrete, norm=norm_discrete,
            edgecolors='none', alpha=0.9, zorder=2
        )

    stable_mask = c >= 0.5
    if np.any(stable_mask):
        ax.scatter(
            x[stable_mask], y[stable_mask],
            s=8, c=np.ones(np.sum(stable_mask)),
            cmap=cmap_discrete, norm=norm_discrete,
            edgecolors='none', alpha=0.9, zorder=3
        )

ax.set_xlim(0, 40)
ax.set_ylim(etot_range[0], etot_range[1])
ax.set_xlabel(r'$\kappa_c$ (ns$^{-1}$)', fontsize=24)
ax.set_ylabel(r'$|E_{tot}|^2$', fontsize=24)
ax.set_title(rf'Intensity branches ($\phi_p={phi_p_arr[desired_phi_p]:.2f}\pi$)', fontsize=28, pad=14)
ax.tick_params(axis='both', labelsize=20)
ax.grid(alpha=0.25)

sm = mpl.cm.ScalarMappable(cmap=cmap_discrete, norm=norm_discrete)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, boundaries=[0, 0.5, 1.0], ticks=[0.25, 0.75], pad=0.02)
cbar.ax.set_yticklabels(['Unstable', 'Stable'], fontsize=24, rotation=90)
cbar.set_label('Stability', fontsize=24)

fig.tight_layout()
plt.show()
plt.close(fig)
clear_output(wait=True)



#%%

import numpy as np

x = np.sort(omega[i, valid_mask])
e = E_tot[i, valid_mask]

# Sort E_tot in the same order as omega
sort_idx = np.argsort(omega[i, valid_mask])
x_sorted = omega[i, valid_mask][sort_idx]
e_sorted = e[sort_idx]

# Filter where E_tot > 15
mask = e_sorted > 16
x_filtered = x_sorted[mask]
omega_filtered = x_sorted[mask]

# Compute first difference
dx = np.diff(x_filtered)

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
ax.plot(omega_filtered[:-1], dx, '.-', linewidth=2, markersize=8)
ax.set_ylim(0, 2)
ax.set_xlabel(r'$\omega$ (GHz)', fontsize=20)
ax.set_ylabel('First Difference', fontsize=20)
ax.set_title('Frequency Difference', fontsize=22, pad=20)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close(fig)




#%%
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

n_frames = 100


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

    ax.set_title(rf'Equilibrium Branches ($\phi_p={phi_p_arr[desired_phi_p]:.2f}\pi$)', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.view_init(elev=20, azim=250)

    ax.set_xlim(0,40)
    ax.set_ylim(freq_range[0], freq_range[1])
    ax.set_zlim(0, np.nanmax(E_tot)*1.1)

    # Set grid opacity lower
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0, 0, 0, 0.05)  # RGBA with alpha

    # colorbar in its own slot (does NOT resize the 3D axes)
    # Use an independent mappable so legend colors stay fully opaque even
    # when plotted points use low alpha.
    sm = mpl.cm.ScalarMappable(cmap=cmap_discrete, norm=norm_discrete)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, boundaries=[0, 0.5, 1.0], ticks=[0.25, 0.75])
    cbar.set_label('Stability', fontsize=16)
    cbar.ax.set_yticklabels(['Unstable', 'Stable'], rotation=90, fontsize=14)

    # fig.savefig(f'../3d_branch_anim/{frame_num}.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.show()
    plt.close(fig)
    clear_output(wait=True)
    frame_num += 1




#%%
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from IPython.display import clear_output

make_3d_animation = False
save_3d_animation_frames = False
save_static_3d_frames = True
frame_stride = 10
desired_phi_p = len(phi_p_arr)
alpha_indices_to_plot = "each"  # "each" plots one figure per alpha; None plots all; examples: [0], [1, 2]
n_rot_frames = len(kappa_plot)
omega_range_pad_3d = 3.0

if alpha_indices_to_plot == "each":
    alpha_value_sets_to_plot = [np.asarray([alpha_value]) for alpha_value in alpha_arr]
elif alpha_indices_to_plot is None:
    alpha_values_to_plot = alpha_arr
    alpha_value_sets_to_plot = [np.asarray(alpha_values_to_plot)]
else:
    alpha_values_to_plot = np.asarray(alpha_arr)[alpha_indices_to_plot]
    alpha_value_sets_to_plot = [np.asarray(alpha_values_to_plot)]


def plot_3d_manifold_frame(frame_idx, alpha_values_to_plot, show=True, save_path=None):
    plane_kappa = kappa_plot[frame_idx]
    fig = plt.figure(figsize=(10, 8), dpi=300)
    gs = GridSpec(
        1, 2,
        width_ratios=[30, 1],
        wspace=0.05,
        left=0.12,
        right=0.92,
        bottom=0.12,
        top=0.9,
    )

    ax = fig.add_subplot(gs[0], projection='3d', computed_zorder=False)
    cax = fig.add_subplot(gs[1])

    colors_discrete = ['red', 'blue']
    cmap_discrete = mpl.colors.ListedColormap(colors_discrete)
    norm_discrete = mpl.colors.BoundaryNorm([0, 0.5, 1.0], cmap_discrete.N)

    valid_mask = (
        np.isfinite(omega) &
        np.isfinite(phi_p) &
        np.isfinite(alpha_saved) &
        np.isfinite(stable) &
        np.isin(phi_p, phi_p_arr[:desired_phi_p]) &
        np.isin(alpha_saved, alpha_values_to_plot)
    )

    kappa_mesh = kappa_plot[:, np.newaxis]
    before_plane = kappa_mesh < plane_kappa
    after_plane = ~before_plane

    mask_before = valid_mask & before_plane
    if np.any(mask_before):
        i_before, _ = np.where(mask_before)
        ax.scatter(
            kappa_plot[i_before],
            omega[mask_before],
            E_tot[mask_before],
            c=stable[mask_before],
            cmap=cmap_discrete,
            norm=norm_discrete,
            s=4,
            alpha=0.1,
            edgecolors='none',
            depthshade=False,
            zorder=1,
        )

    mask_after = valid_mask & after_plane
    if np.any(mask_after):
        i_after, _ = np.where(mask_after)
        ax.scatter(
            kappa_plot[i_after],
            omega[mask_after],
            E_tot[mask_after],
            c=stable[mask_after],
            cmap=cmap_discrete,
            norm=norm_discrete,
            s=8,
            alpha=1.0,
            edgecolors='none',
            depthshade=False,
            zorder=2,
        )

    ax.set_xlabel(r'$\kappa_c$ (ns$^{-1}$)', labelpad=8, fontsize=18)
    ax.set_ylabel(r'$\omega$ (GHz)', labelpad=8, fontsize=18)
    ax.set_zlabel(r'$|E_{tot}|^2$', labelpad=12, fontsize=18)
    if len(alpha_values_to_plot) == 1:
        alpha_title = rf'$\alpha={alpha_values_to_plot[0]:g}$'
    else:
        alpha_title = rf'$\alpha\in[{np.nanmin(alpha_values_to_plot):g},{np.nanmax(alpha_values_to_plot):g}]$'
    ax.set_title(
        rf'Equilibrium Branches ({alpha_title})',
        fontsize=20,
    )
    ax.tick_params(axis='both', which='major', labelsize=14)

    azim = 250.0 + 360.0 * frame_idx / max(1, n_rot_frames)
    ax.view_init(elev=20, azim=azim)

    ax.set_xlim(0, 40)
    ax.set_ylim(freq_range[0] - omega_range_pad_3d, freq_range[1] + omega_range_pad_3d)
    ax.set_zlim(0, np.nanmax(E_tot)*1.1)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0, 0, 0, 0.05)

    sm = mpl.cm.ScalarMappable(cmap=cmap_discrete, norm=norm_discrete)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, boundaries=[0, 0.5, 1.0], ticks=[0.25, 0.75])
    cbar.set_label('Stability', fontsize=16)
    cbar.ax.set_yticklabels(['Unstable', 'Stable'], rotation=90, fontsize=14)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches=None, transparent=True, facecolor='white')
    if show:
        plt.show()
    plt.close(fig)


def plot_3d_manifold_alpha_row(frame_idx, alpha_value_sets, show=True, save_path=None):
    n_panels = len(alpha_value_sets)
    plane_kappa = kappa_plot[frame_idx]
    panel_fontsize = 34
    axis_label_fontsize = 22
    tick_fontsize = 16
    colorbar_fontsize = 20
    fig = plt.figure(figsize=(4.8 * n_panels + 0.7, 4.8), dpi=300)
    gs = GridSpec(
        1,
        n_panels + 1,
        width_ratios=[1] * n_panels + [0.05],
        wspace=0.02,
        left=0.08,
        right=0.94,
        bottom=0.14,
        top=0.86,
    )

    colors_discrete = ['red', 'blue']
    cmap_discrete = mpl.colors.ListedColormap(colors_discrete)
    norm_discrete = mpl.colors.BoundaryNorm([0, 0.5, 1.0], cmap_discrete.N)
    kappa_mesh = kappa_plot[:, np.newaxis]
    before_plane = kappa_mesh < plane_kappa
    after_plane = ~before_plane

    panel_labels = "abcdefghijklmnopqrstuvwxyz"
    for panel_idx, alpha_values_to_plot in enumerate(alpha_value_sets):
        ax = fig.add_subplot(gs[panel_idx], projection='3d', computed_zorder=False)
        valid_mask = (
            np.isfinite(omega) &
            np.isfinite(phi_p) &
            np.isfinite(alpha_saved) &
            np.isfinite(stable) &
            np.isin(phi_p, phi_p_arr[:desired_phi_p]) &
            np.isin(alpha_saved, alpha_values_to_plot)
        )

        mask_before = valid_mask & before_plane
        if np.any(mask_before):
            i_before, _ = np.where(mask_before)
            ax.scatter(
                kappa_plot[i_before],
                omega[mask_before],
                E_tot[mask_before],
                c=stable[mask_before],
                cmap=cmap_discrete,
                norm=norm_discrete,
                s=.5,
                alpha=0.2 ,
                edgecolors= 'none',
                depthshade=False,
                zorder=2,
            )

        mask_after = valid_mask & after_plane
        if np.any(mask_after):
            i_after, _ = np.where(mask_after)
            ax.scatter(
                kappa_plot[i_after],
                omega[mask_after],
                E_tot[mask_after],
                c=stable[mask_after],
                cmap=cmap_discrete,
                norm=norm_discrete,
                s=5,
                alpha=0.95,
                edgecolors='k',
                linewidths=0.2,
                depthshade=False,
                zorder=1,
            )

        if len(alpha_values_to_plot) == 1:
            alpha_title = rf'$\alpha={alpha_values_to_plot[0]:g}$'
        else:
            alpha_title = rf'$\alpha\in[{np.nanmin(alpha_values_to_plot):g},{np.nanmax(alpha_values_to_plot):g}]$'
        ax.set_title(alpha_title, fontsize=panel_fontsize, pad=8)
        ax.text2D(0.04, 0.86, f"({panel_labels[panel_idx]})", transform=ax.transAxes, fontsize=panel_fontsize)
        ax.set_xlabel(r'$\kappa_c$ (ns$^{-1}$)', labelpad=10, fontsize=axis_label_fontsize)
        ax.set_ylabel(r'$\omega$ (GHz)', labelpad=10, fontsize=axis_label_fontsize)
        ax.set_zlabel(r'$|E_{tot}|^2$', labelpad=12, fontsize=axis_label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.view_init(elev=20, azim=250.0 + 360.0 * frame_idx / max(1, n_rot_frames))
        ax.set_xlim(0, 40)
        ax.set_ylim(freq_range[0] - omega_range_pad_3d, freq_range[1] + omega_range_pad_3d)
        ax.set_zlim(0, np.nanmax(E_tot) * 1.1)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = (0, 0, 0, 0.06)

    cax = fig.add_subplot(gs[-1])
    sm = mpl.cm.ScalarMappable(cmap=cmap_discrete, norm=norm_discrete)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, boundaries=[0, 0.5, 1.0], ticks=[0.25, 0.75])
    cbar.set_label('Stability', fontsize=colorbar_fontsize)
    cbar.ax.set_yticklabels(['Unstable', 'Stable'], rotation=90, fontsize=colorbar_fontsize)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white', pad_inches=0.48)
    if show:
        plt.show()
    plt.close(fig)

if make_3d_animation:
    frame_num = 0
    alpha_values_to_plot = alpha_value_sets_to_plot[0]
    for frame_idx in range(0, n_rot_frames, frame_stride):
        frames_dir = STABILITY_RESULTS_DIR / "3d_anim"
        if save_3d_animation_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)
        save_path = frames_dir / f'{frame_num}.png' if save_3d_animation_frames else None
        plot_3d_manifold_frame(
            frame_idx,
            alpha_values_to_plot,
            show=not save_3d_animation_frames,
            save_path=save_path,
        )
        clear_output(wait=True)
        frame_num += 1
else:
    save_path = STABILITY_RESULTS_DIR / 'final_alpha_row.png' if save_static_3d_frames else None
    plot_3d_manifold_alpha_row(
        n_rot_frames - 1,
        alpha_value_sets_to_plot,
        show=not save_static_3d_frames,
        save_path=save_path,
    )









#%%
from IPython.display import clear_output
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

# --- User targets ---
desired_kappa_ns_inv = 40.0
desired_phi_p = 0.0      # same units as phi_p array
phi_p_tol = 1e-6


for desired_kappa_ns_inv in kappa_vals[::10]*1e-9:


    # Determine kappa axis in ns^-1
    if 'kappa_plot' in globals():
        kappa_axis_ns = np.asarray(kappa_plot, dtype=float)
    elif 'kappa_vals' in globals():
        kappa_axis_ns = np.asarray(kappa_vals, dtype=float) * 1e-9
    else:
        raise RuntimeError("Could not find kappa axis (expected `kappa_plot` or `kappa_vals`).")

    # Select nearest kappa slice
    kappa_idx = int(np.argmin(np.abs(kappa_axis_ns - desired_kappa_ns_inv)))
    kappa_sel = float(kappa_axis_ns[kappa_idx])

    omega_row = np.asarray(omega[kappa_idx, :], dtype=float)
    E_row = np.asarray(E_tot[kappa_idx, :], dtype=float)
    stable_row = np.asarray(stable[kappa_idx, :], dtype=float)

    phi_row = None
    if 'phi_p' in globals():
        phi_arr = np.asarray(phi_p)
        if phi_arr.ndim == 2:
            phi_row = np.asarray(phi_arr[kappa_idx, :], dtype=float)
        elif phi_arr.ndim == 1:
            phi_row = np.asarray(phi_arr, dtype=float)

    valid = np.isfinite(omega_row) & np.isfinite(E_row) & np.isfinite(stable_row)
    if phi_row is not None:
        valid &= np.isfinite(phi_row)

    clear_output(wait=True)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Create discrete colormap for stable/unstable
    colors_discrete = ['red', 'blue']  # unstable, stable
    cmap_discrete = mpl.colors.ListedColormap(colors_discrete)
    norm_discrete = mpl.colors.BoundaryNorm([0, 0.5, 1.0], cmap_discrete.N)

    scatter = ax.scatter(
        omega_row[valid],
        E_row[valid],
        c=stable_row[valid],
        cmap=cmap_discrete,
        norm=norm_discrete,
        s=8,
        alpha=0.5
    )

    # Highlight desired phi_p points
    if phi_row is not None:
        phi_mask = valid & np.isclose(phi_row, desired_phi_p, atol=phi_p_tol, rtol=0.0)
        if np.any(phi_mask):
            ax.scatter(
                omega_row[phi_mask],
                E_row[phi_mask],
                c=stable_row[phi_mask],
                cmap=cmap_discrete,
                norm=norm_discrete,
                s=40,
                alpha=1.0,
                zorder=5,
                label=rf'$\phi_p = {desired_phi_p:.1f}$'
            )
            ax.legend(loc='upper right', fontsize=15)

    ax.set_xlabel(r"$\omega$ (GHz)", fontsize=20)
    ax.set_ylabel(r"$|E_{\mathrm{tot}}|^2$", fontsize=20)
    ax.set_title(rf"$\kappa_c = {kappa_sel:.2f}$ $ns^{{-1}}$", fontsize=22)

    ax.set_xlim(np.nanmin(omega)*1.1, np.nanmax(omega)*1.1)
    ax.set_ylim(0, np.nanmax(E_tot)*1.1)

    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=18)

    # Add colorbar with discrete labels
    cbar = fig.colorbar(scatter, ax=ax, boundaries=[0, 0.5, 1.0], ticks=[0.25, 0.75])
    cbar.set_label('Stability', fontsize=16)
    cbar.ax.set_yticklabels(['Unstable', 'Stable'], fontsize=14, rotation=90)

    fig.tight_layout()
    plt.show()
