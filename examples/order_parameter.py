#%%
# Example usage
import numpy as np
import multiprocessing as mp
from vcsel_lib import VCSEL
from tqdm import tqdm
from sympy import symbols, Eq, solve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
# matplotlib.use('Agg')

from IPython.display import clear_output
import gc





rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def simulate_parallel(task):
    kappa_c_arr, phys, kappa_c_index = task
    # phys['kappa_c_mat'] = kappa_c_arr[:,kappa_c_index,:,:]
    phys['kappa_c_mat'] = kappa_c_arr

    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    history, _, _ = vcsel.generate_history(nd, shape='FR', n_cases=len(nd['phi_p']))
    t, y, _ = vcsel.integrate(history, nd=nd, progress=False)
    
    order_param_row = vcsel.order_parameter(y[:,:,-int(len(t)/2):])
    return kappa_c_index, order_param_row


#%%


# --- Multiprocessing execution ---
if __name__ == "__main__":
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



    self_feedback = 0.0
    coupling = 1.0 
    noise_amplitude = 0.0
 
    detuning = 0.5 # detuning (GHz)
    delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s
    phi_p = 0#np.pi

    
    N_lasers = 2
    

    Tmax = 5e-7
    dt = 0.5*tau_p


    # for ind, Tmax in enumerate(np.linspace(5e-9, 1e-6, 10)):

    steps = int(Tmax / dt)
    time_arr = np.linspace(0, Tmax, steps)
    delay_steps = int(tau / dt)
    segment_len = int(steps/2)
    segment_start = int(steps/2)


    resolution = 30

    # for ramp_start in np.linspace(2, 10, 100):
    ramp_start = 2
    ramp_shape = 100


    # Array of kappa values to simulate
    kappa_c = np.linspace(0e9,20e9,resolution)



    dx = 0.7

    # Generate ramp profile from 0 -> 1 over time_arr
    kappa_c_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0.0, kappa_final=1.0, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme='ATA', dx=dx, plot=False)



    avg_order_param = np.zeros(shape=(resolution, resolution))
    n_iterations = 100

    phi_p_vals = np.linspace(0,2*np.pi,resolution)

    # for i in range(n_iterations):

    i=0

    phys = {
        'tau_p': tau_p,
        'tau_n': tau_n,
        'g0': g0,
        'N0': N0,
        'N_bar': N0 + 1/(g0*tau_p),
        's': s,
        'beta': beta,
        'kappa_c_mat': None,  
        'phi_p_mat': np.ones(shape=(resolution,N_lasers,N_lasers))*phi_p_vals[:,None,None],    
        'I': I,
        'q': q,
        'alpha': alpha,
        'delta': np.sort(np.concatenate([[0], [delta]])),#np.sort(np.concatenate([[0], np.random.uniform(-10*delta, 10*delta, size=N_lasers-1)])),#np.concatenate([[0], [-2*delta, -1*delta, 1*delta, 2*delta]]),
        'coupling': coupling,     
        'self_feedback': self_feedback, 
        'noise_amplitude': noise_amplitude,
        'dt': dt,
        'Tmax': Tmax,
        'tau': tau,  # delay (s)
        'steps': steps,
        'delay_steps': delay_steps,
        'N_lasers': N_lasers

    }



    tasks = [(kappa_c_arr*kappa_c[row], phys, row) for row in range(resolution)]
    order_param = np.zeros(shape=(resolution, resolution))

    #mp.cpu_count()-2


    with mp.Pool(mp.cpu_count()-3) as pool:
        # imap gives results as they come in
        for kappa_idx, order_param_row in tqdm(
            pool.imap_unordered(simulate_parallel, tasks),
            total=len(tasks),
            desc=f"{np.round(phys['delta']/(2 * np.pi * 1e9), 2)} GHz Detuning",
            ncols=80
            ):
            order_param[kappa_idx, :] = order_param_row



    avg_order_param += order_param

    clear_output(wait=True)
    fig, ax = plt.subplots(
        1, 1,
        figsize=(8, 6),
        dpi=300,
        sharex=True,
        sharey=True
    )

    extent = [
        phi_p_vals[0]/np.pi,
        phi_p_vals[-1]/np.pi,
        kappa_c[0]*1e-9,
        kappa_c[-1]*1e-9
    ]

    # --- Left: current ---
    im0 = ax.imshow(
        order_param,                    # <-- CURRENT
        aspect='auto',
        extent=extent,
        origin='lower',
        cmap='jet',
        vmin=0, vmax=1
    )


    ax.set_xlabel(r"Coupling Phase $\phi_p/\pi$", fontsize=24, labelpad=14)
    ax.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=24, labelpad=14)

        # --- Global title (detuning) ---
    delta_GHz = phys['delta'] / (2*np.pi*1e9)
    formatted = []

    for x in delta_GHz:
        if x >= 0:
            # insert a phantom minus to take same width as negative
            formatted.append(r"\phantom{-}" + f"{x:.2f}")
        else:
            formatted.append(f"{x:.2f}")

    title_str = r"$\vec{\delta} = [" + ", ".join(formatted) + r"]$ GHz"
    ax.set_title(title_str, fontsize=26, pad=10)


    # --- Shared colorbar ---
    cbar = fig.colorbar(im0, ax=ax, pad=0.02)
    cbar.set_label("Order Parameter", fontsize=24, labelpad=16)
    cbar.ax.tick_params(labelsize=22)

    # --- Ticks & limits ---

    ax.set_xticks(np.linspace(extent[0], extent[1], 5))
    ax.set_yticks(np.linspace(extent[2], extent[3], 6))
    ax.tick_params(axis='both', labelsize=22)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3]) 



    plt.tight_layout()
    plt.show()
    plt.close(fig)









    











