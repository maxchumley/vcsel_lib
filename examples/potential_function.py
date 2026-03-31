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

#np.hstack([VCSEL.cosine_ramp(time_arr[:int(steps/2)], 20*tau, 50*tau, kappa_initial=0, kappa_final=1), VCSEL.cosine_ramp(time_arr[int(steps/2):], 150*tau, 20*tau, kappa_initial=1, kappa_final=1)]) 

##
# VCSEL.cosine_ramp(time_arr, 20*tau, 10*tau, kappa_initial=0, kappa_final=1)

# Kappa ramp


# kappa_arr = VCSEL.cosine_ramp(time_arr, 5*tau, 100*tau, kappa_initial=kappa_c, kappa_final=kappa_c)

N_lasers = 2
coupling_scheme = 'ATA'
ramp_start = 0
ramp_shape = 0.00000001
dx = 1.0
n_iterations = 1

kappa_c = 20e9
phi_p = 0.0

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
            'kappa_c_mat': kappa_arr[-1,:,:] - kappa_c * np.eye(N_lasers),
            'phi_p_mat': np.ones(shape=(N_lasers,N_lasers))*phi_p*np.pi,
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


vcsel = VCSEL(phys)
nd = vcsel.scale_params()

#%%
def compute_potential(params, Omg):
    
    # Unpack parameters
    tau_p = params['tau_p']
    tau_n = params['tau_n']
    g0 = params['g0']
    N0 = params['N0']
    s = params['s']
    beta = params['beta']
    kappa_c_mat = params['kappa_c_mat']
    phi_p_mat = params['phi_p_mat']
    I = params['I']
    q = params['q']
    alpha = params['alpha']
    delta = params['delta']
    coupling = params['coupling']
    self_feedback = params['self_feedback']
    noise_amplitude = params['noise_amplitude']
    dt = params['dt']
    Tmax = params['Tmax']
    tau = params['tau']
    N_lasers = params['N_lasers']

    # Compute potential function (this is a placeholder, replace with actual computation)
    Omg = Omg[:,None]*(np.ones_like(delta).reshape(1,-1))*(2 * np.pi * 1e9)

    V1 = 1/(2*tau) * (delta - Omg)**2
    V2 = -np.sqrt(1+alpha**2)/(N_lasers*tau**2) * np.matmul(kappa_c_mat, np.cos(Omg*tau+np.atan(alpha)).T).T
    print(V2.shape)

    V = V1 + V2


    return V


omg = np.linspace(-2.2, 8, 1000)
V = compute_potential(phys, omg)

plt.figure(figsize=(8, 6), dpi=200)
plt.plot(omg, V, linewidth=2)
# plt.scatter(test, compute_potential(phys, test)[:,0], color='red', label='Branches')
# plt.scatter(test, compute_potential(phys, test)[:,1], color='red')
plt.xlabel('Omega (GHz)', fontsize=12)
plt.ylabel('Potential V', fontsize=12)
plt.title('Potential Function V(Omega)', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3, linestyle='--')
# plt.legend()
plt.tight_layout()
plt.show()