#%%
# Time series plotting for a VCSEL model

import numpy as np
from vcsel_lib import VCSEL
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import sys
from IPython.display import clear_output
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('font', family='serif')




import matplotlib
# matplotlib.use("Agg")  # disable GUI backend
%matplotlib inline
import matplotlib.pyplot as plt

critical_kappa_values = []
kappa_tot_vals = []
power_fracs = []

starting_ind = 0

detuning_vals = np.linspace(0.0,10,101)
max_detuning = 10.684
# max_detuning = 12.061
detuning_vals[-1] = max_detuning

for detuning in detuning_vals:


    all_eqs = []
    all_order_params = []
    sorted_indices_arr = []
    max_num_eqs = 0
    kappa_vals = np.linspace(0e9, 40e9, 1000)



    eqs = None
    guesses = []

    

    for kappa_ind, kappa_c in enumerate(kappa_vals[starting_ind:]):


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
        self_feedback = 0.0
        coupling = 1.0
        noise_amplitude = 0.0
        # detuning = 0.5
        delta = detuning * 2 * np.pi * 1e9

        # Time discretization
        dt = 0.5 * tau_p#.1e-11
        Tmax = 2e-7
        steps = int(Tmax / dt)
        time_arr = np.linspace(0, Tmax, steps)
        delay_steps = int(tau / dt)



        N_lasers = 2
        coupling_scheme = 'CUSTOM'
        ramp_start = 0
        ramp_shape = 0.00000001
        dx = 1.0

        aMat = np.array([[0, 1],
                         [1, 0]])
        
        kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=kappa_c, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx, aMAT=aMat)

        phys = {
            'tau_p': tau_p,
            'tau_n': tau_n,
            'g0': g0,
            'N0': N0,
            'N_bar': N0 + 1/(g0*tau_p),
            's': s,
            'beta': beta,
            'kappa_c_mat': kappa_arr[-1,:,:],
            'phi_p_mat': np.ones(shape=(N_lasers,N_lasers))*0.0,
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
    

        kappa_max = 20e9

        vcsel = VCSEL(phys)
        nd = vcsel.scale_params()
        n_cases = len(nd['phi_p'])


        eq, results = vcsel.solve_equilibria(nd)

        results = np.array(results)


        if len(results) > 0:

            # clear_output(wait=True)

            S_1 = eq[1]
            S_2 = eq[3]

            phi_tilde = eq[-2]
            omega = eq[-1]

            E_1 = np.sqrt(S_1) * np.exp(1j * (omega * tau + nd['phi_p'][0,1]))
            E_2 = np.sqrt(S_2) * np.exp(1j * (omega * tau + phi_tilde + nd['phi_p'][1,0]))

            order_param = (np.abs(E_1+E_2))**2/(2*(np.abs(E_1)**2+np.abs(E_2)**2))
            print('order param check: ', order_param, 'kappa: ', kappa_c*1e-9)

            if order_param > 0.0:
                critical_kappa_values.append(kappa_c)

                print(f'Detuning: {detuning:.3f} GHz, Critical Kappa: {kappa_c/1e9:.3f} ns^-1')

                starting_ind = kappa_ind + starting_ind

                print("\n")

                # if detuning > 0.0:
                #     raise SystemExit("Stopping code because condition met")
                break
        


#%%



normalized_critical_kappa = np.array(critical_kappa_values)/np.array(critical_kappa_values)[-1]




plt.figure(figsize=(9, 6), dpi=200)
ax1 = plt.gca()
ax1.scatter(detuning_vals[:len(critical_kappa_values)], normalized_critical_kappa, color='b', s=50, alpha=0.7, label='Critical Coupling')
ax1.set_xlabel('Detuning (GHz)', fontsize=16)
ax1.set_xlim(-0.1,6)
ax1.set_ylim(-0.01,0.6)
ax1.set_ylabel('Critical Coupling (ns$^{-1}$)', fontsize=16)
ax1.tick_params(labelsize=14)

plt.title('Critical Coupling vs Detuning', fontsize=18)
plt.grid(alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc='upper left')

plt.tight_layout()
plt.show()



#%%
critical_kappa_data = np.vstack([detuning_vals, normalized_critical_kappa])

np.save('critical_kappa_data_alpha3.npy', critical_kappa_data)