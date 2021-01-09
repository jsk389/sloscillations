# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np 

from sloscillations import frequencies, amplitudes

def test_amplitudes():

    frequency = np.arange(0.00787, 283., 0.00787)

    # Set up frequencies class
    freqs = frequencies.Frequencies(frequency=frequency,
                                          numax=103.2, 
                                          delta_nu=9.57, 
                                          radial_order_range=[-5, 5],
                                          DPi1 = 77.9,
                                          coupling = 0.2,
                                          eps_g = 0.0,
                                          split_core = 0.15,
                                          calc_rot = False)

    # Set up frequencies class
    freqs()

    # Set up class
    amps = amplitudes.Amplitudes(freqs)
    print(f"Henv: {freqs.Henv}")
    print(f"denv: {freqs.denv}")
    amps()
    # l=0 amplitudes
    #amps.generate_radial_modes()
    # l=2 amplitudes
    #amps.generate_quadrupole_modes()
    # l=1 nominal p-mode amplitudes
    #amps.generate_nominal_dipole_modes()


    plt.plot(amps.l0_freqs, amps.l0_amps, 
             color='r', marker='D', linestyle='None', label='$\ell=0$')
    plt.plot(amps.l2_freqs, amps.l2_amps, 
             color='g', marker='s', linestyle='None', label='$\ell=2$')
    plt.plot(amps.l3_freqs, amps.l3_amps, 
             color='y', marker='*', linestyle='None', label='$\ell=3$')
    plt.plot(amps.l1_nom_freqs, amps.l1_nom_amps, 
             color='b', marker='o', linestyle='None', label='Nominal $\ell=1$')
    plt.plot(amps.l1_mixed_freqs, amps.l1_mixed_amps, 
             color='c', marker='^', linestyle='None', label='Mixed $\ell=1$')
    plt.plot(amps.frequency, amps.a0(frequency), '--')
    plt.xlim(amps.l1_nom_freqs.min(), amps.l1_nom_freqs.max())
    plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
    plt.ylabel(r'Amplitude (ppm)', fontsize=18)
    plt.legend(loc='best')
    plt.show()
    
if __name__=="__main__":

    test_amplitudes()