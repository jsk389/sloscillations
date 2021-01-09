# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np 

from sloscillations import frequencies, amplitudes, linewidths, heights

def test_heights():

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
                                          inclination_angle=0.0,
                                          calc_rot = False)

    # Set up frequencies class
    freqs()

    # Set up class
    amps = amplitudes.Amplitudes(freqs)
    print(f"Henv: {freqs.Henv}")
    print(f"denv: {freqs.denv}")
    amps()


    # Linewidths
    lwd = linewidths.Linewidths(amps)

    lwd()

    # Heights
    h = heights.Heights(lwd)#, 1/(frequency[0]*1e-6))

    h()

    print(h.mode_data)

    plt.plot(h.l0_freqs, h.l0_heights, 
             color='r', marker='D', linestyle='None', label='$\ell=0$')
    plt.plot(h.l2_freqs, h.l2_heights, 
             color='g', marker='s', linestyle='None', label='$\ell=2$')
    plt.plot(h.l3_freqs, h.l3_heights, 
             color='y', marker='*', linestyle='None', label='$\ell=3$')
    plt.plot(h.l1_nom_freqs, h.l1_nom_heights, 
             color='b', marker='o', linestyle='None', label='Nominal $\ell=1$')
    plt.plot(h.l1_mixed_freqs, h.l1_mixed_heights, 
             color='c', marker='^', linestyle='None', label='Mixed $\ell=1$')
    #plt.plot(amps.frequency, lwd.a0(frequency), '--')
    plt.xlim(h.l1_nom_freqs.min(), h.l1_nom_freqs.max())
    plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
    plt.ylabel(r'Mode Height (ppm$^{2}\mu$Hz$^{-1}$)', fontsize=18)
    plt.yscale('log')
    plt.legend(loc='best')
    plt.show()

if __name__=="__main__":

    test_heights()