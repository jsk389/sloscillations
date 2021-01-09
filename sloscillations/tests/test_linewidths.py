# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np 

from sloscillations import frequencies, amplitudes, linewidths

def test_linewidths():

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


    # Linewidths
    lwd = linewidths.Linewidths(amps)

    lwd()

    print(lwd.mode_data)

    plt.plot(lwd.l0_freqs, lwd.l0_linewidths, 
             color='r', marker='D', linestyle='None', label='$\ell=0$')
    plt.plot(lwd.l2_freqs, lwd.l2_linewidths, 
             color='g', marker='s', linestyle='None', label='$\ell=2$')
    plt.plot(lwd.l3_freqs, lwd.l3_linewidths, 
             color='y', marker='*', linestyle='None', label='$\ell=3$')
    plt.plot(lwd.l1_nom_freqs, lwd.l1_nom_linewidths, 
             color='b', marker='o', linestyle='None', label='Nominal $\ell=1$')
    plt.plot(lwd.l1_mixed_freqs, lwd.l1_mixed_linewidths, 
             color='c', marker='^', linestyle='None', label='Mixed $\ell=1$')
    #plt.plot(amps.frequency, lwd.a0(frequency), '--')
    plt.xlim(lwd.l1_nom_freqs.min(), lwd.l1_nom_freqs.max())
    plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
    plt.ylabel(r'Linewidth ($\mu$Hz)', fontsize=18)
    plt.yscale('log')
    plt.legend(loc='best')
    plt.show()
    
if __name__=="__main__":

    test_linewidths()