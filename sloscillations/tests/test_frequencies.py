# -*- coding: utf-8 -*-

from sloscillations import frequencies
import numpy as np 

import matplotlib.pyplot as plt

def test_frequencies():
    frequency = np.arange(0.00787, 283., 0.00787)

    # Set up frequencies class
    freqs = frequencies.Frequencies(frequency=frequency,
                                    numax=103.2, 
                                    delta_nu=9.57, 
                                    radial_order_range=[-5, 5])

    # Eventually want this to read in from a configuration file
    params = {'calc_l0': True,
              'calc_l2': True,
              'calc_l3': True,
              'calc_nom_l1': True,
              'calc_mixed': True,
              'calc_rot': True,
              'calc_method': 'Mosser2018_update',
              'DPi1': 77.9,
              'coupling': 0.2,
              'eps_g': 0.0,
              'split_core': 0.5,
              'split_env': 0.0,
              'l': 1,
              'method': 'simple'}
    freqs(params)
    
if __name__=="__main__":
    

    frequency = np.arange(0.00787, 283., 0.00787)

    # Set up frequencies class
    
    freqs = frequencies.Frequencies(frequency=frequency,
                                    numax=85.0, 
                                    delta_nu=6.8, 
                                    radial_order_range=[-5, 6],
                                    DPi1 = 296.8,
                                    coupling = 0.3,
                                    eps_g = 0.0,
                                    split_core = 0.345,
                                    split_env = 0.0)#,
#                                    calc_l0 = True,
#                                    calc_l2 = True,
#                                    calc_l3 = True,
#                                    calc_nom_l1 = True,
#                                    calc_mixed = True,
#                                    calc_rot = True,
#                                    calc_method = 'Mosser2018update',
#                                    l = 1,
#                                    method = 'simple')

    # Eventually want this to read in from a configuration file
    freqs()
    #print(freqs.full_freqs.loc[freqs.full_freqs['m'] == 0, ].tail(75))
    print(freqs.l1_nom_freqs)
    # Plot and echelle to check everything makes sense
    #freqs.generate_tau_values(params['DPi1'], params['coupling'], params['eps_g'])
    freqs.plot_echelle(mixed=freqs.calc_mixed, rotation=freqs.calc_rot)
    print(freqs.mode_data.head(50))
    plt.show()
    print(freqs.l1_mixed_tau_p1)
    for i in range(0, 2):
    # Eventually want this to read in from a configuration file
        params = {'calc_l0': True,
                'calc_l2': True,
                'calc_l3': True,
                'calc_nom_l1': True,
                'calc_mixed': True,
                'calc_rot': True,
                'calc_method': 'Mosser2018_update', 
                'DPi1': 296.8,
                'coupling': 0.3,
                'eps_g': 0.0,
                'split_core': 0.345,
                'split_env': 0.0,
                'l': 1,
                'method': 'simple'}
        freqs(params)
        #freqs.generate_tau_values(params['DPi1'], params['coupling'], params['eps_g'])
        #print(freqs.l1_mixed_tau_p1)
        # Plot and echelle to check everything makes sense
        #freqs.plot_echelle(mixed=True, rotation=True)
        #plt.show()
