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
              'calc_nom_l1': True,
              'calc_mixed': True,
              'calc_rot': True,
              'calc_method': 'Hekker2018',
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
                                    numax=94.50, 
                                    delta_nu=8.79, 
                                    radial_order_range=[-5, 5])

    # Eventually want this to read in from a configuration file
    params = {'calc_l0': True,
            'calc_l2': True,
            'calc_nom_l1': True,
            'calc_mixed': True,
            'calc_rot': False,
            'calc_method': 'Mosser2018_update', 
            'DPi1': 75.10,
            'coupling': 0.145,
            'eps_g': 0.0,
            'split_core': 0.345,
            'split_env': 0.0,
            'l': 1,
            'method': 'simple'}
    freqs(params)
    print(freqs.full_freqs.loc[freqs.full_freqs['m'] == 0, ].tail(75))

    # Plot and echelle to check everything makes sense
    freqs.plot_echelle(l0=False, l2=False, mixed=params['calc_mixed'], rotation=params['calc_rot'])
    plt.show()
