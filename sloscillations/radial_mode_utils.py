# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np 
import scaling_relations as scalings

from scipy.interpolate import interp1d

def calculate_a0(l0_freqs, numax, amax, denv):
    """
    Create an interpolation function to obtain radial mode amplitude
    at any given frequency
    """
    # Don't forget that nu_width scaling relation gives the FWHM
    # NOT width of Gaussian!
    width = denv / (2.0*np.sqrt(2.0*np.log(2)))
    amplitudes = amax * np.sqrt(np.exp(-(l0_freqs - numax)**2.0 / \
                                (2.0 * width ** 2.0)))
    return interp1d(l0_freqs, amplitudes)                   
