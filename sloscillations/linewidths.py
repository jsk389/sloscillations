# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import radial_mode_utils as radial_modes
import numpy as np 
import scaling_relations as scalings
import utils

from amplitudes import Amplitudes
from scipy.interpolate import interp1d

class Linewidths(Amplitudes):
    """
    Class to calculate the amplitudes (and heights) of oscillation
    modes
    """

    # Add this so that can inherit all of the set parameters and
    # calculated frequencies from Frequencies class
    # https://stackoverflow.com/questions/1081253/inheriting-from-instance-in-python
    def __new__(cls, parentInst):
        parentInst.__class__ = Linewidths
        return parentInst

    def __init__(self, parentInst):
        
        # Inherit frequencies class so have all frequencies stored
        pass

    def generate_radial_modes(self):
        """
        Generate radial mode amplitudes
        """
        self.l0_linewidths = utils.compute_linewidths(self.l0_freqs, self.numax)

    def generate_quadrupole_modes(self):
        """
        Generate quadrupole mode amplitudes
        """
        self.l2_linewidths = utils.compute_linewidths(self.l2_freqs, self.numax)

    def generate_nominal_dipole_modes(self):
        """
        Generate nominal l=1 mode amplitudes
        """
        self.l1_nom_linewidths = utils.compute_linewidths(self.l1_nom_freqs, self.numax)
    

    
if __name__=="__main__":

    frequency = np.arange(0.00787, 283., 0.00787)

    # Set up frequencies class
    frequencies = Frequencies(frequency=frequency,
                              numax=103.2, 
                              delta_nu=9.57, 
                              radial_order_range=[-5, 5])

    # l=0 modes
    frequencies.generate_radial_modes()
    # l=2 modes
    frequencies.generate_quadrupole_modes()
    # l=1 nominal p-modes
    frequencies.generate_nominal_dipole_modes()

    # Set up class
    amplitudes = Amplitudes(frequencies)

    # l=0 amplitudes
    amplitudes.generate_radial_modes()
    # l=2 amplitudes
    amplitudes.generate_quadrupole_modes()
    # l=1 nominal p-mode amplitudes
    amplitudes.generate_nominal_dipole_modes()


    plt.plot(amplitudes.l0_freqs, amplitudes.l0_amps, 
             color='r', marker='D', linestyle='None', label='$\ell=0$')
    plt.plot(amplitudes.l2_freqs, amplitudes.l2_amps, 
             color='g', marker='s', linestyle='None', label='$\ell=2$')
    plt.plot(amplitudes.l1_nom_freqs, amplitudes.l1_nom_amps, 
             color='b', marker='o', linestyle='None', label='Nominal $\ell=1$')
    plt.plot(amplitudes.frequency, amplitudes.a0(frequency), '--')
    plt.xlim(amplitudes.l1_nom_freqs.min(), amplitudes.l1_nom_freqs.max())
    plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
    plt.ylabel(r'Amplitude (ppm)', fontsize=18)
    plt.legend(loc='best')
    plt.show()
    