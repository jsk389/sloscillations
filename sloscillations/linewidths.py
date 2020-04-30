# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np 

from . import utils
from . import amplitudes

from scipy.interpolate import interp1d

class Linewidths(amplitudes.Amplitudes):
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
        Generate radial mode linewidths
        """
        self.l0_linewidths = utils.compute_linewidths(self.l0_freqs, self.numax)

    def generate_quadrupole_modes(self):
        """
        Generate quadrupole mode linewidths
        """
        self.l2_linewidths = utils.compute_linewidths(self.l2_freqs, self.numax)

    def generate_nominal_dipole_modes(self):
        """
        Generate nominal l=1 mode linewidths
        """
        self.l1_nom_linewidths = utils.compute_linewidths(self.l1_nom_freqs, self.numax)

    def generate_mixed_dipole_modes(self):
        """
        Generate mixed l=1 mode linewidths
        """
        if not hasattr(self, 'l1_nom_linewidths'):
            self.l1_nom_linewidths = utils.compute_linewidths(self.l1_nom_freqs, self.numax)

        # m=0 components
        self.l1_mixed_linewidths = []
        radial_order = np.unique(self.l1_np)
        for i in range(len(radial_order)):
            cond = (self.l1_np == radial_order[i])
            self.l1_mixed_linewidths = np.append(self.l1_mixed_linewidths,
                                        self.l1_nom_linewidths[i] * (1 - self.l1_zeta[cond]))

        if self.calc_rot:
            # Also generate linewidths for rotationally split components if they exist
            if hasattr(self, 'l1_mixed_freqs_p1') and (self.method=='simple'):
                self.l1_mixed_linewidths_p1 = []
                radial_order = np.unique(self.l1_np)
                for i in range(len(radial_order)):
                    cond = (self.l1_np == radial_order[i])
                    self.l1_mixed_linewidths_p1 = np.append(self.l1_mixed_linewidths_p1,
                                                self.l1_nom_linewidths[i] * (1 - self.l1_zeta[cond]))
            elif hasattr(self, 'l1_mixed_freqs_p1') and (self.method=='Mosser'):
                sys.exit()
            if hasattr(self, 'l1_mixed_freqs_n1') and (self.method=='simple'):
                self.l1_mixed_linewidths_n1 = []
                radial_order = np.unique(self.l1_np)
                for i in range(len(radial_order)):
                    cond = (self.l1_np == radial_order[i])
                    self.l1_mixed_linewidths_n1 = np.append(self.l1_mixed_linewidths_n1,
                                                self.l1_nom_linewidths[i] * (1 - self.l1_zeta[cond]))
            else:
                sys.exit()

    def __call__(self, entries):
        """
        Run computation
        """
        # Update class attributes with new parameters
        self.__dict__.update(entries)

        # l=0 modes
        if self.calc_l0:
            self.generate_radial_modes()
        # l=2 modes
        if self.calc_l2:
            self.generate_quadrupole_modes()
        # l=1 nominal p-modes
        if self.calc_nom_l1:
            self.generate_nominal_dipole_modes()  
        if self.calc_mixed:
            self.generate_mixed_dipole_modes()
    
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
    