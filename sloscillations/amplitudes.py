# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np 

from . import frequencies
from . import radial_mode_utils as radial_modes
from . import scaling_relations as scalings

from scipy.interpolate import interp1d

class Amplitudes(frequencies.Frequencies):
    """
    Class to calculate the amplitudes (and heights) of oscillation
    modes
    """

    # Add this so that can inherit all of the set parameters and
    # calculated frequencies from Frequencies class
    # https://stackoverflow.com/questions/1081253/inheriting-from-instance-in-python
    def __new__(cls, parentInst):
        parentInst.__class__ = Amplitudes
        return parentInst

    def __init__(self, parentInst, evo_state='RGB', mission='Kepler'):
        
        # Inherit frequencies class so have all frequencies stored
        self.evo_state = evo_state
        self.mission = mission
        self.Henv = scalings.Henv_scaling(self.numax,
                                          evo_state=self.evo_state)
        self.denv = scalings.denv_scaling(self.numax,
                                          evo_state=self.evo_state)
        self.amax = scalings.amax_scaling(self.Henv,
                                          self.numax,
                                          self.delta_nu,
                                          mission=self.mission)

        self.a0 = radial_modes.calculate_a0(self.frequency, 
                                            self.numax, 
                                            self.amax, 
                                            self.denv)

        if self.mission == 'Kepler':
            self.vis_tot = 3.16
            self.vis1 = 1.54
            self.vis2 = 0.58
        elif self.mission == 'TESS':
            self.vis_tot = 2.94
            self.vis1 = 1.46
            self.vis2 = 0.46

    def generate_radial_modes(self):
        """
        Generate radial mode amplitudes
        """
        self.l0_amps = self.a0(self.l0_freqs)

    def generate_quadrupole_modes(self):
        """
        Generate quadrupole mode amplitudes
        """
        self.l2_amps = self.a0(self.l2_freqs) * np.sqrt(self.vis2)

    def generate_nominal_dipole_modes(self):
        """
        Generate nominal l=1 mode amplitudes
        """
        self.l1_nom_amps = self.a0(self.l1_nom_freqs) * np.sqrt(self.vis1)

    def generate_mixed_dipole_modes(self):
        """
        Generate mixed l=1 mode amplitudes
        """
        self.l1_mixed_amps = []
        radial_order = np.unique(self.l1_np)
        for i in range(len(radial_order)):
            cond = (self.l1_np == radial_order[i])
            self.l1_mixed_amps = np.append(self.l1_mixed_amps,
                                           self.l1_nom_amps[i] * (1 - self.l1_zeta[cond])**1/2)
    
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
    frequencies = frequencies.Frequencies(frequency=frequency,
                                          numax=103.2, 
                                          delta_nu=9.57, 
                                          radial_order_range=[-5, 5])
    # Eventually want this to read in from a configuration file
    params = {'calc_mixed': True,
              'calc_rot': True,
              'DPi1': 77.9,
              'coupling': 0.2,
              'eps_g': 0.0,
              'split_core': 0.5,
              'split_env': 0.0,
              'l': 1,
              'method': 'simple'}
    frequencies(params)

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
    