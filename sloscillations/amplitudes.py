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

    def __init__(self, parentInst):
        
        # Inherit frequencies class so have all frequencies stored
        if self.Henv is None:
            self.Henv = scalings.Henv_scaling(self.numax,
                                              evo_state=self.evo_state)
        if self.denv is None:
            self.denv = scalings.denv_scaling(self.numax,
                                              evo_state=self.evo_state)
        if self.amax is None:
            self.amax = scalings.amax_scaling(self.Henv,
                                              self.numax,
                                              self.delta_nu,
                                              mission=self.mission)
        if self.inclination_angle is None:
            #print("No inclination angle given therefore defaulting to 90 degrees")
            self.inclination_angle = 90.0
        self.a0 = radial_modes.calculate_a0(self.frequency, 
                                            self.numax, 
                                            self.amax, 
                                            self.denv)

        if self.mission == 'Kepler':
            if self.vis_tot is None:
                self.vis_tot = 3.16
            if self.vis1 is None:
                self.vis1 = 1.54
            if self.vis2 is None:
                self.vis2 = 0.58
            # Need to check this value!
            if self.vis3 is None:
                self.vis3 = 0.07
        elif self.mission == 'TESS':
            if self.vis_tot is None:
                self.vis_tot = 2.94
            if self.vis1 is None:
                self.vis1 = 1.46
            if self.vis2 is None:
                self.vis2 = 0.46
            # Need to update this properly!
            if self.vis3 is None:
                self.vis3 = 0.05

    def generate_radial_modes(self):
        """
        Generate radial mode amplitudes
        """
        self.l0_amps = self.a0(self.l0_freqs)
        self.mode_data.loc[self.mode_data['l'] == 0, 'amplitude'] = self.l0_amps

    def generate_quadrupole_modes(self):
        """
        Generate quadrupole mode amplitudes
        """
        self.l2_amps = self.a0(self.l2_freqs) * np.sqrt(self.vis2)
        self.mode_data.loc[self.mode_data['l'] == 2, 'amplitude'] = self.l2_amps

    def generate_octupole_modes(self):
        """
        Generate l=3 mode amplitudes
        """
        self.l3_amps = self.a0(self.l3_freqs) * np.sqrt(self.vis3)
        self.mode_data.loc[self.mode_data['l'] == 3, 'amplitude'] = self.l3_amps 

    def generate_nominal_dipole_modes(self):
        """
        Generate nominal l=1 mode amplitudes
        """
        self.l1_nom_amps = self.a0(self.l1_nom_freqs) * np.sqrt(self.vis1)
        self.mode_data.loc[self.mode_data['l'] == -1, 'amplitude'] = self.l1_nom_amps

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
        #print(self.__dict__['inclination_angle'])
        #if not hasattr(self, inclination_angle):
        #        sys.exit("No inclination angle given .... exiting!")
        
        # Relative visibilities due to inclination angle - sqrt due to amplitude
        # not calculated for heights
        m0_factor = np.sqrt(np.cos(np.radians(self.inclination_angle))**2)
        m1_factor = np.sqrt(0.5 * np.sin(np.radians(self.inclination_angle))**2)

        self.mode_data.loc[(self.mode_data['l'] == 1) & (self.mode_data['m'] == 0), 'amplitude'] = m0_factor * self.l1_mixed_amps

        if self.calc_rot:

            # Also generate linewidths for rotationally split components if they exist
            if hasattr(self, 'l1_mixed_freqs_p1') and (self.method=='simple'):
                self.l1_mixed_amps_p1 = []
                radial_order = np.unique(self.l1_np)
                for i in range(len(radial_order)):
                    cond = (self.l1_np == radial_order[i])
                    self.l1_mixed_amps_p1 = np.append(self.l1_mixed_amps_p1,
                                                m1_factor * self.l1_nom_amps[i] * (1 - self.l1_zeta[cond])**1/2)
                self.mode_data.loc[(self.mode_data['l'] == 1) & (self.mode_data['m'] == +1), 'amplitude'] = self.l1_mixed_amps_p1

            elif hasattr(self, 'l1_mixed_freqs_p1') and (self.method=='Mosser'):
                sys.exit()
            if hasattr(self, 'l1_mixed_freqs_n1') and (self.method=='simple'):
                self.l1_mixed_amps_n1 = []
                radial_order = np.unique(self.l1_np)
                for i in range(len(radial_order)):
                    cond = (self.l1_np == radial_order[i])
                    self.l1_mixed_amps_n1 = np.append(self.l1_mixed_amps_n1,
                                                m1_factor * self.l1_nom_amps[i] * (1 - self.l1_zeta[cond]))
                self.mode_data.loc[(self.mode_data['l'] == 1) & (self.mode_data['m'] == -1), 'amplitude'] = self.l1_mixed_amps_n1

            else:
                sys.exit()

    def __call__(self, entries=dict()):
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
        if self.calc_l3:
            self.generate_octupole_modes()
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
    