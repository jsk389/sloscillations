# -*- coding: utf-8 -*-

import mixed_modes_utils as mixed_modes
import numpy as np 
import scaling_relations as scalings


class Frequencies:

    def __init__(self, frequency, numax, delta_nu, radial_order_range=[-5, 5]):

        self.numax = numax
        self.delta_nu = delta_nu
        self.frequency = frequency
        self.radial_order_range = radial_order_range

        # Epsilon 
        self.epsilon_p = scalings.epsilon_p(self.delta_nu)
        # Alpha
        self.alpha = scalings.alpha(self.delta_nu)
        
        self.n_max = np.floor(self.numax/self.delta_nu - self.epsilon_p)
        # Set up radial orders
        self.n = np.arange(self.n_max + self.radial_order_range[0],
                           self.n_max + self.radial_order_range[1]+1,
                           1)
        
        # Small separations
        self.d02 = scalings.d02_scaling(self.n, self.delta_nu)
        self.d01 = scalings.d02_scaling(self.n)

    def asymptotic_expression(self, l, d0l=0):
        """
        Generate mode frequencies according to the asymptotic expression
        """
        n_vals = self.n - self.n_max
        return (self.n + self.epsilon_p + l/2, self.alpha/2*n_vals**2) * self.delta_nu \
                - d0l, np.array([l]*len(self.n))

    def generate_radial_modes(self):
        self.l0_freqs, self.l0_l = self.asymptotic_expression(l=0)
        
    def generate_quadrupole_modes(self):
        self.l2_freqs, self.l2_l = self.asymptotic_expression(l=2, d0l=self.d02)

    def generate_nominal_dipole_modes(self):
        self.l1_nom_freqs, self.l1_l = self.asymptotic_expression(l=1, d0l=self.d01)

    def generate_mixed_modes(self, DPi1, coupling, eps_g):
        self.l1_mixed_freqs, self.l1_ml = mixed_modes.all_mixed_l1_freqs(self.delta_nu,
                                                                         self.l1_nom_freqs,
                                                                         DPi1,
                                                                         eps_g,
                                                                         coupling)    

if __name__=="__main__":
    pass