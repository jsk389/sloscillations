# -*- coding: utf-8 -*-

from . import mixed_modes_utils as mixed_modes
import numpy as np 
from . import scaling_relations as scalings

import matplotlib.pyplot as plt
import pandas as pd

class Frequencies(object):

    def __init__(self, frequency, numax, delta_nu, radial_order_range=[-5, 5]):

        self.frequency = frequency
        self.numax = numax
        self.delta_nu = delta_nu
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
        self.d02 = scalings.d02_scaling(self.n-1, self.delta_nu)
        self.d01 = scalings.d01_scaling(self.n)

        # Dipole mixed mode frequency calculation method
        #self.calc_method = 'Mosser2018_update'

    def asymptotic_expression(self, l, d0l=0, n=None):
        """
        Generate mode frequencies according to the asymptotic expression.
        Added n as an optional input because have to generate nominal p-modes
        one radial order above and below radial modes to generate mixed modes
        with full coverage.
        """
        if n is None:
            n_vals = self.n - self.n_max
            return (self.n + self.epsilon_p + l/2 + self.alpha/2*n_vals**2) * self.delta_nu \
                    - d0l, np.array([l]*len(self.n))
        else:
            n_vals = n - self.n_max
            return (n + self.epsilon_p + l/2 + self.alpha/2*n_vals**2) * self.delta_nu \
                    - d0l, np.array([l]*len(n))

    def generate_radial_modes(self):
        """
        Generate l=0 mode frequencies
        """
        self.l0_freqs, self.l0_l = self.asymptotic_expression(l=0)
        self.full_freqs = pd.DataFrame(data=np.c_[self.n.astype(int), 
                                            self.l0_l.astype(int), 
                                            np.nan*np.ones_like(self.l0_freqs),
                                            self.l0_freqs], 
                                       columns=['n', 'l', 'm', 'Frequency'])
        
    def generate_quadrupole_modes(self):
        """
        Generate l=2 mode frequencies
        """
        self.l2_freqs, self.l2_l = self.asymptotic_expression(l=2, d0l=self.d02, n=self.n-1)
        tmp = pd.DataFrame(data=np.c_[(self.n - 1).astype(int), 
                                      self.l2_l.astype(int), 
                                      np.nan*np.ones_like(self.l2_freqs),
                                      self.l2_freqs], 
                           columns=['n', 'l', 'm', 'Frequency'])         
        self.full_freqs = self.full_freqs.append(tmp, ignore_index=True)

    def generate_nominal_dipole_modes(self):
        """
        Generate l=1 nominal p-mode frequencies
        """
        n = np.arange(self.n_max + self.radial_order_range[0],
                                self.n_max + self.radial_order_range[1]+1,
                                1)

        self.l1_nom_freqs, self.l1_l = self.asymptotic_expression(l=1, d0l=self.d01, n=n)
        tmp = pd.DataFrame(data=np.c_[n.astype(int), 
                                      [-1]*len(self.l1_nom_freqs), 
                                      np.nan*np.ones_like(self.l1_nom_freqs),
                                      self.l1_nom_freqs], 
                           columns=['n', 'l', 'm', 'Frequency'])         
        self.full_freqs = self.full_freqs.append(tmp, ignore_index=True)

    def generate_mixed_dipole_modes(self, DPi1, coupling, eps_g, nom_p):
        """
        Generate l=1 mixed mode frequencies
        """
        self.l1_mixed_freqs, self.l1_zeta, self.l1_np, self.l1_g_freqs, self.l1_ng = mixed_modes.all_mixed_l1_freqs(self.delta_nu,
                                                                                       self.l0_freqs,
                                                                                       nom_p,
                                                                                       DPi1,
                                                                                       eps_g,
                                                                                       coupling,
                                                                                       return_order=True,
                                                                                       calc_zeta=True,
                                                                                       method=self.calc_method)  
        #plt.plot(self.l1_mixed_freqs, self.l1_zeta)
        #plt.show()
        #if len(self.l1_nom_freqs) > 1:                           
                 
        cond = (self.l1_mixed_freqs > self.l0_freqs.min()) & (self.l1_mixed_freqs < self.l0_freqs.max())
        #else:
        #cond = (self.l1_mixed_freqs > self.l1_nom_freqs - self.delta_nu/2) & (self.l1_mixed_freqs < self.l1_nom_freqs + self.delta_nu/2)
        self.l1_mixed_freqs = self.l1_mixed_freqs[cond]
        self.l1_g_freqs = self.l1_g_freqs[cond]
        self.l1_zeta = self.l1_zeta[cond]
        self.l1_ng = self.l1_ng[cond]
        # Add correction to make these equal actual radial mode order
        # not just index of nominal l=1 p-mode frequency
        self.l1_np = self.l1_np[cond] + self.n.min()
        # Still need to assign radial  order to mixed modes!
        tmp = pd.DataFrame(data=np.c_[self.l1_np,
                                      [1]*len(self.l1_mixed_freqs), 
                                      [0]*len(self.l1_mixed_freqs),
                                      self.l1_mixed_freqs,
                                      np.arange(0, len(self.l1_mixed_freqs), 1), 
                                      self.l1_zeta], 
                           columns=['n', 'l', 'm', 'Frequency', 'n_g', 'zeta'])         
        self.full_freqs = self.full_freqs.append(tmp, ignore_index=True)

    def generate_rotational_splittings(self, split_core, DPi1, coupling, eps_g, split_env=0., l=1, 
                                       method='simple'):
        """
        Generate rotational splittings
        """

        #print(self.calc_method)

        if l == 1:
            if method == 'simple':
                # Use method from Dehevuels et al. (2015)
                splitting = (split_core/2 - split_env) * self.l1_zeta + split_env
                self.l1_mixed_freqs_p1 = self.l1_mixed_freqs + splitting
                self.l1_mixed_freqs_n1 = self.l1_mixed_freqs - splitting
                tmp = pd.DataFrame(data=np.c_[self.l1_np,
                                              [1]*len(self.l1_mixed_freqs), 
                                              [+1]*len(self.l1_mixed_freqs),
                                              self.l1_mixed_freqs_p1,
                                              np.arange(0, len(self.l1_mixed_freqs), 1), 
                                              self.l1_zeta], 
                                   columns=['n', 'l', 'm', 'Frequency', 'n_g', 'zeta'])         
                self.full_freqs = self.full_freqs.append(tmp, ignore_index=True)                
                tmp = pd.DataFrame(data=np.c_[self.l1_np,
                                              [1]*len(self.l1_mixed_freqs), 
                                              [-1]*len(self.l1_mixed_freqs),
                                              self.l1_mixed_freqs_n1,
                                              np.arange(0, len(self.l1_mixed_freqs), 1), 
                                              self.l1_zeta], 
                                   columns=['n', 'l', 'm', 'Frequency', 'n_g', 'zeta'])         
                self.full_freqs = self.full_freqs.append(tmp, ignore_index=True) 

            elif method == 'Mosser':
                # User method from Mosser et al. (2019)
                # Stretch pds for given DPi1, coupling and eps_g values
                zeta_func = mixed_modes.zeta_interp(self.frequency,
                                                          self.l1_nom_freqs, 
                                                          self.delta_nu, 
                                                          DPi1, 
                                                          coupling, 
                                                          eps_g)
                self.l1_mixed_freqs_p1, self.l1_mixed_freqs_n1, \
                    self.l1_int_zeta_p, self.l1_int_zeta_n = mixed_modes.l1_theoretical_rot_M(self.l1_mixed_freqs,
                                                                                              split_core, 
                                                                                              zeta_func)
                tmp = pd.DataFrame(data=np.c_[np.arange(0, len(self.l1_mixed_freqs), 1), 
                                    [1]*len(self.l1_mixed_freqs), 
                                    [+1]*len(self.l1_mixed_freqs),
                                    self.l1_mixed_freqs_p1,
                                    self.l1_int_zeta_p], 
                    columns=['n', 'l', 'm', 'Frequency', 'zeta'])   
                self.full_freqs = self.full_freqs.append(tmp, ignore_index=True)                
                tmp = pd.DataFrame(data=np.c_[np.arange(0, len(self.l1_mixed_freqs), 1), 
                                              [1]*len(self.l1_mixed_freqs), 
                                              [-1]*len(self.l1_mixed_freqs),
                                              self.l1_mixed_freqs_n1,
                                              self.l1_int_zeta_n], 
                                columns=['n', 'l', 'm', 'Frequency', 'zeta'])         
                self.full_freqs = self.full_freqs.append(tmp, ignore_index=True) 
            else:
                sys.exit("Oh dear, method keyword isn't correct!")
        else:
            pass

    def generate_tau_values(self, DPi1, coupling, eps_g):
        print("NEED TO UPDATE FUNCTION ARGUMENTS TO TAKE IN MIXED MODE CALCULATION METHOD!")
        new_frequency, tau, zeta_func = mixed_modes.stretched_pds(self.frequency, 
                                                    self.l1_nom_freqs,
                                                    self.delta_nu, DPi1, coupling, eps_g)
        self.l1_mixed_tau = mixed_modes.peaks_stretched_period(self.l1_mixed_freqs, 
                                                               new_frequency, 
                                                               tau)
        self.l1_mixed_tau_p1 = mixed_modes.peaks_stretched_period(self.l1_mixed_freqs_p1, 
                                                               new_frequency, 
                                                               tau)
        self.l1_mixed_tau_n1 = mixed_modes.peaks_stretched_period(self.l1_mixed_freqs_n1, 
                                                               new_frequency, 
                                                               tau)

    def plot_echelle(self, l0=True, l2=True, l1=True, mixed=True, rotation=True):
        """
        Plot an echelle of frequencies
        """
        if l0:
            df_l0 = self.full_freqs.loc[self.full_freqs['l'] == 0, ]
            plt.plot(df_l0.Frequency % self.delta_nu - self.epsilon_p + 0.1*self.delta_nu, 
                    df_l0.Frequency,
                    color='r', marker='D', label='$\ell=0$', linestyle='None', zorder=1, markersize=5)
        if l2:
            df_l2 = self.full_freqs.loc[self.full_freqs['l'] == 2, ]
            plt.plot(df_l2.Frequency % self.delta_nu - self.epsilon_p + 0.1*self.delta_nu, 
                    df_l2.Frequency, 
                    color='g', marker='s', label='$\ell=2$', linestyle='None', zorder=1, markersize=5)
        if l1:
            df_l1 = self.full_freqs.loc[self.full_freqs['l'] == -1, ]
            plt.plot(df_l1.Frequency % self.delta_nu - self.epsilon_p + 0.1*self.delta_nu, 
                    df_l1.Frequency, 
                    color='b', marker='o', label='Nominal $\ell=1$', linestyle='None', zorder=1, markersize=5)
        if mixed:
            df_l1_m0 = self.full_freqs.loc[(self.full_freqs['l'] == 1) & (self.full_freqs['m'] == 0), ]
            if rotation:
                df_l1_mp1 = self.full_freqs.loc[(self.full_freqs['l'] == 1) & (self.full_freqs['m'] == +1), ]
                df_l1_mn1 = self.full_freqs.loc[(self.full_freqs['l'] == 1) & (self.full_freqs['m'] == -1), ]

            for i in range(len(df_l1_m0.Frequency)):
                color = next(plt.gca()._get_lines.prop_cycler)['color']
                plt.plot(df_l1_m0.Frequency.iloc[i] % self.delta_nu - self.epsilon_p + 0.1*self.delta_nu, 
                        df_l1_m0.Frequency.iloc[i], 
                        color=color, marker='v', label='Mixed $\ell=1$, $m=0$', linestyle='None', zorder=0, markersize=3)
                if rotation:
                    plt.plot(df_l1_mp1.Frequency.iloc[i] % self.delta_nu - self.epsilon_p + 0.1*self.delta_nu, 
                            df_l1_mp1.Frequency.iloc[i],
                            color=color, marker='<', label='Mixed $\ell=1$, $m=+1$', linestyle='None', zorder=0, markersize=3)
                    plt.plot(df_l1_mn1.Frequency.iloc[i] % self.delta_nu - self.epsilon_p + 0.1*self.delta_nu, 
                            df_l1_mn1.Frequency.iloc[i], 
                            color=color, marker='>', label='Mixed $\ell=1$, $m=-1$', linestyle='None', zorder=0, markersize=3)
        plt.xlim(0, self.delta_nu)
        plt.xlabel(r'$\nu$ mod $\Delta\nu$ ($\mu$Hz)', fontsize=18)
        plt.ylabel(r'Frequency ($\mu$Hz)', fontsize=18)
        #plt.show()

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
        # l=1 mixed modes
        if self.calc_mixed:
            self.generate_mixed_dipole_modes(DPi1=self.DPi1, 
                                             coupling=self.coupling, 
                                             eps_g=self.eps_g,
                                             nom_p = self.l1_nom_freqs)
            if self.calc_rot:
                # l=1 rotation
                if self.method == 'simple':
                    # This is for consistency between formulations, so simple and Mosser
                    # give same results for given core splitting.
                    self.split_core *= 2
                self.generate_rotational_splittings(split_core=self.split_core, 
                                                        DPi1=self.DPi1, 
                                                        coupling=self.coupling, 
                                                        eps_g=self.eps_g, 
                                                        split_env=self.split_env, 
                                                        l=1, method=self.method)     
        if (self.calc_rot) & (self.l > 1):
            # l=1 rotation
            if self.method == 'simple':
                # This is for consistency between formulations, so simple and Mosser
                # give same results for given core splitting.
                self.split_core /= 2
            self.generate_rotational_splittings(split_core=self.split_core, 
                                                    DPi1=self.DPi1, 
                                                    coupling=self.coupling, 
                                                    eps_g=self.eps_g, 
                                                    split_env=self.split_env, 
                                                    l=1, method=self.method)     

if __name__=="__main__":
    

    frequency = np.arange(0.00787, 283., 0.00787)

    # Set up frequencies class
    frequencies = Frequencies(frequency=frequency,
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

    # Plot and echelle to check everything makes sense
    frequencies.plot_echelle(mixed=True, rotation=True)
    