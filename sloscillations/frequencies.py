# -*- coding: utf-8 -*-

from . import mixed_modes_utils as mixed_modes
import numpy as np 
from . import scaling_relations as scalings

import matplotlib.pyplot as plt
import pandas as pd
import yaml

class Frequencies(object):

    def _setup_attrs(self):
        ## Setup attributed
        self.delta_nu = None
        self.epsilon_p = None
        self.alpha = None
        self.radial_order_range = None
        self.d01 = None
        self.d02 = None
        self.d03 = None
        self.Henv = None
        self.denv = None
        self.amax = None
        self.DPi1 = None
        self.coupling = None
        self.eps_g = None
        self.split_core = None
        self.split_env = 0.0
        self.calc_l0 = True
        self.calc_l2 = True
        self.calc_l3 = True
        self.calc_nom_l1 = True
        self.calc_mixed = True
        self.calc_rot = False
        self.calc_method = None
        self.l = 1
        self.method = None
        self.mission = 'Kepler'
        self.evo_state = 'RGB'
        self.inclination_angle = None
        self.vis_tot = None
        self.vis1 = None
        self.vis2 = None
        self.vis3 = None
        self.T = None
        self.osamp = 1
        self.store_mode_data = False

    def __init__(self, frequency, numax, **kwargs):

        # Predefine all attributes with default values
        self._setup_attrs()
        #####################
        self.frequency = frequency
        self.numax = numax
        
        # Update attributes from kwargs
        self.__dict__.update(kwargs)

        #self.delta_nu = delta_nu
        #self.radial_order_range = radial_order_range

        if self.delta_nu is None:
            self.delta_nu = scalings.delta_nu(self.numax)
        #else:
        #    self.delta_nu = delta_nu

        # Epsilon
        if self.epsilon_p is None:
            self.epsilon_p = scalings.epsilon_p(self.delta_nu)
        #else:
        #    self.epsilon_p = epsilon_p
        # Alpha
        if self.alpha is None:
            self.alpha = scalings.alpha(self.delta_nu)
        #else:
        #    self.alpha = alpha

        
       
        # use np.floor or ?
        self.n_max = (self.numax/self.delta_nu) - self.epsilon_p
        # Radial order range
        if self.radial_order_range is None:
            max_radial_order = np.floor(self.frequency.max()/self.delta_nu)
            self.radial_order_range = [-self.n_max, max_radial_order-self.n_max]
        # Set up radial orders
        self.n = np.arange(np.floor(self.n_max) + np.floor(self.radial_order_range[0]),
                           np.floor(self.n_max) + np.floor(self.radial_order_range[1])+1,
                           1)

        self.n = self.n[self.n > 0]        

        # Compute individual delta nu values
        self.delta_nu_indiv = self.delta_nu * (1 +  self.alpha * (self.n - self.n_max))

        # Small separations
        self.d03 = scalings.d03_scaling(self.n-1, self.delta_nu)
        self.d02 = scalings.d02_scaling(self.n-1, self.delta_nu)
        self.d01 = scalings.d01_scaling(self.n, self.delta_nu)

        if self.method is None:
            self.method = "simple"

        # Dipole mixed mode frequency calculation method
        if self.calc_method is None:
            self.calc_method = 'Mosser2018_update'

        # Period spacing
        if self.DPi1 is None:
            self.DPi1 = scalings.DPi1_scaling(self.delta_nu, self.evo_state)
        # Coupling:
        if self.coupling is None:
            self.coupling = scalings.coupling_scaling(self.evo_state)
        # Epsilon g:
        if self.eps_g is None:
            self.eps_g = 0.0

        if self.split_core is None:
            self.calc_rot = False

        #tmp = self.__dict__

        #tmp = {k: (v.tolist() if type(v) == np.ndarray else v) for k, v in tmp.items()}

        #with open('freqs_data.json', 'w') as f:
        #    yaml.dump(tmp, f)#, ensure_ascii=False, indent=4)
        #sys.exit()
        #self.freq_attrs = self.__dict__

    def asymptotic_expression(self, l, d0l=0, n=None):
        """
        Generate mode frequencies according to the asymptotic expression.
        Added n as an optional input because have to generate nominal p-modes
        one radial order above and below radial modes to generate mixed modes
        with full coverage.
        """
        if n is None:
            n_vals = self.n - self.n_max
            return (self.n + self.epsilon_p + l/2 + (self.alpha/2)*n_vals**2) * self.delta_nu \
                    - d0l, np.array([l]*len(self.n))
        else:
            n_vals = n - self.n_max
            return (n + self.epsilon_p + l/2 + (self.alpha/2)*n_vals**2) * self.delta_nu \
                    - d0l, np.array([l]*len(n))

    def generate_radial_modes(self):
        """
        Generate l=0 mode frequencies
        """
        self.l0_freqs, self.l0_l = self.asymptotic_expression(l=0)

        if self.store_mode_data:
            self.mode_data = pd.DataFrame(data=np.c_[self.n.astype(int), 
                                                self.l0_l.astype(int), 
                                                np.nan*np.ones_like(self.l0_freqs),
                                                self.l0_freqs], 
                                        columns=['n', 'l', 'm', 'frequency'])
        
    def generate_quadrupole_modes(self):
        """
        Generate l=2 mode frequencies
        """
        self.l2_freqs, self.l2_l = self.asymptotic_expression(l=2, d0l=self.d02, n=self.n-1)

        if self.store_mode_data:
            tmp = pd.DataFrame(data=np.c_[(self.n - 1).astype(int), 
                                        self.l2_l.astype(int), 
                                        np.nan*np.ones_like(self.l2_freqs),
                                        self.l2_freqs], 
                            columns=['n', 'l', 'm', 'frequency'])         
            self.mode_data = self.mode_data.append(tmp, ignore_index=True, sort=True)

    def generate_octupole_modes(self):
        """
        Generate l=3 mode frequencies
        """
        self.l3_freqs, self.l3_l = self.asymptotic_expression(l=3, d0l=self.d03, n=self.n-1)

        # 20/12/2020 Make sure l3 are in desired range
        #cond = (self.l3_freqs > self.l0_freqs.min()) & (self.l3_freqs < self.l0_freqs.max())
        #self.l3_freqs = self.l3_freqs[cond]
        #self.l3_l = self.l3_l[cond]
        if self.store_mode_data:
            tmp = pd.DataFrame(data=np.c_[(self.n - 1).astype(int), 
                                        self.l3_l.astype(int), 
                                        np.nan*np.ones_like(self.l3_freqs),
                                        self.l3_freqs], 
                            columns=['n', 'l', 'm', 'frequency'])         
            self.mode_data = self.mode_data.append(tmp, ignore_index=True, sort=True)  

    def generate_nominal_dipole_modes(self):
        """
        Generate l=1 nominal p-mode frequencies
        """
        #n = np.arange(np.floor(self.n_max) + self.radial_order_range[0],
        #                        np.floor(self.n_max) + self.radial_order_range[1]+1,
        #                        1)
        #print(n, self.n)

        self.l1_nom_freqs, self.l1_l = self.asymptotic_expression(l=1, d0l=self.d01, n=self.n)
        #print(f"Nominal p-mode freqs: {self.l1_nom_freqs}")
        if self.store_mode_data:
            tmp = pd.DataFrame(data=np.c_[self.n.astype(int), 
                                        [-1]*len(self.l1_nom_freqs), 
                                        np.nan*np.ones_like(self.l1_nom_freqs),
                                        self.l1_nom_freqs], 
                            columns=['n', 'l', 'm', 'frequency'])         
            self.mode_data = self.mode_data.append(tmp, ignore_index=True, sort=True)

    def generate_mixed_dipole_modes(self, DPi1, coupling, eps_g, nom_p):
        """
        Generate l=1 mixed mode frequencies
        """
        self.l1_mixed_freqs, self.l1_np, self.l1_g_freqs, self.l1_ng = mixed_modes.all_mixed_l1_freqs(self.delta_nu_indiv,
                                                                                       self.l0_freqs,
                                                                                       nom_p,
                                                                                       DPi1,
                                                                                       eps_g,
                                                                                       coupling,
                                                                                       return_order=True,
                                                                                       method=self.calc_method)  
        #plt.plot(self.l1_mixed_freqs, self.l1_zeta)
        #plt.show()
        #if len(self.l1_nom_freqs) > 1:                           
        #print("TODO: Need to fix issue where frequencies are generated outside frequency array given!")
        #cond = (self.l1_mixed_freqs > self.l0_freqs.min()) & (self.l1_mixed_freqs < self.l0_freqs.max())
        cond = (self.l1_mixed_freqs > self.frequency.min()) & (self.l1_mixed_freqs < self.frequency.max())
        #else:
        #cond = (self.l1_mixed_freqs > self.l1_nom_freqs - self.delta_nu/2) & (self.l1_mixed_freqs < self.l1_nom_freqs + self.delta_nu/2)
        self.l1_mixed_freqs = self.l1_mixed_freqs[cond]
        self.l1_g_freqs = self.l1_g_freqs[cond]
        self.l1_ng = self.l1_ng[cond]
        # Add correction to make these equal actual radial mode order
        # not just index of nominal l=1 p-mode frequency
        self.l1_np = self.l1_np[cond] + self.n.min()

        # Create zeta
        self.l1_zeta = np.interp(self.l1_mixed_freqs, self.osamp_frequency, self.zeta)

        if self.store_mode_data:
            # Still need to assign radial  order to mixed modes!
            tmp = pd.DataFrame(data=np.c_[self.l1_np,
                                        [1]*len(self.l1_mixed_freqs), 
                                        [0]*len(self.l1_mixed_freqs),
                                        self.l1_mixed_freqs,
                                        self.l1_ng, #np.arange(0, len(self.l1_mixed_freqs), 1), 
                                        self.l1_zeta], 
                            columns=['n', 'l', 'm', 'frequency', 'n_g', 'zeta'])         
            self.mode_data = self.mode_data.append(tmp, ignore_index=True, sort=True)

    def generate_rotational_splittings(self, split_core, DPi1=80.0, coupling=0.2, eps_g=0.0, split_env=0., l=1, 
                                       method='simple'):
        """
        Generate rotational splittings
        """

        #print(self.calc_method)

        if l == 1:
            if method == 'simple':
                # Use method from Dehevuels et al. (2015)
                splitting = (split_core - split_env) * self.l1_zeta + split_env
                self.l1_mixed_freqs_p1 = self.l1_mixed_freqs + splitting
                self.l1_mixed_freqs_n1 = self.l1_mixed_freqs - splitting

                if self.store_mode_data:
                    tmp = pd.DataFrame(data=np.c_[self.l1_np,
                                                [1]*len(self.l1_mixed_freqs), 
                                                [+1]*len(self.l1_mixed_freqs),
                                                self.l1_mixed_freqs_p1,
                                                self.l1_ng, #np.arange(0, len(self.l1_mixed_freqs), 1), 
                                                self.l1_zeta], 
                                    columns=['n', 'l', 'm', 'frequency', 'n_g', 'zeta'])         
                    self.mode_data = self.mode_data.append(tmp, ignore_index=True, sort=True)           
                    tmp = pd.DataFrame(data=np.c_[self.l1_np,
                                                [1]*len(self.l1_mixed_freqs), 
                                                [-1]*len(self.l1_mixed_freqs),
                                                self.l1_mixed_freqs_n1,
                                                self.l1_ng, #np.arange(0, len(self.l1_mixed_freqs), 1), 
                                                self.l1_zeta], 
                                    columns=['n', 'l', 'm', 'frequency', 'n_g', 'zeta'])         
                    self.mode_data = self.mode_data.append(tmp, ignore_index=True, sort=True)

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
                
                if self.store_mode_data:
                    tmp = pd.DataFrame(data=np.c_[np.arange(0, len(self.l1_mixed_freqs), 1), 
                                        [1]*len(self.l1_mixed_freqs), 
                                        [+1]*len(self.l1_mixed_freqs),
                                        self.l1_mixed_freqs_p1,
                                        self.l1_int_zeta_p], 
                        columns=['n', 'l', 'm', 'frequency', 'zeta'])   
                    self.mode_data = self.mode_data.append(tmp, ignore_index=True, sort=True)                
                    tmp = pd.DataFrame(data=np.c_[np.arange(0, len(self.l1_mixed_freqs), 1), 
                                                [1]*len(self.l1_mixed_freqs), 
                                                [-1]*len(self.l1_mixed_freqs),
                                                self.l1_mixed_freqs_n1,
                                                self.l1_int_zeta_n], 
                                    columns=['n', 'l', 'm', 'frequency', 'zeta'])         
                    self.mode_data = self.mode_data.append(tmp, ignore_index=True, sort=True)
            else:
                sys.exit("Oh dear, method keyword isn't correct!")
        else:
            pass

    def generate_tau_values(self, compute_all_tau=True, shift=0.0):
        #DPi1, coupling, eps_g, 
        # Recompute tau if needed, but for example if only changing rotational
        # splitting then don't need to recompute tau for m=0 modes, only need 
        # to interpolate to compute tau for m=+/-1
    
        if compute_all_tau:
            self.new_frequency, self.tau, self.new_zeta = mixed_modes.stretched_pds(self.osamp_frequency, 
                                                                                     self.zeta)#,
                                                                                     #oversample=self.osamp)
                                                                                    
        if self.calc_mixed:
            self.l1_mixed_tau = mixed_modes.peaks_stretched_period(self.l1_mixed_freqs, 
                                                                self.new_frequency, 
                                                                self.tau)

            # Compute shift
            self.shift = mixed_modes.compute_tau_shift(self.l1_mixed_tau, self.DPi1)
            self.l1_mixed_tau -= self.shift * self.DPi1
            if self.calc_rot:
                self.l1_mixed_tau_p1 = mixed_modes.peaks_stretched_period(self.l1_mixed_freqs_p1, 
                                                                    self.new_frequency, 
                                                                    self.tau)
                self.l1_mixed_tau_p1 -= self.shift * self.DPi1
                self.l1_mixed_tau_n1 = mixed_modes.peaks_stretched_period(self.l1_mixed_freqs_n1, 
                                                                    self.new_frequency, 
                                                                    self.tau)
                self.l1_mixed_tau_n1 -= self.shift * self.DPi1

    def plot_echelle(self, l0=True, l2=True, l3=True, l1=True, mixed=True, 
                     rotation=True, shift=None):
        """
        Plot an echelle of frequencies
        """
        if shift is None:
            shift = 0.1 * self.delta_nu

        if l0:
            df_l0 = self.mode_data.loc[self.mode_data['l'] == 0, ]
            plt.plot(df_l0.frequency % self.delta_nu - self.epsilon_p + shift, 
                    df_l0.frequency,
                    color='r', marker='D', label='$\ell=0$', linestyle='None', zorder=1, markersize=5)
        if l2:
            df_l2 = self.mode_data.loc[self.mode_data['l'] == 2, ]
            plt.plot(df_l2.frequency % self.delta_nu - self.epsilon_p + shift, 
                    df_l2.frequency, 
                    color='g', marker='s', label='$\ell=2$', linestyle='None', zorder=1, markersize=5)
        if l3:
            df_l3 = self.mode_data.loc[self.mode_data['l'] == 3, ]
            plt.plot(df_l3.frequency % self.delta_nu - self.epsilon_p + shift, 
                    df_l3.frequency, 
                    color='c', marker='*', label='$\ell=3$', linestyle='None', zorder=1, markersize=5) 
        if l1:
            df_l1 = self.mode_data.loc[self.mode_data['l'] == -1, ]
            plt.plot(df_l1.frequency % self.delta_nu - self.epsilon_p + shift, 
                    df_l1.frequency, 
                    color='b', marker='o', label='Nominal $\ell=1$', linestyle='None', zorder=1, markersize=5)
        if mixed:
            df_l1_m0 = self.mode_data.loc[(self.mode_data['l'] == 1) & (self.mode_data['m'] == 0), ]
            if rotation:
                df_l1_mp1 = self.mode_data.loc[(self.mode_data['l'] == 1) & (self.mode_data['m'] == +1), ]
                df_l1_mn1 = self.mode_data.loc[(self.mode_data['l'] == 1) & (self.mode_data['m'] == -1), ]

            for i in range(len(df_l1_m0.frequency)):
                color = next(plt.gca()._get_lines.prop_cycler)['color']
                plt.plot(df_l1_m0.frequency.iloc[i] % self.delta_nu - self.epsilon_p + shift, 
                        df_l1_m0.frequency.iloc[i], 
                        color=color, marker='v', label='Mixed $\ell=1$, $m=0$', linestyle='None', zorder=0, markersize=3)
                if rotation:
                    plt.plot(df_l1_mp1.frequency.iloc[i] % self.delta_nu - self.epsilon_p + shift, 
                            df_l1_mp1.frequency.iloc[i],
                            color=color, marker='<', label='Mixed $\ell=1$, $m=+1$', linestyle='None', zorder=0, markersize=3)
                    plt.plot(df_l1_mn1.frequency.iloc[i] % self.delta_nu - self.epsilon_p + shift, 
                            df_l1_mn1.frequency.iloc[i], 
                            color=color, marker='>', label='Mixed $\ell=1$, $m=-1$', linestyle='None', zorder=0, markersize=3)
        plt.xlim(0, self.delta_nu)
        plt.xlabel(r'$\nu$ mod $\Delta\nu$ ($\mu$Hz)', fontsize=18)
        plt.ylabel(r'Frequency ($\mu$Hz)', fontsize=18)
        #plt.show()

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
        # l=3 modes
        if self.calc_l3:
            self.generate_octupole_modes()
        # l=1 nominal p-modes
        if self.calc_nom_l1:
            self.generate_nominal_dipole_modes()
        else:
            n = np.arange(np.floor(self.n_max) + self.radial_order_range[0],
                                    np.floor(self.n_max) + self.radial_order_range[1]+1,
                                    1)
            tmp = pd.DataFrame(data=np.c_[n.astype(int), 
                                        [-1]*len(self.l1_nom_freqs), 
                                        np.nan*np.ones_like(self.l1_nom_freqs),
                                        self.l1_nom_freqs], 
                            columns=['n', 'l', 'm', 'frequency']) 
            self.mode_data = self.mode_data.append(tmp, ignore_index=True, sort=True)
        
        # Compute zeta
        self.osamp_frequency, self.zeta = mixed_modes.interpolated_zeta(
                                                        self.frequency,
                                                        self.delta_nu_indiv,
                                                        self.l0_freqs,
                                                        self.l1_nom_freqs,
                                                        self.coupling,
                                                        self.DPi1,
                                                        self.osamp,
                                                        plot=False,
                                                        )

        # l=1 mixed modes
        if self.calc_mixed:
            self.generate_mixed_dipole_modes(DPi1=self.DPi1, 
                                             coupling=self.coupling, 
                                             eps_g=self.eps_g,
                                             nom_p = self.l1_nom_freqs,
                                            )

            #plt.plot(self.frequency, self.zeta)
            #plt.plot(self.l1_mixed_freqs, self.l1_zeta, '.')
            #plt.show()
            #sys.exit()
            if self.calc_rot:
                # l=1 rotation
                if self.method == 'simple':
                    # This is for consistency between formulations, so simple and Mosser
                    # give same results for given core splitting.
                    self.generate_rotational_splittings(split_core=self.split_core, 
                                                            DPi1=self.DPi1, 
                                                            coupling=self.coupling, 
                                                            eps_g=self.eps_g, 
                                                            split_env=self.split_env, 
                                                            l=1, method=self.method)     
                else:
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
    plt.show()


    