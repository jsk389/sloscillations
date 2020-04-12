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

    def all_mixed_l1_freqs(self, nu_p, DPi1, eps_g, coupling):
        #l1_freqs = []
        #for i in range(len(freq_zero)):
        #    l1_freqs.append(find_mixed_l1_freq(freq, delta_nu, freq_zero[i], 
        #                                                      period_spacing, 
        #                                                      epsilon_g, coupling))
        #return np.array(l1_freqs)

        l1_freqs = []
        for i in range(len(nu_p)):
            tmp = self.find_mixed_l1_freqs(self.delta_nu, nu_p[i], 
                                    DPi1, eps_g, coupling)
            l1_freqs.append(tmp)

        return np.array(list(itertools.chain(*l1_freqs)))

    def find_mixed_l1_freqs(self, nu_p, DPi1, eps_g, coupling):
        """
        Find all mixed modes in a given radial order
        """

        nmin = np.ceil(1 / (DPi1*1e-6 * (nu_p + ( self.delta_nu/2))) - (1/2) - eps_g)
        nmax = np.ceil(1 / (DPi1*1e-6 * (nu_p - ( self.delta_nu/2))) - (1/2) - eps_g)
        #nmin -= 10
        nmax += 2
        #st.write(nmin, nmax)
        frequencies = []
        for i in np.arange(nmin, nmax, 1):
            tmp = self.find_mixed_l1_freq(nu_p, DPi1, eps_g, coupling, i)
            frequencies = np.append(frequencies, tmp)
    #    print(frequencies)
        return np.sort(frequencies[np.isfinite(frequencies)])

    def find_mixed_l1_freq(self, DeltaNu, pone, DPi1, eps_g, coupling, N):
        """
        Find individual mixed mode
        """

        def opt_func(nu):
            theta_p = (np.pi / self.delta_nu) * (nu - pone)
            theta_g = np.pi * (1 / (DPi1*1e-6*nu) - eps_g)
            return theta_p - np.arctan(coupling * np.tan(theta_g))
        
        lower_bound = 1 / (DPi1*1e-6 * (N + 1/2 + eps_g)) + np.finfo(float).eps * 1e4
        upper_bound = 1 / (DPi1*1e-6 * (N - 1 + 1/2 + eps_g)) - np.finfo(float).eps * 1e4


        #nu = np.linspace(pone-0.5*DeltaNu, pone+0.5*DeltaNu, 10000)

        #plt.plot(nu, opt_func(nu))
        #plt.axvline(lower_bound)
        #plt.axvline(upper_bound)
        #plt.show()
        #sys.exit()

        #st.write(lower_bound, upper_bound, pone)
        #sys.exit()
        #print(lower_bound, upper_bound)
        #x = np.linspace(lower_bound-10, upper_bound+10, 1000)
        #plt.plot(x, opt_func(x))
        #plt.show()
        #sys.exit()

        try:
            return brentq(opt_func, lower_bound, upper_bound)
        except:
            print("No mixed modes found!")
            return np.nan
    

if __name__=="__main__":
    pass