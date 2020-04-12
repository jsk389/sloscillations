# -*- coding: utf-8 -*-

import itertools
import numpy as np 

from scipy.optimize import brentq

def all_mixed_l1_freqs(delta_nu, nu_p, DPi1, eps_g, coupling):
    #l1_freqs = []
    #for i in range(len(freq_zero)):
    #    l1_freqs.append(find_mixed_l1_freq(freq, delta_nu, freq_zero[i], 
    #                                                      period_spacing, 
    #                                                      epsilon_g, coupling))
    #return np.array(l1_freqs)

    l1_freqs = []
    for i in range(len(nu_p)):
        tmp = find_mixed_l1_freqs(delta_nu, nu_p[i], 
                                DPi1, eps_g, coupling)
        l1_freqs.append(tmp)

    return np.array(list(itertools.chain(*l1_freqs)))

def find_mixed_l1_freqs(nu_p, DPi1, eps_g, coupling):
    """
    Find all mixed modes in a given radial order
    """

    nmin = np.ceil(1 / (DPi1*1e-6 * (nu_p + (delta_nu/2))) - (1/2) - eps_g)
    nmax = np.ceil(1 / (DPi1*1e-6 * (nu_p - (delta_nu/2))) - (1/2) - eps_g)
    #nmin -= 10
    nmax += 2
    #st.write(nmin, nmax)
    frequencies = []
    for i in np.arange(nmin, nmax, 1):
        tmp = find_mixed_l1_freq(nu_p, DPi1, eps_g, coupling, i)
        frequencies = np.append(frequencies, tmp)
#    print(frequencies)
    return np.sort(frequencies[np.isfinite(frequencies)])

def find_mixed_l1_freq(delta_nu, pone, DPi1, eps_g, coupling, N):
    """
    Find individual mixed mode
    """

    def opt_func(nu):
        theta_p = (np.pi / delta_nu) * (nu - pone)
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