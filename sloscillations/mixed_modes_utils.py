# -*- coding: utf-8 -*-

import itertools
import numpy as np 

from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import brentq

def all_mixed_l1_freqs(delta_nu, nu_p, DPi1, eps_g, coupling, calc_zeta=True):

    l1_freqs = []
    zeta = []
    for i in range(len(nu_p)):
        tmp = find_mixed_l1_freqs(delta_nu, nu_p[i], 
                                DPi1, eps_g, coupling)
        if calc_zeta == True:
            tmp_zeta = calculate_zeta(tmp, nu_p[i], delta_nu, DPi1, coupling, eps_g)
            zeta.append(tmp_zeta)
        l1_freqs.append(tmp)

    if calc_zeta:
        return np.array(list(itertools.chain(*l1_freqs))), np.array(list(itertools.chain(*zeta)))
    else:
        return np.array(list(itertools.chain(*l1_freqs)))

def find_mixed_l1_freqs(delta_nu, nu_p, DPi1, eps_g, coupling):
    """
    Find all mixed modes in a given radial order
    """

    nmin = np.ceil(1 / (DPi1*1e-6 * (nu_p + (delta_nu/2))) - (1/2) - eps_g)
    nmax = np.ceil(1 / (DPi1*1e-6 * (nu_p - (delta_nu/2))) - (1/2) - eps_g)
    nmax += 2

    frequencies = []
    for i in np.arange(nmin, nmax, 1):
        tmp = find_mixed_l1_freq(delta_nu, nu_p, DPi1, eps_g, coupling, i)
        frequencies = np.append(frequencies, tmp)

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

    try:
        return brentq(opt_func, lower_bound, upper_bound)
    except:
        return np.nan

def l1_rot_from_zeta(nu_0, nu_m, drot, zeta_fun):
    """
    Find rotational splitting 
    """
    
    # Upper and lower limits for integration
    # Minimum value of nu_0 and nu_m or maximum value
    lower_limit = nu_0 if nu_0 < nu_m else nu_m
    upper_limit = nu_0 if nu_0 > nu_m else nu_m

    # Integrate zeta over that range
    res = quad(zeta_fun, lower_limit, upper_limit)
    int_zeta = res[0] / (nu_m - nu_0)
    return nu_0 + drot * int_zeta, int_zeta

def l1_rot_from_zeta_iter(nu_0, nu_m, drot, zeta_fun, tol, max_iters=50, curr_iter=1):
    # Compute rotational splitting iteratively
    # If no rotational splitting then return nu_m, or nu_0?

    #if drot == 0:
    #    return nu_0, np.nan
    if curr_iter >= max_iters:
        print("Maximum number of iterations reached without convergence")
        return nu_m, np.nan
    
    nu_m_new, int_zeta = l1_rot_from_zeta(nu_0, nu_m, drot, zeta_fun)

    if abs(nu_m_new - nu_m) < tol:
        return nu_m_new, int_zeta
    else:
        return l1_rot_from_zeta_iter(nu_0, nu_m_new, drot, zeta_fun,
                                     tol, max_iters, curr_iter+1)

def l1_theoretical_rot_M(l1_m0_freqs, drot, zeta_fun, max_iters=50, tol=1e-4):

    l_mp1_freqs = []
    int_zeta_p = []
    l_mn1_freqs = []
    int_zeta_n = []
    for i in range(len(l1_m0_freqs)):
        tmp_p1, tmp_iz_p = l1_rot_from_zeta_iter(l1_m0_freqs[i], l1_m0_freqs[i]+drot,
                              drot, zeta_fun, tol, max_iters)
        tmp_n1, tmp_iz_n = l1_rot_from_zeta_iter(l1_m0_freqs[i], l1_m0_freqs[i]-drot,
                              drot, zeta_fun, tol, max_iters)
        l_mp1_freqs = np.append(l_mp1_freqs, tmp_p1)
        int_zeta_p = np.append(int_zeta_p, tmp_iz_p)
        l_mn1_freqs = np.append(l_mn1_freqs, tmp_n1)
        int_zeta_n = np.append(int_zeta_n, tmp_iz_n)

        #st.write(l1_m0_freqs[i], tmp_p1, tmp_p1-l1_m0_freqs[i])
        #st.write(l1_m0_freqs[i], tmp_n1, l1_m0_freqs[i]-tmp_n1)
        #sys.exit()
    return l_mp1_freqs, l_mn1_freqs, int_zeta_p, int_zeta_n

def calculate_zeta(freq, nu_p, DeltaNu, DPi1, coupling, eps_g):
    # Deheuvels et al. (2015) <http://dx.doi.org/10.1051/0004-6361/201526449>
    a1 = np.cos(np.pi * ((1 / (freq * DPi1*1e-6)) - eps_g))**2
    a2 = np.cos(np.pi * ((freq - nu_p) / DeltaNu))**2
    a3 = (freq**2 * DPi1*1e-6) / (coupling * DeltaNu)
    b = 1 + a1 * a3 / a2
    return 1/b
    
def zeta_interp(freq, nu_p, delta_nu, 
                DPi1, coupling, eps_g,
                numDPi1=100, DPi1_range=[0.99, 1.01]):
    # Interpolate zeta function
    l1_freqs = []
    zeta = []
    DPi1_vals = np.linspace(DPi1_range[0]*DPi1, DPi1_range[1]*DPi1, numDPi1)

    for i in range(len(DPi1_vals)):
        #print(DPi1_vals[i])
        tmp_l1_freqs, tmp_zeta = all_mixed_l1_freqs(delta_nu, nu_p, DPi1_vals[i], eps_g, coupling, calc_zeta=True)

        l1_freqs = np.append(l1_freqs, tmp_l1_freqs)
        zeta = np.append(zeta, tmp_zeta)

    l1_freqs = l1_freqs.ravel()
    zeta = zeta.ravel()

    idx = np.argsort(l1_freqs)
    l1_freqs = l1_freqs[idx]
    zeta = zeta[idx]


    zeta_fun = interpolate.interp1d(l1_freqs, zeta)
    return zeta_fun