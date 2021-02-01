# -*- coding: utf-8 -*-

import itertools
import numpy as np 
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.integrate import quad



def all_mixed_l1_freqs(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling, return_order=True, method='Mosser2018_update'):

    l1_freqs = []
    l1_g_freqs = []
    order = []
    N_g = []

    if method == "Mosser2018_update":
        search_function = find_mixed_l1_freqs_Mosser2018_update
    else:
        sys.exit("Other methods not yet implemented")

    for i in range(len(nu_p)):

        if nu_p[i] > nu_zero[-1]:
            radial = np.array([nu_zero[-1], nu_zero[-1] + delta_nu[i]])
        else:
            radial = np.array([nu_zero[i], nu_zero[i+1]])
            
        tmp, tmp_g, tmp_ng = search_function(delta_nu[i], radial, nu_p[i], 
                                         DPi1, eps_g, coupling)
        order.append([i]*len(tmp))
        l1_freqs.append(tmp)
        l1_g_freqs.append(tmp_g)
        N_g.append(tmp_ng)

    if return_order:
        return np.array(list(itertools.chain(*l1_freqs))), \
               np.array(list(itertools.chain(*order))), \
               np.array(list(itertools.chain(*l1_g_freqs))), \
               np.array(list(itertools.chain(*N_g)))
    else:
        return np.array(list(itertools.chain(*l1_freqs)))

def find_mixed_l1_freqs_Mosser2015(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling):
    """
    Helper function for Mosser2015 method of finding mixed mode frequencies
    """
    nmin = np.floor(1 / (DPi1*1e-6 * nu_zero[1]) - eps_g)
    nmax = np.floor(1 / (DPi1*1e-6 * nu_zero[0]) - eps_g)

    N_modes = (delta_nu * 1e-6) / (DPi1 * (nu_p*1e-6)**2)

    N = np.arange(nmin, nmax + 2, 1)

    frequencies, g_mode_freqs, N_g = find_mixed_l1_freq_Mosser2015_(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling, N)

    idx = np.argsort(frequencies[np.isfinite(frequencies)])
    return frequencies[np.isfinite(frequencies)][idx], g_mode_freqs[np.isfinite(frequencies)][idx], N_g[np.isfinite(frequencies)][idx]

def find_mixed_l1_freqs_Mosser2018(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling):
    """
    Helper function for Mosser2018 method of finding mixed mode frequencies
    """
    nmin = np.floor(1 / (DPi1*1e-6 * nu_zero[1]) - eps_g)
    nmax = np.floor(1 / (DPi1*1e-6 * nu_zero[0]) - eps_g)

    N_modes = (delta_nu * 1e-6) / (DPi1 * (nu_p*1e-6)**2)

    N = np.arange(nmin, nmax + 2, 1)

    frequencies, g_mode_freqs, N_g = find_mixed_l1_freq_Mosser2018_(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling, N)

    idx = np.argsort(frequencies[np.isfinite(frequencies)])
    return frequencies[np.isfinite(frequencies)][idx], g_mode_freqs[np.isfinite(frequencies)][idx], N_g[np.isfinite(frequencies)][idx]

def find_mixed_l1_freqs_oldMosser2018(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling):
    """
    Helper function for old Mosser2018 method of finding mixed mode frequencies
    """
    nmin = np.floor(1 / (DPi1*1e-6 * nu_zero[1]) - eps_g)
    nmax = np.floor(1 / (DPi1*1e-6 * nu_zero[0]) - eps_g)

    N_modes = (delta_nu * 1e-6) / (DPi1 * (nu_p*1e-6)**2)

    N = np.arange(nmin, nmax + 2, 1)

    frequencies = []
    g_mode_freqs = []
    N_g = []
    for i in np.arange(nmin, nmax, 1):
        tmp, tmp_g, tmp_ng = find_mixed_l1_freq_Mosser2018_(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling, i)
        frequencies = np.append(frequencies, tmp)
        g_mode_freqs = np.append(g_mode_freqs, tmp_g)
        N_g = np.append(N_g, tmp_ng)
    return np.sort(frequencies[np.isfinite(frequencies)]), np.sort(g_mode_freqs[np.isfinite(g_mode_freqs)]), np.sort(N_g[np.isfinite(N_g)])

def find_mixed_l1_freqs_Mosser2018_update(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling):
    """
    Helper function for our update to Mosser 2018 method (addition of 1/2 in nmin and nmax e.g.)
    """

    nmin = np.floor(1 / (DPi1*1e-6 * nu_zero[1]) - (1/2) - eps_g)
    nmax = np.floor(1 / (DPi1*1e-6 * nu_zero[0]) - (1/2) - eps_g)

    N_modes = (delta_nu * 1e-6) / (DPi1 * (nu_p*1e-6)**2)

    N = np.arange(nmin, nmax + 2, 1)

    frequencies, g_mode_freqs, N_g = find_mixed_l1_freq_Mosser2018_update_(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling, N)

    idx = np.argsort(frequencies[np.isfinite(frequencies)])
    return frequencies[np.isfinite(frequencies)][idx], g_mode_freqs[np.isfinite(frequencies)][idx], N_g[np.isfinite(frequencies)][idx]

def find_mixed_l1_freqs_Hekker(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling):
    """
    Helper function for method explain in Hekker & JCD review paper
    """

    nmin = np.floor(1 / (DPi1*1e-6 * nu_zero[1]) - (1/2) - eps_g)
    nmax = np.floor(1 / (DPi1*1e-6 * nu_zero[0]) - (1/2) - eps_g)

    N_modes = (delta_nu * 1e-6) / (DPi1 * (nu_p*1e-6)**2)

    N = np.arange(nmin, nmax, 1)
    frequencies, g_mode_freqs, N_g = find_mixed_l1_freq_Hekker(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling, N)

    idx = np.argsort(frequencies[np.isfinite(frequencies)])
    return frequencies[np.isfinite(frequencies)][idx], g_mode_freqs[np.isfinite(frequencies)][idx], N_g[np.isfinite(frequencies)][idx]

def opt_funcM(nu, nu_g, pzero, pone, DPi1, coupling):
    theta_p = (np.pi / (pzero[1]-pzero[0])) * (nu - pone)
    theta_g = np.pi/DPi1 * 1e6 * (1/nu - 1/nu_g) + np.pi/2
    y = np.tan(theta_p) - coupling * np.tan(theta_g)
    return y  

def find_mixed_l1_freq_Mosser2018_update_(delta_nu, pzero, pone, DPi1, eps_g, coupling, N):
    """
    Find mixed modes using updated Mosser 2018 method.
    """



    nu_g = 1 / (DPi1*1e-6 * (N + 1/2 + eps_g))
    # Go +/- 1/2 * DPi1 away from g-mode period
    lower_bound = 1 / (DPi1*1e-6 * (N     + 1/2  + 1/2 + eps_g)) + 0.220446049250313e-16 * 1e4 #np.finfo(float).eps * 1e4
    upper_bound = 1 / (DPi1*1e-6 * (N - 1 + 1/2  + 1/2 + eps_g)) - 0.220446049250313e-16 * 1e4#np.finfo(float).eps * 1e4

    #print(lower_bound, upper_bound, nu_g)
    #f = np.linspace(pzero[0], pzero[1], 10000)#[1:-1]
    #dnu = np.diff(pzero)
    solns = []
    #solns = np.zeros(len(nu_g))
    for i in range(len(nu_g)):
        #print(upper_bound[i], lower_bound[i], pzero)
        if (upper_bound[i] > pzero[1]):
            #print("BEFORE UPP: ", upper_bound)
            upper_bound[i] = pzero[1]# + 0.05*dnu# - np.finfo(float).eps*1e4
            #print("AFTER UPP: ", upper_bound)
        elif (lower_bound[i] < pzero[0]):
            #print("BEFORE LOW: ", lower_bound)
            lower_bound[i] = pzero[0]# - 0.05*dnu# + np.finfo(float).eps*1e4
            #print("AFTER LOW: ", lower_bound)
        #print(upper_bound[i], lower_bound[i], pzero)
        if (upper_bound[i] < lower_bound[i]) or (lower_bound[i] > upper_bound[i]):
            #print("Bad boundary conditions")
            pass
        else:
            #print("Valid")
            ff = np.linspace(lower_bound[i], upper_bound[i], 1000)
            y = opt_funcM(ff, nu_g[i], pzero, pone, DPi1, coupling)
    
            

            idx = np.where(np.diff(np.sign(y)) > 0)[0]
            #if len(idx) == 0:
            #    soln = np.array([])
            #else:
            #    soln = ff[idx]
            
            if len(idx) > 0:
                solns = np.append(solns, ff[idx])

    #solns = np.stack(solns)
    #print(solns)

    theta_p = (np.pi / (pzero[1]-pzero[0])) * (solns - pone)
    # Approximate pure g-mode frequencies and radial orders
    #print(solns)
    g_period = 1/(solns*1e-6) - DPi1/np.pi * np.arctan2(np.tan(theta_p), coupling) 

    n_g = np.floor(g_period / DPi1 - eps_g - 1/2)
    return solns, 1e6/g_period, n_g

def find_mixed_l1_freq_(delta_nu, pzero, pone, DPi1, eps_g, coupling, N, method='Mosser2018_update'):
    """
    Find individual mixed mode
    """
    def opt_func(nu):
        theta_p = (np.pi / (pzero[1]-pzero[0])) * (nu - pone)
        #theta_g = np.pi * (1 / (DPi1*1e-6*nu) - eps_g)
        if method == 'Mosser2015':
            theta_g = np.pi * (1 / (DPi1*1e-6*nu) - eps_g)
        elif method == 'Mosser2018':
            theta_g = np.pi/DPi1 * 1e6 * (1/nu - 1/nu_g)
        elif method == 'Mosser2018_update':
            theta_g = np.pi/DPi1 * 1e6 * (1/nu - 1/nu_g) + np.pi/2
        y = np.tan(theta_p) - coupling * np.tan(theta_g)
        #y = theta_p - np.arctan2(coupling, 1/np.tan(theta_g))
        return y# - np.tan(theta_p)# - val + np.finfo(float).eps * 1e10

    def opt_funcM(nu, nu_g):
        theta_p = (np.pi / (pzero[1]-pzero[0])) * (nu - pone)
        if method == 'Mosser2018':
            theta_g = np.pi/DPi1 * 1e6 * (1/nu - 1/nu_g)
        elif method == 'Mosser2018_update':
            theta_g = np.pi/DPi1 * 1e6 * (1/nu - 1/nu_g) + np.pi/2
        y = np.tan(theta_p) - coupling * np.tan(theta_g)
        #y = theta_p - np.arctan2(coupling, 1/np.tan(theta_g))
        return y# - np.tan(theta_p)# - val + np.finfo(float).eps * 1e10

    if method == 'Mosser2015':
        nu_g = 1 / ((N + eps_g)*DPi1*1e-6)
        lower_bound = 1 / (DPi1*1e-6 * (N.max()     + 1/2 + eps_g)) + np.finfo(float).eps * 1e4
        upper_bound = 1 / (DPi1*1e-6 * (N.min() - 1 + 1/2 + eps_g)) - np.finfo(float).eps * 1e4
    elif method == 'Mosser2018':
        nu_g = 1 / (DPi1*1e-6 * (N + eps_g))
        lower_bound = 1 / (DPi1*1e-6 * (N     + 1/2 + eps_g)) + np.finfo(float).eps * 1e4
        upper_bound = 1 / (DPi1*1e-6 * (N - 1 + 1/2 + eps_g)) - np.finfo(float).eps * 1e4
    elif method == 'Mosser2018_update':
        nu_g = 1 / (DPi1*1e-6 * (N + 1/2 + eps_g))
        # Go +/- 1/2 * DPi1 away from g-mode period
        lower_bound = 1 / (DPi1*1e-6 * (N     + 1/2  + 1/2 + eps_g)) + np.finfo(float).eps * 1e4
        upper_bound = 1 / (DPi1*1e-6 * (N - 1 + 1/2  + 1/2 + eps_g)) - np.finfo(float).eps * 1e4

    if method != 'Mosser2015':
        #print(lower_bound, upper_bound, nu_g)
        f = np.linspace(pzero[0], pzero[1], 10000)#[1:-1]
        dnu = np.diff(pzero)
        solns = []
        for i in range(len(nu_g)):
            if (upper_bound[i] > pzero[1]):
                #print("BEFORE UPP: ", upper_bound)
                upper_bound[i] = pzero[1]# + 0.05*dnu# - np.finfo(float).eps*1e4
                #print("AFTER UPP: ", upper_bound)
            elif (lower_bound[i] < pzero[0]):
                #print("BEFORE LOW: ", lower_bound)
                lower_bound[i] = pzero[0]# - 0.05*dnu# + np.finfo(float).eps*1e4
                #print("AFTER LOW: ", lower_bound)
            #print(upper_bound[i], lower_bound[i], pzero)
            if (upper_bound[i] < lower_bound[i]) or (lower_bound[i] > upper_bound[i]):
                #print("Bad boundary conditions")
                pass
            else:
                ff = np.linspace(lower_bound[i], upper_bound[i], 1000)
                y = opt_funcM(ff, nu_g[i])
        
                idx = np.where(np.diff(np.sign(y)) > 0)[0]
                if len(idx) == 0:
                    soln = np.array([])
                else:
                    soln = ff[idx]
                

                solns = np.append(solns, soln)


        theta_p = (np.pi / (pzero[1]-pzero[0])) * (solns - pone)
        # Approximate pure g-mode frequencies and radial orders
        g_period = 1/(solns*1e-6) - DPi1/np.pi * np.arctan2(np.tan(theta_p), coupling) 
        if method == 'Mosser2018':
            n_g = np.floor(g_period / DPi1 - eps_g) 
        else:
            n_g = np.floor(g_period / DPi1 - eps_g - 1/2)
        return solns, 1e6/g_period, n_g
    
    if method == 'Mosser2015':

        upper_bound = upper_bound.max()
        lower_bound = lower_bound.min()
        if (upper_bound > pzero[1]):
            #print("BEFORE UPP: ", upper_bound)
            upper_bound = pzero[1]# - np.finfo(float).eps*1e4
            #print("AFTER UPP: ", upper_bound)
        elif (lower_bound < pzero[0]):
            #print("BEFORE LOW: ", lower_bound)
            lower_bound = pzero[0]# + np.finfo(float).eps*1e4
            #print("AFTER LOW: ", lower_bound)

        low = opt_func(lower_bound)
        upp = opt_func(upper_bound)

        if upper_bound < lower_bound:
            print("OH DEAR")
            return np.nan, np.nan, np.nan
        
        f = np.linspace(pzero[0], pzero[1], 10000)
        y = opt_func(f)
        
        idx = np.where(np.diff(np.sign(opt_func(f))) > 0)[0]
        if len(idx) == 0:
            soln = np.array([])
        else:
            soln = (f[idx] + f[idx+1])/2

        theta_p = (np.pi / (pzero[1]-pzero[0])) * (soln - pone)
        # Approximate pure g-mode frequencies and radial orders
        g_period = 1/(soln*1e-6) - DPi1/np.pi * np.arctan2(np.tan(theta_p), coupling) 

        n_g = (np.floor(g_period / DPi1) - eps_g) 
       
        if len(soln) < 1:
            return np.array([np.nan]), np.array([np.nan]), np.array([np.nan])
        else:
            return soln, 1e6/g_period, n_g


def find_mixed_l1_freqs(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling, method='Mosser2018_update'):
    """
    Find all mixed modes in a given radial order
    """
    

    if 'old' in method:
        frequencies = []
        g_mode_freqs = []
        N_g = []
        for i in np.arange(nmin, nmax, 1):
            tmp, tmp_g, tmp_ng = find_mixed_l1_freq(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling, i, method=method[3:])
            frequencies = np.append(frequencies, tmp)
            g_mode_freqs = np.append(g_mode_freqs, tmp_g)
            N_g = np.append(N_g, tmp_ng)
        return np.sort(frequencies[np.isfinite(frequencies)]), np.sort(g_mode_freqs[np.isfinite(g_mode_freqs)]), np.sort(N_g[np.isfinite(N_g)])
    elif 'Hekker2018' in method:
        N = np.arange(nmin, nmax, 1)
        frequencies, g_mode_freqs, N_g = find_mixed_l1_freqs_hekker(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling, N)
    else:
        frequencies, g_mode_freqs, N_g = find_mixed_l1_freq_(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling, N, method=method)
    #sys.exit()
    #for i in np.arange(nmin, nmax, 1):
    #    tmp, tmp_g, tmp_ng = find_mixed_l1_freq(delta_nu, nu_zero, nu_p, DPi1, eps_g, coupling, i, method=method)
    #    frequencies = np.append(frequencies, tmp)
    #    g_mode_freqs = np.append(g_mode_freqs, tmp_g)
    #    N_g = np.append(N_g, tmp_ng)
    #print("NUMBER OF MIXED MODES FOUND: ", len(frequencies[np.isfinite(frequencies)]))
    # 03/01/2021 - changing this bit as sorting incorrectly!
    #return np.sort(frequencies[np.isfinite(frequencies)]), np.sort(g_mode_freqs[np.isfinite(g_mode_freqs)]), np.sort(N_g[np.isfinite(N_g)])
    idx = np.argsort(frequencies[np.isfinite(frequencies)])
    return frequencies[np.isfinite(frequencies)][idx], g_mode_freqs[np.isfinite(frequencies)][idx], N_g[np.isfinite(frequencies)][idx]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_mixed_l1_freqs_hekker(delta_nu, pzero, pone, DPi1, eps_g, coupling, N):

    f = np.linspace(pzero[0], pzero[1], 10000)

    # Compute theta_p
    theta_p = np.pi/(pzero[1]-pzero[0]) * (f - pone)

    #theta_p[theta_p > np.pi/2] = np.pi - theta_p[theta_p > np.pi/2]
    #theta_p[theta_p < -np.pi/2] = -np.pi - theta_p[theta_p < -np.pi/2]


    # Compute phi
    phi = np.arctan2(coupling, np.tan(theta_p))
    #plt.plot(f, (theta_p), '.')
    #plt.show()
    #f = f[cond]

    #plt.plot(f, phi)
    #plt.show()

    # Compute psi
    psi = 1 / (DPi1*f*1e-6) + phi/np.pi - eps_g - 1/2
    # Compute integer radial orders
    k = np.arange(np.floor(psi.max()), np.floor(psi.min()), -1)
    # Find nearest integer psi value
    print(psi.min(), np.ceil(psi.min()), psi.max(), k.min(), k.max())
    vec_nearest = lambda x: find_nearest(psi, x)
    index = np.vectorize(vec_nearest)(k)
    ng = -psi[index]

    # Check ng is close enough to given integer
    mod = ng % 1
    mod[mod > 0.5] = 1 - mod[mod > 0.5]
    cond = mod < 1e-2

    #ng = np.floor(ng[cond])
    ng = np.round(ng[cond])
    #print(ng)


    # Compute period of mixed modes
    mixed_period = (-ng + eps_g + 1/2 - phi[index][cond]/np.pi) * DPi1

    #plt.plot(f, psi, '.')
    #for i in (1e6/mixed_period):
    #    plt.axvline(i, color='r', linestyle='--', alpha=0.5)
    #plt.axvline(pzero[0], color='r', alpha=0.5)
    #plt.axvline(pzero[1], color='r', alpha=0.5)
    #plt.show()

    # Compute mixed mode frequency
    mixed_nu = 1e6/mixed_period
    # Compute underlying g-mode periods
    g_period = (-ng + eps_g + 1/2) * DPi1
    #print(mixed_nu, 1e6/g_period)
    
    return mixed_nu, 1e6/g_period, ng



def find_mixed_l1_freq(delta_nu, pzero, pone, DPi1, eps_g, coupling, N, method='Mosser2015'):
    """
    Find individual mixed mode
    """

    def opt_func(nu, val=0):
        theta_p = (np.pi / delta_nu) * (nu - pone)
        #theta_g = np.pi * (1 / (DPi1*1e-6*nu) - eps_g)
        if method == 'Mosser2015':
            theta_g = np.pi * (1 / (DPi1*1e-6*nu) - eps_g)
        elif method == 'Mosser2018':
            theta_g = np.pi/DPi1 * 1e6 * (1/nu - 1/nu_g)
        elif method == 'Mosser2018_update':
            theta_g = np.pi/DPi1 * 1e6 * (1/nu - 1/nu_g) + np.pi/2
        y = np.tan(theta_p) - coupling * np.tan(theta_g)
        #y = theta_p - np.arctan2(coupling, 1/np.tan(theta_g))
        return y# - val + np.finfo(float).eps * 1e10


    if method == 'Mosser2015':
        nu_g = 1 / ((N + eps_g)*DPi1*1e-6)
        lower_bound = 1 / (DPi1*1e-6 * (N     + 1/2 + eps_g)) + np.finfo(float).eps * 1e4
        upper_bound = 1 / (DPi1*1e-6 * (N - 1 + 1/2 + eps_g)) - np.finfo(float).eps * 1e4
    elif method == 'Mosser2018':
        nu_g = 1 / (DPi1*1e-6 * (N + eps_g))
        lower_bound = 1 / (DPi1*1e-6 * (N     + 1/2 + eps_g)) + np.finfo(float).eps * 1e4
        upper_bound = 1 / (DPi1*1e-6 * (N - 1 + 1/2 + eps_g)) - np.finfo(float).eps * 1e4
    elif method == 'Mosser2018_update':
        nu_g = 1 / (DPi1*1e-6 * (N + 1/2 + eps_g))
        # Go +/- 1/2 * DPi1 away from g-mode period
        lower_bound = 1 / (DPi1*1e-6 * (N     + 1/2 + eps_g)) + np.finfo(float).eps * 1e4
        upper_bound = 1 / (DPi1*1e-6 * (N - 1 + 1/2 + eps_g)) - np.finfo(float).eps * 1e4
    
    if (upper_bound > pzero[1]):
        #print("BEFORE UPP: ", upper_bound)
        upper_bound = pzero[1]- np.finfo(float).eps*1e4
        #print("AFTER UPP: ", upper_bound)
    elif (lower_bound < pzero[0]):
        #print("BEFORE LOW: ", lower_bound)
        lower_bound = pzero[0] + np.finfo(float).eps*1e4
        #print("AFTER LOW: ", lower_bound)

    #low = opt_func(lower_bound)
    #upp = opt_func(upper_bound)

    if upper_bound < lower_bound:
        return np.nan, np.nan, np.nan

        
    #print(soln)
    #if len(soln) > 1
    """
    try: 
        brentq(opt_func, lower_bound, upper_bound)
    except:
        f = np.linspace(lower_bound, upper_bound, 1000)
        plt.plot(f, opt_func(f, val=0), '.')
        plt.axvline(lower_bound, linestyle='--', color='k')
        plt.axvline(upper_bound, linestyle='--', color='k')
        plt.axvline(nu_g, color='C1')
        plt.axvline(pzero[1], linestyle=':', color='r')
        plt.axvline(pone, linestyle='--', color='r')
        plt.axvline(pzero[0], color='r')
        plt.xlim(lower_bound -0.1, upper_bound + 0.1)
        #plt.xlim(pzero[0], pzero[1])
        plt.ylabel(r'$\tan\theta_{p}-q\tan\theta_{g}$', fontsize=18)
        plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
        #plt.axvline(brentq(opt_func, lower_bound, upper_bound), linestyle='--', color='b')
        #plt.title(soln)
        #for i in soln:
        #    plt.axvline(i, linestyle='--', color='b')
        plt.ylim(-10, 10)
        plt.show()
    """
    #print(opt_func(soln))
    #    soln = soln[abs(opt_func(soln)).argmin()]
   # res = root

    #print(lower_bound, pone, pone+delta_nu/2, upper_bound, nu_g)
    try:
        soln = brentq(opt_func, lower_bound, upper_bound)
        return soln, nu_g, N
    except:
        return np.nan, np.nan, np.nan


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

    return l_mp1_freqs, l_mn1_freqs, int_zeta_p, int_zeta_n

def calculate_zeta_(freq, nu_p, DeltaNu, DPi1, coupling, eps_g):
    # Deheuvels et al. (2015) <http://dx.doi.org/10.1051/0004-6361/201526449>
    a1 = np.cos(np.pi * ((1 / (freq * DPi1*1e-6)) - eps_g))**2
    a2 = np.cos(np.pi * ((freq - nu_p) / DeltaNu))**2
    a3 = (freq**2 * DPi1*1e-6) / (coupling * DeltaNu)
    b = 1 + a1 * a3 / a2
    return 1/b

def calculate_zeta(freq, nu_p, DeltaNu, DPi1, coupling, eps_g):
    theta_p = np.pi * (freq - nu_p) / DeltaNu
    N = (DeltaNu*1e-6) / ((nu_p*1e-6)**2 * DPi1)
    denominator = coupling**2*np.cos(theta_p)**2 + np.sin(theta_p)**2
    inv_zeta = 1 + coupling/N * 1/denominator
    return 1 / inv_zeta 

def _interpolated_zeta(frequency, delta_nu, nu_zero, nu_p, coupling, DPi1, plot=False):
    """
    Compute zeta for each radial order
    """
    zeta_max = np.zeros(len(nu_p))
    model = np.zeros_like(frequency)
    for i in range(len(nu_p)):
        # Only compute θₚ from one radial mode frequency to the next
        if i == len(nu_zero)-1:
            dnu = delta_nu[i] + (delta_nu[i] - delta_nu[i-1])
            cond = (frequency > nu_zero[i]) & (frequency < nu_zero[i] + dnu)
            
            # Estimate deltanu from radial mode frequencies
        else:
            cond = (frequency > nu_zero[i]) & (frequency < nu_zero[i+1])
            dnu = delta_nu[i]
        
        θₚ = np.pi*(frequency[cond] - nu_p[i])/dnu
        N = (dnu*1e-6)/(DPi1 * (nu_p[i]*1e-6)**2)
        frac = 1 + (coupling/N) * (coupling**2*np.cos(θₚ)**2 + np.sin(θₚ)**2)**-1
        zeta_max[i] = 1/(1 + coupling/N)
        if plot:
            plt.plot(frequency[cond], frac**-1 + (1 - zeta_max[i]))
        model[cond] = frac**-1 + (1 - zeta_max[i])
    return model, zeta_max

def interpolated_zeta(frequency, delta_nu, nu_zero, nu_p, coupling, DPi1, plot=False):
    """
    Compute the mixing function zeta for all frequency values

    Inputs:

        :params freq: Full frequency array
        :type   freq: numpy.ndarray

        :params delta_nu: Large frequency separation
        :type   delta_nu: float

        :params nu_zero: Array of radial mode frequencies
        :type   nu_zero: numpy.ndarray

        :params nu_p: Array of nominal p-mode frequencies
        :type   nu_p: numpy.ndarray

        :params coupling: Mode coupling
        :type   coupling: float

        :params DPi1: l=1 period spacing
        :type   DPi1: float

    """
   
   # N = (delta_nu*1e-6)/(DPi1 * (nu_p*1e-6)**2)
    
    #zeta_max = (1 + (coupling/N))**-1
    #zeta_min = (1 + (1/(coupling*N)))**-1

    # Compute zeta over each radial order
    model, zeta_max = _interpolated_zeta(frequency, delta_nu, nu_zero, nu_p, 
                         coupling, DPi1, plot=plot)

    # Interpolate zeta_max across all frequency
    # TODO: 27/12/2020 why nu_p? Should it be nu_zero?
    backg = np.interp(frequency, nu_p, zeta_max)
    
    # Add background back into zeta
    full_model = model - (1 - backg)
    
    return full_model #, zeta_max, zeta_min


def zeta_interp(freq, nu_zero, nu_p, delta_nu, 
                DPi1, coupling, eps_g,
                numDPi1=100, DPi1_range=[0.99, 1.01], return_full=False):
    # Interpolate zeta function
    l1_freqs = []
    zeta = []
    DPi1_vals = np.linspace(DPi1_range[0]*DPi1, DPi1_range[1]*DPi1, numDPi1)

    for i in range(len(DPi1_vals)):
        #print(DPi1_vals[i])
        tmp_l1_freqs, tmp_zeta = old_all_mixed_l1_freqs(delta_nu, nu_zero, nu_p, DPi1_vals[i], eps_g, coupling, return_order=False, calc_zeta=True)

        l1_freqs = np.append(l1_freqs, tmp_l1_freqs)
        zeta = np.append(zeta, tmp_zeta)
        
        #plt.scatter(tmp_l1_freqs, tmp_zeta, marker='.', label=DPi1_vals[i])
    #plt.legend(loc='best')
    #plt.show()

    l1_freqs = l1_freqs.ravel()
    zeta = zeta.ravel()

    idx = np.argsort(l1_freqs)
    l1_freqs = l1_freqs[idx]
    zeta = zeta[idx]



    zeta_fun = interpolate.interp1d(l1_freqs, zeta)

    if return_full:
        return l1_freqs, zeta, zeta_fun
    return zeta_fun

def stretched_pds(frequency, zeta, oversample=1):
    # Compute frequency bin-width
    bw = frequency[1]-frequency[0]

    # Compute dtau
    if oversample > 1:
        frequency = np.arange(frequency.min(), frequency.max(), bw/oversample)
        zeta = np.interp(frequency, frequency, zeta)
        dtau = 1 / (zeta*(frequency*1e-6)**2)
    else:
        dtau = 1 / (zeta*(frequency*1e-6)**2)
    

    #dtau[np.isnan(dtau)] = 0
    #dtau = dtau[np.isfinite(dtau)]
    # Compute tau
    tau = np.cumsum(dtau)*(bw/oversample * 1e-6)# + 13.8
    #print(tau)

#    tau -= shift
    #print(min(tau), frequency[tau == np.min(tau)])
    #tau -= np.min(tau)
 

    return frequency, tau, zeta #, shift

def compute_tau_shift(tau, DPi1):
    """
    Compute shift in tau to line up m=0 at tau mod DeltaPi1 = 0
    """
    # There is a problem when the value of tau % DPi is on the border of here there is wrapping
    # and so to check for that we compute both the median and the mean, if they vary by more than 5e-2
    # then we automatically set to 0 as an approximation.
    # Compute shift properly
    mean_shift = np.mean(((tau % DPi1) / DPi1) - 1)
    median_shift = np.median(((tau % DPi1) / DPi1) - 1)
    if np.abs(mean_shift - median_shift) < 5e-2:
        return mean_shift
    else:
        return 0.0
    #shift = np.mean(((tau % DPi1) / DPi1) - 1)
    #return shift


def peaks_stretched_period(frequency, pds_frequency, tau):
    assert len(tau) == len(pds_frequency)
    return np.interp(frequency, pds_frequency, tau)

def oldstretched_pds(frequency, nu_zero, nom_l1_freqs, DeltaNu, 
                  DPi1, coupling, eps_g, 
                  numDPi1=100, DPi1_range=[0.99, 1.01], oversample=1):
    # Compute frequency bin-width
    bw = frequency[1]-frequency[0]
    cond = (frequency > nom_l1_freqs.min()) & (frequency < nom_l1_freqs.max())
    frequency = frequency[cond]
    # Compute interpolated zeta across entire frequency range
    l1_freqs, zz, zeta_fun = zeta_interp(frequency, nu_zero,
                                         nom_l1_freqs, DeltaNu,
                                         DPi1, coupling, eps_g,
                                         numDPi1, DPi1_range,
                                         return_full=True)
    # Compute dtau
    if oversample > 1:
        new_freq = np.arange(frequency.min(), frequency.max(), bw/oversample)
        #dtau = 1 / (zz*(pds.frequency.values)**2) * 1e6
        dtau = 1 / (zeta_fun(new_freq)*(new_freq)**2) * 1e6
    else:
        new_freq = frequency
        dtau = 1 / (zeta_fun(new_freq)*(new_freq)**2) * 1e6
   
    #dtau[np.isnan(dtau)] = 0
    dtau = dtau[np.isfinite(dtau)]
    # Compute tau
    tau = -np.cumsum(dtau)*(bw/oversample)
    tau -= np.min(tau)

    #plt.plot(pds.frequency, dtau)
    #plt.plot(pds.frequency, tau)
    #plt.show()

    # Place tau into seconds
    #tau *= 1e6
    # Compute tau values of l1 frequencies to shift tau
    #l1_tau = np.interp(l1_freqs, pds.frequency.values, tau)
    l1_tau = np.interp(l1_freqs, new_freq, tau)
    l1_x = ((l1_tau + DPi1/2) % DPi1) - DPi1/2
    #l1_x = l1_tau % DPi1
    tau_shift = np.median(l1_x)
    #st.write(tau_shift)
    #l1_tau = l1_tau - tau_shift + DPi1
    # Compute l1 zeta
    #l1_zeta = np.interp(l1_freqs, pds.frequency.values, zz)

    #tau = tau - tau_shift + DPi1

    return new_freq, tau, zeta_fun