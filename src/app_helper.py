import numpy as np 

from scipy import interpolate
from scipy.optimize import brentq


import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as plt

import functools
from scipy.integrate import quad
import streamlit as st

import itertools

#https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]#, idx

def _sLor(f, A, b, c):
    return A / (1 + (f / b) ** c)

def _sinc(x):
    return np.sinc(x/np.pi)

def bgModel(nu, theta, nuNyq, individual=True):
    """
    Background model value at a given frequency 'nu'
    """
    Pn, A1, b1, A2, b2, A3, b3, Pg, numax, sigmaEnv = theta
    sc = _sinc(np.pi * nu / (2 * nuNyq)) ** 2
    comp1 = sc * _sLor(nu, A1, b1, 4)
    comp2 = sc * _sLor(nu, A2, b2, 4)
    comp3 = sc * _sLor(nu, A3, b3, 4)
    gauss = sc * Pg * np.exp(-((nu - numax) ** 2) / (2 * sigmaEnv ** 2))
    if individual == True:
        return comp1, comp2, comp3, gauss, Pn
    return comp1 + comp2 + comp3 + gauss + Pn

def lorentzian(f, amp, lw, freq):
    height = 2*amp**2/(np.pi*2*lw)
    x = (2/lw) * (f - freq)
    return height / (1 + x**2)

def sinc_sq(f, amp, lw, freq):
    height = amp**2 / (np.pi * (f[1]-f[0]))
    return height*np.sinc((f-freq)/(f[1]-f[0]))**2

def model(f, row):
    if np.isfinite(row['linewidth']):
        return lorentzian(f, row['amplitude'], row['linewidth'], row['frequency'])
    else:
        #print(row['linewidth'])
        return sinc_sq(f, row['amplitude'], row['linewidth'], row['frequency'])

def construct_MLEmodel(pds, peaks):
    fit_model02 = np.zeros(len(pds))
    fit_model1 = np.zeros(len(pds))
    for idx, i in peaks.loc[(peaks['l'] == 0) | (peaks['l'] == 2), ].iterrows():
        #plt.plot(pds['frequency'], model(pds['frequency'], i))
        #plt.show()
        #fit_model02 += model(pds['frequency'], i)
        fit_model02 += model(pds['frequency'].values, i)
        
    for idx, i in peaks.loc[(peaks['l'] != 0) & (peaks['l'] != 2), ].iterrows():
        #plt.plot(pds['frequency'], model(pds['frequency'], i))
        #plt.show()
        fit_model1 += model(pds['frequency'].values, i)
    return fit_model02, fit_model1

def echelle(freq, power, dnu, fmin=0., fmax=None, offset=0.0):
    # This is a slightly modified version of Dan Hey's fantastic echelle code
    # https://github.com/danhey/echelle/blob/master/echelle/echelle.py
    """Calculates the echelle diagram. Use this function if you want to do
    some more custom plotting.
    
    Parameters
    ----------
    freq : array-like
        Frequency values
    power : array-like
        Power values for every frequency
    dnu : float
        Value of deltanu
    fmin : float, optional
        Minimum frequency to calculate the echelle at, by default 0.
    fmax : float, optional
        Maximum frequency to calculate the echelle at. If none is supplied, 
        will default to the maximum frequency passed in `freq`, by default None
    offset : float, optional
        An offset to apply to the echelle diagram, by default 0.0
    
    Returns
    -------
    array-like
        The x, y, and z values of the echelle diagram.
    """
    if fmax is None:
        fmax = freq[-1]

    fmin = fmin - offset
    fmax = fmax - offset
    freq = freq - offset

    if fmin <= 0.0:
        fmin = 0.0
    else:
        fmin = fmin - (fmin % dnu)

    # trim data
    index = (freq>=fmin) & (freq<=fmax)
    trimx = freq[index]

    samplinginterval = np.median(trimx[1:-1] - trimx[0:-2])# * 0.1
    xp = np.arange(fmin,fmax+dnu,samplinginterval)
    yp = np.interp(xp, freq, power)

    n_stack = int(np.ceil((fmax-fmin)/dnu))
    n_element = int(np.ceil(dnu/samplinginterval))

    morerow = 2
    arr = np.arange(1,n_stack) * dnu
    arr2 = np.array([arr,arr])
    yn = np.reshape(arr2,len(arr)*2,order="F")
    yn = np.insert(yn,0,0.0)
    yn = np.append(yn,n_stack*dnu) + fmin + offset

    xn = np.arange(1,n_element+1)/n_element * dnu
    #print(yn)
    z = np.zeros([n_stack*morerow,n_element])
    for i in range(n_stack):
        for j in range(i*morerow,(i+1)*morerow):
            z[j,:] = yp[n_element*(i):n_element*(i+1)]
    return xn, yn, z

def l0_from_UP(N, eps_p, alpha_, n_max, DeltaNu):
    # Theoretical radial mode frequencies from Universal Pattern
    return ((N + eps_p + (alpha_/2) * (N - n_max)**2) * DeltaNu)

def l1_nominal_p_freqs(freqs_zero, deltanu, d1=None):
    if d1 is not None:
        return freqs_zero + deltanu/2 + d1
    else:
        d1 = 0.0553 - 0.036*np.log(deltanu)
        return freqs_zero + deltanu/2 + d1

def find_mixed_l1_freq(DeltaNu, pone, DPi1, eps_g, coupling, N):

    def opt_func(nu):
        theta_p = (np.pi / DeltaNu) * (nu - pone)
        theta_g = np.pi * (1 / (DPi1*1e-6*nu) - eps_g)
        return theta_p - np.arctan(coupling * np.tan(theta_g))
    
#    lower_bound = 1 / (DPi1*1e-6 * (N + 1/2 + eps_g)) + np.finfo(float).eps * 1e4
#    upper_bound = 1 / (DPi1*1e-6 * (N - 1 + 1/2 + eps_g)) - np.finfo(float).eps * 1e4

    #st.write(np.finfo(float).eps * 1e4)
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
        return np.nan

def find_mixed_l1_freqs(DeltaNu, nu_p, DPi1, eps_g, coupling):
    #l1_freqs = []
    #for i in range(len(freq_zero)):
    #    l1_freqs.append(find_mixed_l1_freq(freq, delta_nu, freq_zero[i], 
    #                                                      period_spacing, 
    #                                                      epsilon_g, coupling))
    #return np.array(l1_freqs)

    nmin = np.ceil(1 / (DPi1*1e-6 * (nu_p + (DeltaNu/2))) - (1/2) - eps_g)
    nmax = np.ceil(1 / (DPi1*1e-6 * (nu_p - (DeltaNu/2))) - (1/2) - eps_g)
    #nmin -= 10
    nmax += 2
    #st.write(nmin, nmax)
    frequencies = []
    for i in np.arange(nmin, nmax, 1):
        tmp = find_mixed_l1_freq(DeltaNu, nu_p, DPi1, eps_g, coupling, i)
        frequencies = np.append(frequencies, tmp)
#    print(frequencies)
    return np.sort(frequencies[np.isfinite(frequencies)])

def find_mixed_l1_freqs_alt(DeltaNu, nu_p, DPi1, eps_g, coupling):
    # Calculate the mixed mode frequencies ...
    nu = np.arange(nu_p - DeltaNu, nu_p + 1 * DeltaNu, 1e-5)
    nu *= 1e-6
    lhs = np.pi * (nu - (nu_p * 1e-6)) / (DeltaNu*1e-6)
    rhs = np.arctan(coupling * np.tan(np.pi/(DPi1 * nu) - eps_g))
    mixed1 = np.zeros(100)
    counter = 0
    for i in np.arange(0, nu.size-1):
        if lhs[i] - rhs[i] < 0 and lhs[i+1] - rhs[i+1] > 0:
            mixed1[counter] = nu[i]
            counter += 1
    mixed1 = mixed1[:counter]
    # add in the rotational splitting ...
    mixed1 *= 1e6
    return mixed1

def all_mixed_l1_freqs(DeltaNu, nu_p, DPi1, eps_g, coupling):
    #l1_freqs = []
    #for i in range(len(freq_zero)):
    #    l1_freqs.append(find_mixed_l1_freq(freq, delta_nu, freq_zero[i], 
    #                                                      period_spacing, 
    #                                                      epsilon_g, coupling))
    #return np.array(l1_freqs)

    l1_freqs = []
    for i in range(len(nu_p)):
        tmp = find_mixed_l1_freqs(DeltaNu, nu_p[i], 
                                  DPi1, eps_g, coupling)
        l1_freqs.append(tmp)

    return np.array(list(itertools.chain(*l1_freqs)))

def zeta_Mosser(freq, nu_p, DeltaNu, DPi1, coupling, eps_g):
    # Mosser et al. 2018: A&A, AA/2018/32777
    N = DeltaNu / (DPi1*1e-6 * freq**2)

    theta_p = (np.pi / DeltaNu) * (freq - nu_p)
    b = 1 + (coupling / N) / ((coupling**2) * np.cos(theta_p)**2 + np.sin(theta_p)**2)
    res = 1/b

    return res

def zeta_Deheuvels(freq, nu_p, DeltaNu, DPi1, coupling, eps_g):
    # Deheuvels et al. (2015) <http://dx.doi.org/10.1051/0004-6361/201526449>
    a1 = np.cos(np.pi * ((1 / (freq * DPi1*1e-6)) - eps_g))**2
    a2 = np.cos(np.pi * ((freq - nu_p) / DeltaNu))**2
    a3 = (freq**2 * DPi1*1e-6) / (coupling * DeltaNu)
    b = 1 + a1 * a3 / a2
    return 1/b

def calc_zeta(freq_zero, nu_p, DeltaNu, DPi1, coupling, eps_g):
    zeta = []
    #l1_freqs = all_mixed_l1_freqs(DeltaNu, nu_p,
    #                              DPi1, eps_g, coupling)
    l1_freqs = []
    for i in range(len(nu_p)):
        tmp_l1_freqs = find_mixed_l1_freqs(DeltaNu, nu_p[i], DPi1, eps_g, coupling)
        zeta = np.append(zeta, zeta_Mosser(tmp_l1_freqs, nu_p[i], DeltaNu, DPi1,
                                coupling, eps_g))
        l1_freqs = np.append(l1_freqs, tmp_l1_freqs)
    return l1_freqs, zeta

#@st.cache
def zeta_interp(freq, freq_zero, nu_p, DeltaNu, 
                DPi1, coupling, eps_g,
                numDPi1, DPi1_range, return_func=True):
    # Interpolate zeta function
    l1_freqs = []
    zeta = []
    DPi1_vals = np.linspace(DPi1_range[0]*DPi1, DPi1_range[1]*DPi1, numDPi1)

    for i in range(len(DPi1_vals)):
        #print(DPi1_vals[i])
        tmp_l1_freqs, tmp_zeta = calc_zeta(freq_zero, nu_p, DeltaNu,
                                           DPi1_vals[i], coupling, eps_g)
        l1_freqs = np.append(l1_freqs, tmp_l1_freqs)
        zeta = np.append(zeta, tmp_zeta)

    l1_freqs = l1_freqs.ravel()
    zeta = zeta.ravel()

    idx = np.argsort(l1_freqs)
    l1_freqs = l1_freqs[idx]
    zeta = zeta[idx]

    zeta_fun = interpolate.interp1d(l1_freqs, zeta)
    #interp_zeta = np.interp(freq, l1_freqs, zeta)
    interp_zeta = zeta_fun(freq)
    if return_func:
        return l1_freqs, interp_zeta, zeta_fun
    return l1_freqs, interp_zeta

def stretched_pds(pds, freq_zero, DeltaNu, 
                  DPi1, coupling, eps_g, 
                  numDPi1=100, DPi1_range=[0.99, 1.01], oversample=1):
    # Compute frequency bin-width
    bw = pds.frequency[1]-pds.frequency[0]
    # Compute interpolated zeta across entire frequency range
    nom_l1_freqs = l1_nominal_p_freqs(freq_zero, DeltaNu)
    l1_freqs, zz, zeta_fun = zeta_interp(pds.frequency.values, freq_zero, 
                                    nom_l1_freqs, DeltaNu,
                                    DPi1, coupling, eps_g, numDPi1, DPi1_range)
    # Compute dtau
    if oversample > 1:
        new_freq = np.arange(pds.frequency.min(), pds.frequency.max(), bw/oversample)
        #dtau = 1 / (zz*(pds.frequency.values)**2) * 1e6
        dtau = 1 / (zeta_fun(new_freq)*(new_freq)**2) * 1e6
    else:
        new_freq = pds.frequency.values
        dtau = 1 / (zeta_fun(new_freq)*(new_freq)**2) * 1e6
   
    dtau[np.isnan(dtau)] = 0
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

    tau = tau - tau_shift + DPi1

    return new_freq, tau, zeta_fun#, pds.power

def peaks_stretched_period(frequency, pds_frequency, tau):
    assert len(tau) == len(pds_frequency)
    return np.interp(frequency, pds_frequency, tau)

def l1_rot_from_zeta(pds, nu_0, nu_m, drot, zeta_fun):
    
    # Upper and lower limits for integration
    # Minimum value of nu_0 and nu_m or maximum value
    lower_limit = nu_0 if nu_0 < nu_m else nu_m
    upper_limit = nu_0 if nu_0 > nu_m else nu_m

    # Integrate zeta over that range
    res = quad(zeta_fun, lower_limit, upper_limit)
    int_zeta = res[0] / (nu_m - nu_0)
    return nu_0 + drot * int_zeta

def l1_rot_from_zeta_iter(pds, nu_0, nu_m, drot, zeta_fun, tol, max_iters=50, curr_iter=1):
    # Compute rotational splitting iteratively
    # If no rotational splitting then return nu_m, or nu_0?
    #st.write(curr_iter, max_iters)
    #st.write(drot)
    if drot == 0:
        return nu_m
    if curr_iter >= max_iters:
        st.write("Maximum number of iterations reached without convergence")
        return nu_m
    
    nu_m_new = l1_rot_from_zeta(pds, nu_0, nu_m, drot, zeta_fun)

    #st.write(abs(nu_m_new - nu_m))
    if abs(nu_m_new - nu_m) < tol:
        return nu_m_new
    else:
        return l1_rot_from_zeta_iter(pds, nu_0, nu_m_new, drot, zeta_fun,
                                     tol, max_iters, curr_iter+1)

def l1_theoretical_rot_M(pds, l1_m0_freqs, drot, zeta_fun, max_iters=50, tol=1e-4):

    l_mp1_freqs = []
    l_mn1_freqs = []
    for i in range(len(l1_m0_freqs)):
        tmp_p1 = l1_rot_from_zeta_iter(pds, l1_m0_freqs[i], l1_m0_freqs[i]+drot,
                              drot, zeta_fun, tol, max_iters)
        tmp_n1 = l1_rot_from_zeta_iter(pds, l1_m0_freqs[i], l1_m0_freqs[i]-drot,
                              drot, zeta_fun, tol, max_iters)
        l_mp1_freqs = np.append(l_mp1_freqs, tmp_p1)
        l_mn1_freqs = np.append(l_mn1_freqs, tmp_n1)

        #st.write(l1_m0_freqs[i], tmp_p1, tmp_p1-l1_m0_freqs[i])
        #st.write(l1_m0_freqs[i], tmp_n1, l1_m0_freqs[i]-tmp_n1)
        #sys.exit()
    return l_mp1_freqs, l_mn1_freqs

#https://stackoverflow.com/questions/29166353/how-do-you-add-error-bars-to-bokeh-plots-in-python
def errorbar(fig, x, y, xerr=None, yerr=None, color='red', 
             point_kwargs={}, error_kwargs={}):

  fig.circle(x, y, color=color, **point_kwargs)

  if np.all(xerr != None):
      x_err_x = []
      x_err_y = []
      for px, py, err in zip(x, y, xerr):
          x_err_x.append((px - err, px + err))
          x_err_y.append((py, py))
      fig.multi_line(x_err_x, x_err_y, color=color, **error_kwargs)

  if np.all(yerr != None):
      y_err_x = []
      y_err_y = []
      for px, py, err in zip(x, y, yerr):
          y_err_x.append((px, px))
          y_err_y.append((py - err, py + err))
      fig.multi_line(y_err_x, y_err_y, color=color, **error_kwargs)