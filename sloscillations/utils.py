
# -*- coding: utf-8 -*-
import astropy.units as u
import celerite
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

from celerite import terms

from . import scaling_relations

def compute_mixed_heights(amplitude, linewidth, T, zeta):
    """
    Compute the mixed mode height using the expression from eqn 6.148 of 
    Basu & Chaplin (2017)
    """
    return ((2 * amplitude**2 * T) / (np.pi * T * linewidth*1e-6 + (2/(1-zeta)))) * 1e-6

def compute_model(frequency, freq, lwd, height):
    x = (2/lwd) * (frequency - freq)
    return height / (1 + x**2)

def compute_heights(amplitude, linewidth, T):
    """
    Compute the mode height from Fletcher et al. (2006) 
    (http://articles.adsabs.harvard.edu/pdf/2006MNRAS.371..935F)
    """
    return ((2 * amplitude**2 * T) / (np.pi * T * linewidth*1e-6 + 2)) * 1e-6

def compute_linewidths(freqs, numax, Teff=4800):
    """
    Compute mode linewidths using eqn 4.7 from Lund et al. (2016)
    """
    #alpha, gamma_a, gamma_dip, W_dip, nu_dip, fwhm_dip = scaling_relations.gamma_scaling(numax)
    #print(alpha, gamma_a, gamma_dip, W_dip, nu_dip)
    #print(freqs, numax)
    #term1 = alpha * np.log(freqs / numax) + np.log(gamma_a)
    #term2 = np.log(gamma_dip) / (1 + ((2*np.log(freqs/nu_dip))/np.log(W_dip/numax))**2)

    #lnGamma = term1 + term2
    #return np.exp(lnGamma)
    return np.ones_like(freqs) * scaling_relations.gamma_0_scaling(Teff)

def calculateS0(a, b):
    """
    Compute S0 from granulation amplitude a and timescale b according to
    equation A4 of Pereira et al. (2019)
    """
    omega = 2*np.pi*b# * 1e6
    return (a**2 / omega)*np.sqrt(2)

def gran_backg(f, a, b):
    """
    Compupte the granulation background model according to
    Kallinger et al. (2014)
    """
    model = np.zeros(len(f))
    for i in range(len(a)):
        height = ((2.0 * np.sqrt(2))/np.pi) * a[i]**2/b[i]
        model += height / (1 + (f*1e6/b[i])**4)
    return model

def compute_background(t, amp, freqs, white=0):
    """
    Compute granulation background using Gaussian process
    """

    if white == 0:
        white = 1e-6

    kernel = terms.JitterTerm(log_sigma = np.log(white))

    S0 = calculateS0(amp, freqs)
    print(f"S0: {S0}")
    for i in range(len(amp)):
        kernel += terms.SHOTerm(log_S0=np.log(S0[i]),
                                log_Q=np.log(1/np.sqrt(2)),
                                log_omega0=np.log(2*np.pi*freqs[i]))
    gp = celerite.GP(kernel)
    return kernel, gp, S0

def compute_backg_model(f, S0, freqs):
    """
    Compute background model in power spectrum using equation A8 of
    Pereira et al. (2019)
    """
    omega0 = freqs
    omega = 1e6
    model = np.zeros(len(f))
    for i in range(len(S0)):
        model += (4*(S0[i]) / ((omega/omega0[i])**4 + 1))
    return model
