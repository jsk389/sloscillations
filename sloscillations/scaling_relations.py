# -*- coding: utf-8 -*-

import numpy as np 

##########################################################
###             FREQUENCIES                            ###
##########################################################

def delta_nu(numax, deterministic=True):
    """
    Use scaling relation from Mosser et al. (2012)
    """
    if deterministic:
        return 0.276 * numax ** 0.751

def epsilon_p(dnu, deterministic=True):
    """
    Use scaling relation from Corsaro et al. 2012 for epsilon
    """
    if deterministic:
        return 0.634 + 0.546 * np.log10(dnu)

def alpha(dnu, deterministic=True):
    """
    Eqn (6) of Vrard et al. (2015), originally from Mosser
    et al. (2013)
    """
    if deterministic:
        return 0.015 * dnu **-0.32

def d03_scaling(n, dnu, deterministic=True):
    """
    Scaling relation for d03 from Mosser et al. 2010 (Universal Pattern)
    d03 = 0.280 * dnu
    """
    if deterministic:
        return np.ones_like(n) * 0.280 * dnu
    else:
        return np.random.normal(0.280, 0.012, n) * dnu
            
def d02_scaling(n, dnu, deterministic=True):
    """
    Scaling relation for d02 from Corsaro et al. 2012
    d02 = (0.112 +/- 0.016) * dnu
    """   
    if deterministic:
        return np.ones_like(n) * 0.121 * dnu + 0.035
    else:
        return (np.random.normal(0.121, 0.003, n) * dnu + \
                 np.random.normal(0.035, 0.012))

def d01_scaling(n, dnu, deterministic=True):
    """
    Use scaling relation from Mosser et al. 2018
    d01 = dnu/2 + (0.109 +/- 0.012)
    """   
    if deterministic:
        #return np.ones_like(n) * -0.0553 + -0.036*np.log10(dnu)
        #return np.ones_like(n) * 0.109 * dnu
        #return (np.ones_like(n) * -0.056 + -0.002*np.log10(dnu))*dnu
        return -(np.ones_like(n) * 0.0553 - 0.036*np.log10(dnu))*dnu
    else:
        pass
        #return np.random.normal(0.109, 0.012, n) * dnu

###########################################################
###             AMPLITUDES AND LINEWIDTHS               ###
###########################################################

def denv_scaling(numax, evo_state='RGB'):
    """
    Compute the width of the oscillation envelope as a function of
    evolutionary state
    """
    if evo_state == 'RGB':
        return 0.986 * numax ** 0.694
    elif evo_state == 'RC':
        return 1.00 * numax ** 0.766
    else:
        pass   

def Henv_scaling(numax, evo_state='RGB', deterministic=True):
    """
    Compute the height of the oscillation envelope as a function of 
    evolutionary state
    """
    if evo_state == 'RGB':
        return 2.37e7 * numax ** -2.31#1.37e7 * numax ** -2.201
    elif evo_state == 'RC':
        return 31.0e7 * numax ** -3.12
    else:
        pass

def amax_scaling(Henv, numax, dnu, vis_tot=None, mission='Kepler'):
    """
    Compute radial mode amplitude at numax
    """
    # Set up relative visibility for given mission
    if (vis_tot is None) and (mission == 'Kepler'):
        vis_tot = 3.16
        factor = 1
    elif (vis_tot is None) and (mission == 'TESS'):
        vis_tot = 2.94
        # Accounting for redder passband of TESS
        factor = 0.85
    else:
        pass

    return factor * np.sqrt(Henv * dnu / vis_tot)        
    

def gamma_0_scaling(Teff=4800, deterministic=True):
    """
    Use scaling relation from Corsaro et al. 2012
    gamma = gamma0*exp[(Teff-5777)/T0]
    #with updated parameters from Lund et al. (2016)
    """
    if deterministic:
        gamma0 = 1.39 #1.02 #1.39
        T0 = 604 #436 #601
    else:
        gamma0 = np.random.normal(1.02, 0.07) #1.39, 0.1)
        T0 = np.random.normal(436, 24) #601, 3)
    return gamma0*np.exp((Teff-5777)/T0) 

def gamma_scaling(numax, deterministic=True):
    """
    Use the linewidth relation from Appourchaux et al. (2014a) which
    has been modified by Lund et al. (2016) and using values from
    Lund et al. (2016)
    """
    alpha = 2.95 * (numax/3090) - 0.39
    gamma_a = 3.32 * (numax/3090) + 3.08
    gamma_dip = -0.47 * (numax/3090) + 0.62
    W_dip = 4637 * (numax / 3090) - 141
    nu_dip = 2984 * (numax / 3090) + 60
    fwhm_dip = 1253 * (numax / 3090) - 85

    return alpha, gamma_a, gamma_dip, W_dip, nu_dip, fwhm_dip

###########################################################
###             MIXED MODE PARAMETERS                   ###
###########################################################

def DPi1_scaling(dnu, evo_state):
    """
    Compute DPi1 given evolutionary state and delta nu
    """
    if evo_state == 'RGB':
        return 1.95*dnu + 56.2
    else:
        return 300.0

def coupling_scaling(evo_state):
    """
    Scaling for coupling given evolutionary state
    """
    if evo_state == 'RGB':
        return 0.15
    else:
        return 0.3