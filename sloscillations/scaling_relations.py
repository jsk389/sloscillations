# -*- coding: utf-8 -*-

import numpy as np 

##########################################################
###             FREQUENCIES                            ###
##########################################################

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
            
def d02_scaling(n, dnu, deterministic=True):
    """
    Scaling relation for d02 from Corsaro et al. 2012
    d02 = (0.112 +/- 0.016) * dnu
    """   
    if deterministic:
        return np.ones_like(n) * 0.121 * dnu + 0.035
    else:
        return np.random.normal(0.121, 0.003, n) * dnu + \
               np.random.normal(0.035, 0.012)

def d01_scaling(n, deterministic=True):
    """
    Use scaling relation from Corsaro et al. 2012
    d01 = dnu/2 + (0.109 +/- 0.012)
    """   
    if deterministic:
        return np.ones_like(n) * 0.109
    else:
        return np.random.normal(0.109, 0.012, n)

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
        return 1.37e7 * numax ** -2.201
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
    """
    if deterministic:
        gamma0 = 1.39
        T0 = 601
    else:
        gamma0 = np.random.normal(1.39, 0.1)
        T0 = np.random.normal(601, 3)
    return gamma0*np.exp((self.Teff-5777)/T0)   