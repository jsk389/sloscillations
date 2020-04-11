# -*- coding: utf-8 -*-

import numpy as np

from celerite import terms

class AsteroTerm(terms.SHOTerm):

    parameter_names = (
        "height", "freq", "lwd",
    )

    def __init__(self, *args, **kwargs):
        super(AsteroTerm, self).__init__(*args, **kwargs)

    #def get_complex_coefficients(self, params):
    #    (log_S, log_omega, log_Q) = params
    #    
    #    a = np.exp(log_S + log_omega + log_Q)
    #    b = a / np.sqrt(4*np.exp(2*log_Q) - 1)
    #    c = np.exp(log_omega) / (2 * np.exp(log_Q))
    #    d = c * np.sqrt(4*np.exp(2*log_Q) - 1)
    #    return (
    #        a,
    #        b,
    #        c,
    #        d
    #    )

    def get_all_coefficients(self, params):

        # Convert to cycles per day
        # Frequency was in units of uHz now cpd
        nu0_cpd = params[1] * (86400.0 / 1e6)
        # Linewidth was in units of uHz now cpd
        lwd_cpd = params[2] * (86400.0 / 1e6)
        # Height was in units of ppm^2/uHz now in ppm^2/cpd
        height_cpd = params[0] * (1e6/86400.0)

        # Account for normalisation in height
        height = height_cpd * np.sqrt(np.pi/2) / 4
        # Convert frequency from cpd to radians
        omega0 = 2 * np.pi * nu0_cpd 
        # Compute the oscillation quality factor
        Q = nu0_cpd / lwd_cpd

        #
        coeffs = []
        for i in range(1):
            log_S0 = np.log(height) - 2*np.log(Q)
            log_Q = np.log(Q)
            log_omega0 = np.log(omega0) # - (i*2*np.pi)*86400.0/1e6) Don't forget omega is in units of rads/day
            coeffs.append(super(AsteroTerm, self).get_all_coefficients([log_S0, log_Q, log_omega0]))
        return [np.concatenate(args) for args in zip(*coeffs)]

    #def get_terms(self):
    #    coeffs = self.get_complex_coefficients()
    #    return [terms.ComplexTerm(*(np.log(args))) for args in zip(*coeffs)]