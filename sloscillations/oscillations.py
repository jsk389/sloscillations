# -*- coding: utf-8 -*-

import astropy.units as u
import celerite
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

from astropy.units import cds
from celerite import terms
from .asteroterm import AsteroTerm
cds.enable()

class Oscillations(object):
    """
    Class to generate oscillation modes using a 
    Gaussian process in the time domain
    """

    def __init__(self, time):
        """

        Parameters
        ----------
        time : array_like
            Array of time-steps at which to evaluate the 
            granulation background model
        params : array_like
            Array containing the heights, frequencies
            and linewidths of the oscillation modes to
            be computed.

        """
        self.time = time

    def _compute_mode_kernel(self, params):
        # Convert parameters to cycles per day
        lwd = params[2] * (86400.0 / 1e6)
        nu0 = params[1] * (86400.0 / 1e6)
        height = params[0] * (86400.0/1e6)

        Q = (2 * np.pi * nu0) / (2 * np.pi * lwd)
        omega = 2 * np.pi * nu0
        S0 = height * (lwd/nu0)**2 * 4 * np.sqrt(2/np.pi)

        kernel = AsteroTerm(height=params[0],
                            freq = params[1],
                            lwd = params[2])

        return kernel

    def _compute_modes_kernel(self, params, white=0):
        """
        doc-string needed
        """ 

        for i in range(len(params)):
            if i == 0:
                kernel = self._compute_mode_kernel(params[i,:])
            else:
                kernel += self._compute_mode_kernel(params[i,:])
        if white > 0:
            kernel += terms.JitterTerm(log_sigma = np.log(white))

        return kernel

    def compute_gp(self, params, white=0, return_kernel=True):

        kernel = self._compute_modes_kernel(params, white=white)
        gp = celerite.GP(kernel)
        if return_kernel:
            return kernel, gp
        return gp


    def _single_lor_model(self, f, params):
        x = (2 / params[2]) * (f - params[1])
        model = (params[0] / (1 + x**2))

        return model  

    def compute_lor_model(self, f, params, dt, white=0):

        # dt must be in seconds!
        backg = 2e-6 * white**2 * dt

        model = np.ones_like(f) * backg

        for i in range(len(params)):
            model += self._single_lor_model(f, params[i,:])
        
        return model

    def _single_osc_model(self, f, params):

        # Convert parameters to cycles per day
        lwd = params[2] * (86400.0 / 1e6)
        nu0 = params[1] * (86400.0 / 1e6)
        height = params[0]# * (1e6/86400.0)

        Q = (2 * np.pi * nu0) / (2 * np.pi * lwd)
        omega0 = 2 * np.pi * nu0
        omega = 2 * np.pi * f * (86400.0 / 1e6)
        S0 = height * (lwd/nu0)**2# * 4 * np.sqrt(2/np.pi)

        H = S0*omega0**4
        x0 = (omega**2 - omega0**2)
        x1 = (omega*omega0)/Q
        return (H / (x0**2 + x1**2))   

    def compute_full_model(self, f, params, dt, white=0):

        model = np.ones_like(f) * (2e-6*white**2*dt)
        for i in range(len(params)):
            model += self._single_osc_model(f, params[i,:])
        return model

if __name__=="__main__":

    # Cadence in seconds
    dt = 29.4*60
    # Time array - 1000 days
    t = np.arange(0, 1000*86400, dt)
    # Compute Nyquist frequency
    nyq = 1.0 / (2*dt)
    # Compute bin width
    bw = 1.0 / t[-1]
    # Convert time array into days from seconds
    t = (t/86400.0) * u.day
  
    # Set up oscillation test parameters
    params = np.array([10.0, 100.0, 0.2]).reshape(1,-1)
    # White noise level (in ppm)
    white = 1.0
    # Compute the kernel of the oscillation mode
    osc = Oscillation(t)
    kernel, gp = osc.compute_gp(params, white=white)
    # Compute
    gp.compute(t.value)
    # model from Kallinger (2014)
    # Sample from gp 
    y = gp.sample()

    # Plot time-series
    plt.plot(t/86400.0, y)
    plt.show()

    # Give units of ppm for lightkurve
    y = y*cds.ppm

    #
    lc = lk.LightCurve(time=t, flux=y)

    # Approximate Nyquist Frequency and frequency bin width in terms of days
    nyquist = 0.5 * (1./(np.median(np.diff(lc.time))))

    # Compute periodogram
    ps = lc.to_periodogram(normalization='psd',
                           freq_unit=u.microhertz)

    # Compute frequency array for analytical mode profile computation
    # and for evaluating gp psd
    f = np.arange(bw, nyq, bw) * 1e6
    # Standard lorentzian profile
    lor = osc.compute_lor_model(f, params, dt, white=white)
    # Analytical psd for chosen gp kernel
    full = osc.compute_full_model(f, params, dt, white=white)

    # Convert frequency to 1/day for computation of psd
    psd = kernel.get_psd(2*np.pi*f*(86400.0/1e6))

    # Get back into correct units i.e. normalisation
    psd *= (2 / (t.max().value))
    psd *= (2 / (f[1]-f[0]))

    # Plot PSD and analytical models
    plt.plot(ps.frequency, ps.power)

    # Have to add in white noise component to psd as celerite doesn't do
    # it by default 
    plt.plot(f, psd + 2e-6*white**2*dt, color='r')

    print(np.sum(lor)/np.sum(psd + 2e-6*white**2*dt))
    plt.plot(f, lor, color='k')
    plt.plot(f, full, color='g')
    #plt.plot(f*1e6, backg_model)
    plt.xlim(99, 101)
    plt.show()

    #plt.plot(f, (lor) / full)
    plt.plot(f, ps.power / lor)
    plt.show()
