# -*- coding: utf-8 -*-

import astropy.units as u
import lightkurve as lk
import numpy as np
import pytest

from astropy.units import cds
from sloscillations import oscillations

import matplotlib.pyplot as plt

cds.enable()

def test_oscillations():
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
    osc = oscillations.Oscillations(t)
    kernel, gp = osc.compute_gp(params, white=white)
    # Compute
    gp.compute(t.value)
    # model from Kallinger (2014)
    # Sample from gp 
    y = gp.sample()

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
    psd += (2e-6*white**2*dt)

    #plt.plot(f, psd)
    #plt.plot(f, lor)
    #plt.show()

    #plt.plot(f, psd/lor)
    #plt.show()

    assert np.allclose(psd, full)

if __name__=="__main__":

    #test_oscillations()
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
    osc = oscillations.Oscillations(t)
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
