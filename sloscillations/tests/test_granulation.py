# -*- coding: utf-8 -*-

import astropy.units as u
import lightkurve as lk
import numpy as np
import pytest

from astropy.units import cds
from sloscillations import granulation

import matplotlib.pyplot as plt

cds.enable()

def test_granulation():

    dt = 29.4*60
    t = np.arange(0, 1000*86400, dt)
    nyq = 1.0 / (2*dt)
    bw = 1.0 / t[-1]
    t = (t/86400.0) * u.day
    numax = 100.0

    a = 3382 * numax ** -0.609
    b1 = 0.317 * numax ** 0.970
    b2 = 0.948 * numax ** 0.992

    amps = np.array([a, a])# * cds.ppm
    freqs = np.array([b1, b2])# * u.Hertz

    gran = granulation.Granulation(t, numax)


    kernel, gp, S0 = gran.compute_kernel()
    gp.compute(t.value)
    # model from Kallinger (2014)
    y = gp.sample()

    y = y*cds.ppm

    lc = lk.LightCurve(time=t, flux=y)

    # Approximate Nyquist Frequency and frequency bin width in terms of days
    nyquist = 0.5 * (1./(np.median(np.diff(lc.time))))

    ps = lc.to_periodogram(normalization='psd',
                           freq_unit=u.microhertz)
 

    f = np.arange(bw, nyq, bw) * 1e6

    backg_model = gran.compute_backg_model(f*1e-6)
    # Convert frequency to 1/day for computation of psd
    psd = kernel.get_psd(2*np.pi*f*(86400.0/1e6))# / (2 * np.pi)
    print(t.max())
    # Get back into correct units
    psd *= (2 / (t.max().value))
    psd *= (2 / (f[1]-f[0]))

    backg_model = gran.gran_backg(f)
    gran_model = gran.compute_backg_model(f)
    gran_model *= (2 / (t.max().value))
    gran_model *= (2 / (f[1]-f[0]))


   #plt.plot(f*1e6, backg_model)
    assert np.allclose(1/np.mean(psd/backg_model), 1)
    assert np.allclose(1/np.mean(psd/gran_model), 1)

    #print("bkg -> psd: ", 1/np.mean(psd/backg_model))
    #print("gp -> psd: ", 1/np.mean(psd/gran_model))
   
if __name__=="__main__":

    dt = 29.4*60
    t = np.arange(0, 1000*86400, dt)
    nyq = 1.0 / (2*dt)
    bw = 1.0 / t[-1]
    t = (t/86400.0) * u.day
    numax = 100.0

    a = 3382 * numax ** -0.609
    b1 = 0.317 * numax ** 0.970
    b2 = 0.948 * numax ** 0.992

    print(2*np.sqrt(2)/np.pi * a**2/b1)
    print(2*np.sqrt(2)/np.pi * a**2/b2)
    print(b1, b2)

    amps = np.array([a, a])# * cds.ppm
    freqs = np.array([b1, b2])# * u.Hertz

    gran = granulation.Granulation(t, dt, numax)


    kernel, gp, S0 = gran.compute_kernel()
    gp.compute(t.value)
    # model from Kallinger (2014)
    y = gp.sample()

    #plt.plot(t/86400.0, y)
    #plt.show()

    y = y*cds.ppm
    print(y)
    lc = lk.LightCurve(time=t, flux=y)

    # Approximate Nyquist Frequency and frequency bin width in terms of days
    nyquist = 0.5 * (1./(np.median(np.diff(lc.time))))
    #fs = (1./(lc.time[-1] - lc.time[0])) / oversample_factor
    ps = lc.to_periodogram(normalization='psd',
                           freq_unit=u.microhertz)
                           #maximum_frequency=nyq*1e6*u.microhertz,
                           #minimum_frequency=bw*1e6*u.microhertz)
    #ps.plot()
    #plt.show()


    f = np.arange(bw, nyq, bw) * 1e6

    backg_model = gran.compute_backg_model(f*1e-6)
    # Convert frequency to 1/day for computation of psd
    psd = kernel.get_psd(2*np.pi*f*(86400.0/1e6))# / (2 * np.pi)
    print(t.max())
    # Get back into correct units
    psd *= (2 / (t.max().value))
    psd *= (2 / (f[1]-f[0]))

    plt.plot(ps.frequency, ps.power, color='k')
    plt.plot(f, psd, color='r')
    backg_model = gran.gran_backg(f)
    gran_model = gran.compute_backg_model(f)
    gran_model *= (2 / (t.max().value))
    gran_model *= (2 / (f[1]-f[0]))

    print("S0: ", S0*4*np.sqrt(2/np.pi) / (1e6/86400.0))
    print("a^2", amps**2/freqs * (2*np.sqrt(2)/np.pi))
    print((S0*4*np.sqrt(2/np.pi) / (1e6/86400.0))/(amps**2/freqs * (2*np.sqrt(2)/np.pi)))

    for i in freqs:
        plt.axvline(i, color='r', linestyle='--')
    plt.plot(f, gran_model, color='C0')
    plt.plot(f, backg_model, color='C1')
    #plt.plot(f*1e6, backg_model)
    print("bkg -> psd: ", 1/np.mean(psd/backg_model))
    print("gp -> psd: ", 1/np.mean(psd/gran_model))
    plt.show()