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
   