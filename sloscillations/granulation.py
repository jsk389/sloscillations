# -*- coding: utf-8 -*-

import astropy.units as u
import celerite
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

from astropy.units import cds
from celerite import terms

cds.enable()

class Granulation(object):
    """
    Class to generate the granulation background using a 
    Gaussian process in the time domain
    """

    def __init__(self, time, dt, numax, kmag=11.0, amplitude=None, frequencies=None, white=None, n_comps=2):
        """

        Parameters
        ----------
        time : array_like
            Array of time-steps at which to evaluate the 
            granulation background model
        amplitude : array_like
            Array containing the amplitudes of the individual 
            granulation components
        frequencies : array_like
            Array containing the characteristic frequencies of 
            the individual granulation components 
            (must equal length of amplitude)
        white_noise : {0}, optional
            Value of white noise level in time-domain, if not 
            set then defaults to 0

        """

        self.time = time
        self.dt = dt
        self.kmag = kmag
        # Number of granulation components defaults to 2 - only 2 are currently supported
        self.n_comps = n_comps

        # Compute granulation parameters from scaling relations if not given
        self.numax = numax
        if amplitude is None:
            self.amplitude = self._scaling_relation(parameter='amplitude')
        else:
            self.amplitude = amplitude
        if frequencies is None: 
            self.frequencies = self._scaling_relation(parameter='frequencies')
        else:
            self.frequencies = frequencies
        if white is None:
            self.white = 0.0 #self._scaling_relation(parameter='white')
        else:
            self.white = white

    def _scaling_relation(self, parameter):
        """
        Scaling relations for granulation amplitudes and timescales as a function of numax
        """
        if parameter == 'amplitude':
            # Amplitude of the granulation components
            return np.array([3382 * self.numax ** -0.609] * self.n_comps)
        elif parameter == 'frequencies':
            # Characteristic frequencies of the granulation components
            return np.array([0.317 * self.numax ** 0.970, 0.948 * self.numax ** 0.992])
        elif parameter == 'white':
            # Compute white noise contribution using the formulae given in Chaplin 
            # et al. (2014)
            c = 3.46*10**(0.4*(12 - self.kmag) + 8)
            sigma = 1e6 * np.sqrt(c + 7e7) / c
            # Compute white noise from sigma using the formula given in Chaplin et
            # al. (2014)
            return sigma
        else:
            sys.exit(f'Incorrent parameter name given {parameter}')


    def calc_Bmax(self, frequency):
        """
        Calculate the power in the granulation background at numax
        """
        model = self.gran_backg(frequency)
        idx = np.argmin(abs(frequency - self.numax))
        return model[idx]

    def calculateS0(self):
        # Convert freq from uHz to 1/day
        omega_w = 2*np.pi*self.frequencies# / (1e6/86400.0)
       
        # Convert amplitude from units of ppm/uHz^1/2 into units of ppm / day^1/2 for GP
        new_a = self.amplitude# * np.sqrt(1e6/86400.0)

        return (new_a**2 / omega_w)*np.sqrt(np.pi) * (1e6/86400.0), omega_w / (1e6/86400.0)

    def gran_backg(self, frequency):

        # Need to make sure that frequency is in uHz! Use ASTROPY UNITS!

        if np.isscalar(self.amplitude):
            self.amplitude = np.array([self.amplitude]*self.n_comps)
        model = np.zeros(len(frequency))
        model += 2.0e-6 * self.white**2.0 * self.dt
        for i in range(len(self.frequencies)):
            height = ((2.0 * np.sqrt(2))/np.pi) * self.amplitude[i]**2/self.frequencies[i]
            model += height / (1 + (frequency/self.frequencies[i])**4)
        return model

    def compute_kernel(self):
        """
        doc-string needed
        """

        if self.white == 0:
            self.white = 1e-12

        self.kernel = terms.JitterTerm(log_sigma = np.log(self.white))

        self.S0, self.omega_w = self.calculateS0()
        #print(f"S0: {self.S0}")
        for i in range(len(self.S0)):
            self.kernel += terms.SHOTerm(log_S0=np.log(self.S0[i]),
                                    log_Q=np.log(1/np.sqrt(2)),
                                    log_omega0=np.log(self.omega_w[i]))
        self.gp = celerite.GP(self.kernel)
        return self.kernel, self.gp, self.S0

    def compute_backg_model(self, frequency):
        omega0 = 2*np.pi*self.frequencies
        omega = frequency * 2 * np.pi
        model = np.zeros(len(frequency))
        for i in range(len(self.S0)):
            model += (np.sqrt(2/np.pi)*(self.S0[i]) / ((omega/omega0[i])**4 + 1))
        return model

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

    gran = Granulation(t, numax)


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
