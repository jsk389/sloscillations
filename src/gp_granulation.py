import astropy.units as u
import celerite
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

from astropy.units import cds
from celerite import terms

cds.enable()

def calculateS0(a, freq):
    # Convert freq from uHz to 1/day
    omega_w = 2*np.pi*freq / (1e6/86400.0)
    omega = 2*np.pi*freq# * 1e6
    
    # Convert a**2 into units of ppm^2 / day
    new_a = a * np.sqrt(1e6/86400.0)
    #return (a**2/(np.sqrt(2)*omega)) * np.sqrt(2)

    #return (a**2 / omega)*np.sqrt(np.pi)/np.sqrt(2)
    return (new_a**2 / omega_w)*np.sqrt(2), omega_w
    #return np.sqrt(2) * a**2 / omega# / np.sqrt(np.pi)

def gran_backg(f, a, b):

    model = np.zeros(len(f))
    for i in range(len(a)):
        height = ((2.0 * np.sqrt(2))/np.pi) * a[i]**2/b[i]
        model += height / (1 + (f/b[i])**4)
    return model

def compute_background(amp, freqs, white=0):
    """
    doc-string needed
    """

    if white == 0:
        white = 1e-12

    #white = 0

    kernel = terms.JitterTerm(log_sigma = np.log(white))

    S0, omega_w = calculateS0(amp, freqs)
    print(f"S0: {S0}")
    for i in range(len(amp)):
        kernel += terms.SHOTerm(log_S0=np.log(S0[i]),
                                log_Q=np.log(1/np.sqrt(2)),
                                log_omega0=np.log(omega_w[i]))
    gp = celerite.GP(kernel)
    return kernel, gp, S0

def compute_backg_model(f, S0, freqs):
    omega0 = freqs
    omega = 2*np.pi*f
    model = np.zeros(len(f))
    for i in range(len(S0)):
        model += (4*(S0[i]) / ((omega/omega0[i])**4 + 1))
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

    amps = np.array([a, a])# * cds.ppm
    freqs = np.array([b1, b2])# * u.Hertz

    kernel, gp, S0 = compute_background(amps, freqs)
    gp.compute(t.value)
    # model from Kallinger (2014)
    y = gp.sample()

    y = y*cds.ppm
    print(y)
    lc = lk.LightCurve(time=t, flux=y)

    # Approximate Nyquist Frequency and frequency bin width in terms of days
    nyquist = 0.5 * (1./(np.median(np.diff(lc.time))))
    #fs = (1./(lc.time[-1] - lc.time[0])) / oversample_factor
    ps = lc.to_periodogram(normalization='psd',
                           freq_unit=u.microhertz)



    f = np.arange(bw, nyq, bw) * 1e6

    #backg_model = compute_backg_model(f*1e-6, S0, freqs*1e-6)
    backg_model = gran_backg(f, amps, freqs)
    # Convert frequency to 1/day for computation of psd
    psd = kernel.get_psd(2*np.pi*f*(86400.0/1e6))# / (2 * np.pi)
    print(t.max())
    # Get back into correct units
    psd *= (2 / (t.max().value))
    psd *= (2 / (f[1]-f[0]))

    plt.plot(ps.frequency, ps.power)
    plt.plot(f, psd, color='r')
    plt.plot(f, backg_model*10, color='g')
    gran = gran_backg(f, amps, freqs)
    plt.show()

    plt.hist(ps.power.value/gran, bins=300, histtype='step', density=True)
    plt.hist(ps.power.value/psd, bins=300, histtype='step', density=True)
    from scipy.stats import chi2
    plt.hist(chi2.rvs(2, size=10000)/2, bins=100, histtype='step', density=True)

    plt.show()
