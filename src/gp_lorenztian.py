import astropy.units as u
import celerite
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

from astropy.units import cds
from celerite import terms

cds.enable()

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
        nu0_cpd = params[1] * (86400.0 / 1e6)
        lwd_cpd = params[2] * (86400.0 / 1e6)
        height_cpd = params[0] * (1e6/86400.0)

        # Account for normalisation in height
        height = height_cpd * np.sqrt(np.pi/2) / 4
        # 
        omega0 = 2 * np.pi * nu0_cpd 
        Q = nu0_cpd / lwd_cpd

        #
        coeffs = []
        for i in range(1):
            log_S0 = np.log(height) - 2*np.log(Q)
            log_Q = np.log(Q)
            log_omega0 = np.log(omega0)
            coeffs.append(super(AsteroTerm, self).get_all_coefficients([log_S0, log_Q, log_omega0]))
        return [np.concatenate(args) for args in zip(*coeffs)]

    #def get_terms(self):
    #    coeffs = self.get_complex_coefficients()
    #    return [terms.ComplexTerm(*(np.log(args))) for args in zip(*coeffs)]

def compute_mode(params, white=0):
    """
    doc-string needed
    """

    #if white == 0:
    #    white = 1e-12


    #print(f"S0: {S0}")

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
    kernel += terms.JitterTerm(log_sigma = np.log(white))

    gp = celerite.GP(kernel)
    return kernel, gp, S0

def compute_lor_model(f, params, dt, white=0):

    # dt must be in seconds!
    backg = 2e-6 * white**2 * dt

    model = np.ones_like(f) * backg
    
    x = (2 / params[2]) * (f - params[1])
    model += (params[0] / (1 + x**2))

    return model

def compute_full_model(f, params, dt, white=0):

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
    return (H / (x0**2 + x1**2)) + (2e-6*white**2*dt)

if __name__=="__main__":

    dt = 29.4*60
    t = np.arange(0, 1000*86400, dt)
    nyq = 1.0 / (2*dt)
    bw = 1.0 / t[-1]
    t = (t/86400.0) * u.day
  
    params = [10.0, 100.0, 0.2]
    white = 1.0
    kernel, gp, S0 = compute_mode(params, white=white)
    gp.compute(t.value)
    # model from Kallinger (2014)
    y = gp.sample()

    plt.plot(t/86400.0, y)
    plt.show()

    y = y*cds.ppm
    print(y)
    lc = lk.LightCurve(time=t, flux=y)

    # Approximate Nyquist Frequency and frequency bin width in terms of days
    nyquist = 0.5 * (1./(np.median(np.diff(lc.time))))

    ps = lc.to_periodogram(normalization='psd',
                           freq_unit=u.microhertz)

    f = np.arange(bw, nyq, bw) * 1e6
    lor = compute_lor_model(f, params, dt, white=white)
    full = compute_full_model(f, params, dt, white=white)

    #backg_model = compute_backg_model(f*1e-6, S0, freqs*1e-6)
#    backg_model = gran_backg(f, amps, freqs)
    # Convert frequency to 1/day for computation of psd
    psd = kernel.get_psd(2*np.pi*f*(86400.0/1e6))# / (2 * np.pi)
    print(t.max())
    # Get back into correct units
    psd *= (2 / (t.max().value))
    psd *= (2 / (f[1]-f[0]))

    #plt.plot(ps.frequency, ps.power)
    # Have to add in white noise component to psd as celerite doesn't do
    # it by default 
    plt.plot(f, psd + 2e-6*white**2*dt, color='r')
    print(np.sum(lor)/np.sum(psd + 2e-6*white**2*dt))
    plt.plot(f, lor, color='k')
    plt.plot(f, full, color='g')
    #plt.plot(f*1e6, backg_model)
    plt.show()

    plt.plot(f, (lor) / full)
    plt.show()
