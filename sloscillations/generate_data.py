# -*- coding: utf-8 -*-

import astropy.units as u
import json
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

from astropy.units import cds
from . import utils
from . import granulation, frequencies, linewidths, amplitudes, heights, oscillations

from scipy.interpolate import interp1d
#from frequencies import Frequencies
#from amplitudes import Amplitude
#from linewidths import Linewidths
#from heights import Heights

cds.enable()


class GenerateData:
    """
    Generate the oscillation modes
    """

    def _setup_attrs(self):
        ## Setup attributed
        self.delta_nu = None
        self.epsilon_p = None
        self.alpha = None
        self.radial_order_range = [-5, 6]
        self.d01 = None
        self.d02 = None
        self.d03 = None
        self.Henv = None
        self.denv = None
        self.amax = None
        self.DPi1 = None
        self.coupling = None
        self.eps_g = None
        self.split_core = None
        self.split_env = 0.0
        self.calc_l0 = True
        self.calc_l2 = True
        self.calc_l3 = True
        self.calc_nom_l1 = True
        self.calc_mixed = True
        self.calc_rot = True
        self.calc_method = 'Mosser2018update'
        self.l = 1
        self.method = 'simple'
        self.mission = 'Kepler'
        self.evo_state = 'RGB'
        self.inclination_angle = None
        self.vis_tot = None
        self.vis1 = None
        self.vis2 = None
        self.vis3 = None
        self.T = None
        self.white = 0.
        self.a1 = None
        self.a2 = None
        self.amplitude = None
        self.b1 = None
        self.b2 = None
        self.frequencies = None

    def _write_attrs(self, fname='test_params.txt'):
        """
        Write out attributes to file
        """
        ## Setup attributed
        df = pd.DataFrame(columns = ['DeltaNu', 'eps_p', 'alpha', 
                                     'd01', 'd02', 'd03', 'Henv', 'denv', 'amax',
                                     'DPi1', 'coupling', 'eps_g', 
                                     'split_core', 'split_env', 
                                     'calc_l0', 'calc_l2', 'calc_l3', 'calc_nom_l1',
                                     'calc_mixed', 'calc_rot', 'calc_method',
                                     'l', 'method', 'mission', 'evo_state',
                                     'inclination_angle', 
                                     'vis_tot', 'vis1', 'vis2', 'vis3', 'T',
                                     'white', 'a1', 'a2', 'b1', 'b2'])
        df['DeltaNu'] = self.delta_nu
        df['eps_p'] = self.epsilon_p
        df['alpha'] = self.alpha 
        df['d01'] = self.d01
        df['d02'] = self.d02 
        df['d03'] = self.d03 
        df['Henv'] = self.Henv 
        df['denv'] = self.denv
        df['amax'] = self.amax 
        df['DPi1'] = self.DPi1
        df['coupling'] = self.coupling
        df['eps_g'] = self.eps_g 
        df['split_core'] = self.split_core
        df['split_env'] = self.split_env
        df['calc_l0'] = self.calc_l0
        df['calc_l2'] = self.calc_l2
        df['calc_l3'] = self.calc_l3
        df['calc_nom_l1'] = self.calc_nom_l1
        df['calc_mixed'] = self.calc_mixed
        df['calc_rot'] = self.calc_rot
        df['calc_method'] = self.calc_method
        df['l'] = self.l
        df['method'] = self.method
        df['mission'] = self.mission
        df['evo_state'] = self.evo_state
        df['inclination_angle'] = self.inclination_angle
        df['vis_tot'] = self.vis_tot 
        df['vis1'] = self.vis1 
        df['vis2'] = self.vis2
        df['vis3'] = self.vis3
        df['T'] = self.T
        df['white'] = self.white
        df['a1'] = self.a1
        df['a2'] = self.a2
        df['b1'] = self.b1
        df['b2'] = self.b2
        df.to_csv(fname, index=False)


    def __init__(self, metadata, data=None):

        self._setup_attrs()

        # DataFrame containing metadata e.g. length of timeseries etc.
        self.metadata = metadata
        # Import metadata
        self._import_metadata(self.metadata)

        # Compute frequency array
        self._construct_frequency_array()
        # DataFrame containing all the frequency, amplitude and linewidth
        # data
        # If it isn't given than calculate the data
        if data is None:
            self.precalculate()

        if self.a1 is not None and self.a2 is not None:
            self.amplitude = [self.a1, self.a2]
                
        if self.b1 is not None and self.b2 is not None:
            self.frequencies = [self.b1, self.b2]
        

        self.granulation = granulation.Granulation(time=None, dt=self.dt, 
                                                   numax=self.numax, 
                                                   amplitude=self.amplitude,
                                                   frequencies=self.frequencies,
                                                   white=self.white)
        
        #self.power_spectrum_model()

        


    def _import_metadata(self, metadata):
        """
        Import the metadata and extract the relevant quantities
        """
        #############################
        # Timeseries based quantities
        #############################
        for (columnName, columnData) in metadata.iteritems():
            setattr(self, columnName, columnData.values)
        
        #self.dt = metadata['dt'].values
        #self.T = metadata['T'].values

        ###############################
        # Global oscillation quantities
        ###############################
        #self.numax = metadata['numax'].values
        # DeltaNu
        #if 'delta_nu' in metadata.columns:
        #    self.delta_nu = metadata['delta_nu'].values
        #else:
        #    self.delta_nu = None
        # Epsilon p
        #if 'eps_p' in metadata.columns:
        #    self.eps_p = metadata['eps_p'].values
        #else:
        #    self.eps_p = None
        # Alpha
        #if 'alpha' in metadata.columns:
        #    self.alpha = metadata['alpha'].values
        #else:
        #    self.alpha = None
        ## Radial order range
        #if hasattr(self, 'radial_order_range'):
        #    self.radial_order_range = [-self.radial_order_range.values//2,
        #                                self.radial_order_range.values//2 + 1]
        #else:
        #    self.radial_order_range = [-5, 6]

        ################################
        # Mixed mode quantities
        ################################
        # Period spacing
        #if 'DPi1' in metadata.columns:
        #    self.DPi1 = metadata['DPi1'].values
        #else:
        #    sys.exit('Period spacing DPi1 not given in metadata')
        # Coupling
        #if 'coupling' in metadata.columns:
        #    self.coupling = metadata['coupling'].values
        #else:
        #    sys.exit('Coupling not given in metadata')
        # Epsilon g
        #if 'eps_g' in metadata.columns:
        #    self.eps_g = metadata['eps_g'].values
        #else:
        #    print('Epsilon_g not given in metadata, therefore setting to 0')
        #    self.eps_g = 0.0
        # Rotational splitting
        #if 'drot' in metadata.columns:
        #    self.drot = metadata['drot'].values
        #else:
        #    print('Rotational splitting (drot) not give in metadata, therefore setting to 0')
        #    self.drot = 0.0
        # Inclination angle
        #if 'inclination_angle' in metadata.columns:
        #    self.inc = metadata['inclination_angle'].values
        #else:
        #    print("Inclination angle not given in metadata, therefore setting to 90 degrees")

    def _construct_frequency_array(self):
        
        # Check either length of timeseries (T) and cadence (dt) are given,
        # or that Nqyuist frequency (nyq) and frequency resolution (bw) are
        # given.
        if hasattr(self, 'T') and hasattr(self, 'dt'):
            self.nyq = 1e6 / (2*self.dt)
            self.bw = 1e6 / self.T
        elif hasattr(self, 'nyq') and hasattr(self, 'bw'):
            self.T = 1e6 / self.bw
            self.dt = 1e6/(2*self.nyq)
        else:
            sys.exit('Either T and dt or nyq and bw need to be given!')
        self.frequency = np.arange(self.bw, self.nyq, self.bw)



    def precalculate(self, calc_l0=True, calc_l2=True, calc_l3=True,
                           calc_nom_l1=True, calc_mixed=True, calc_rot=True):
        """
        Compute the mode frequencies, amplitudes, linewidths, heights etc.
        """

        # Construct the frequency array


        # Construct parameter dictionary from metadata
        print(f"Inclination Angle: {self.inclination_angle} degrees")
        # Set up frequencies class
        self.freqs = frequencies.Frequencies(frequency=self.frequency,
                            numax=self.numax, 
                            delta_nu=self.delta_nu, 
                            epsilon_p=self.epsilon_p,
                            alpha=self.alpha,
                            radial_order_range=self.radial_order_range,
                            DPi1=self.DPi1,
                            coupling=self.coupling,
                            eps_g=self.eps_g,
                            split_core=self.split_core,
                            split_env=self.split_env,
                            inclination_angle=self.inclination_angle,
                            calc_rot=True,
                            Henv=1114.117,
                            denv=2*np.sqrt(2*np.log(2)) * 9.778543)
        # Compute frequencies
        print(self.freqs.freq_attrs)
        self.freqs()
        sys.exit()

        # Compute amplitudes
        self.amps = amplitudes.Amplitudes(self.freqs)
        self.amps()

        # Linewidths
        self.lwd = linewidths.Linewidths(self.amps)
        self.lwd()

        # Heights
        self.h = heights.Heights(self.lwd, self.T)
        self.h()

        self.data = self.h.mode_data
        #print(h.mode_data.loc[h.mode_data['m'] == -1, ])
      
    def power_spectrum_model(self):
        """
        Compute model in power spectrum
        """
        model = self.granulation.gran_backg(self.frequency)
        #model = np.zeros(len(self.frequency))
        for idx, row in self.data.iterrows():
            if row['l'] != -1:
                model += utils.compute_model(self.frequency,
                                             row['frequency'],
                                             row['linewidth'],
                                             row['height'])
        test = pd.read_csv('../../../TACO/data/008546976/pds.csv')
        #plt.plot(test['frequency'], test['power'], zorder=-1, color='k')
        #plt.plot(self.frequency, -np.log(np.random.uniform(0, 1, len(model)))*model, color='k', zorder=0)
        #plt.plot(self.frequency, model, color='r', lw=2, zorder=0)
        #l1 = self.data.loc[self.data['l'] == 1, ]
        #grouped = l1.groupby(['n_g'])
        #for name, group in grouped:
        #    color = next(plt.gca()._get_lines.prop_cycler)['color']
        #    for idx, i in group.iterrows():
        #        if (i['m'] == -1) or (i['m'] == +1):
        #            plt.scatter(i['frequency'], model.max()/2.5, color=color, marker='v', zorder=1)
        #        elif i['m'] == 0:
        #            plt.scatter(i['frequency'], model.max()/2, color=color, marker='v', zorder=1)#

            #print(name, group)
        #plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
        #plt.ylabel(r'PSD (ppm$^{2}\mu$Hz$^{-1}$', fontsize=18)
        #plt.show()
        return model

    def __call__(self, t=None):
        
        if t is None:
            sys.exit('No time array given, can\'t continute!')

        kernel, gp, S0 = self.granulation.compute_kernel()
        gp.compute(t.value)
        # model from Kallinger (2014)
        y = gp.sample()

        print(self.data)
        data = self.data.loc[self.data['l'] >= 0, ]

        # Compute the kernel of the oscillation mode
        osc = oscillations.Oscillations(t)
        for idx, i in data.iterrows():
            # Set up oscillation test parameters
            params = np.array([i['height'], 
                               i['frequency'], 
                               i['linewidth']]).reshape(1,-1)          

            tmp_kernel, gp = osc.compute_gp(params, white=0)
            kernel += tmp_kernel
            # Compute
            gp.compute(t.value)
            # model from Kallinger (2014)
            # Sample from gp 
            y += gp.sample()

        #plt.plot(t, y)
        #plt.show()

#       
        lc = lk.LightCurve(time=t, flux=y*cds.ppm)

        # Approximate Nyquist Frequency and frequency bin width in terms of days
        nyquist = 0.5 * (1./(np.median(np.diff(lc.time))))

        # Compute periodogram
        ps = lc.to_periodogram(normalization='psd',
                                freq_unit=u.microhertz)

        # Compute frequency array for analytical mode profile computation
        # and for evaluating gp psd
        f = np.arange(self.bw, self.nyq, self.bw)
        # Standard lorentzian profile
        #lor = osc.compute_lor_model(f, params, self.dt, white=white)
        # Analytical psd for chosen gp kernel
        #full = osc.compute_full_model(f, params, self.dt, white=white)

        # Convert frequency to 1/day for computation of psd
        psd = kernel.get_psd(2*np.pi*f*(86400.0/1e6))

        # Get back into correct units i.e. normalisation
        psd *= (2 / (t.max().value))
        psd *= (2 / (f[1]-f[0]))

        # Plot PSD and analytical models
        #plt.plot(ps.frequency, ps.power, 'k')
        #plt.plot(f, psd + 2e-6*self.white**2*self.dt, color='r')
        #plt.plot(self.frequency, self.power_spectrum_model(), color='g')

        #plt.show()

        #plt.plot(self.frequency, (psd + 2e-6*self.white**2*self.dt) / self.power_spectrum_model())
        #plt.show()
        #print(f"Epsilon p: {self.eps_p}")
        self._write_attrs()

        return t, y, self.data



        # Generate background parameters

    
if __name__=="__main__":

    metadata = pd.read_csv('tests/metadata.csv')
    dat = GenerateData()

    