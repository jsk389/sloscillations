# -*- coding: utf-8 -*-

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd


from astropy.units import cds
from sloscillations import generate_data

cds.enable()


if __name__=="__main__":
    metadata = pd.read_csv('metadata.csv')
    dat = generate_data.GenerateData(metadata)
    dt = 29.4*60.0
    t = np.arange(0, metadata['T'].values, metadata['dt'].values)
    t = (t/86400.0) * u.day
    t, y, data = dat(t)
    np.savetxt('test_data.txt', np.c_[t.value, y])
    data.to_csv('test_data_freqs.csv', index=False)