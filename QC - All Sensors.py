# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:21:54 2024

@author: EvanKittle
"""
"""filters data"""
"""creates a dictionary to produce multiple subplots based on the number of unique sensors in the dataset"""
"""fft plot, conversions for every unique sensor, autoscale and log y-scale, channel label"""


import numpy as np

from matplotlib import pyplot as plt
from constants import CONVERSION
    
from dsp import notch_out_60Hz_harmonics, noise_density, butter_bandpass_filter, _butter_bandpass

filteredtemp = butter_bandpass_filter(data, 20, 200, 2000)
notchedtemp = notch_out_60Hz_harmonics(filteredtemp, 2000, 1000, notch_width=2)
f, fft = noise_density(notchedtemp, int(fs), fs)


uniquesensors = np.unique(sensors)
fig, axes = plt.subplots(len(uniquesensors),1)

i = 0
sensordict = {}
for channel_name in uniquesensors:
    if len(uniquesensors) == 1:
        sensordict[channel_name] = axes
    else :
        sensordict[channel_name] = axes[i]
    i = i+1

freqdomainmean = np.mean(fft, axis=2)
for channel_number, channel_name in enumerate(sensors):
    subplot = sensordict[channel_name]
    datasubset = freqdomainmean[:, channel_number]
    subplot.loglog(f, datasubset/CONVERSION[channel_name], label=f"{channel_name}")
    subplot.set_xlim(10, 500)
    subplot.autoscale(enable=True, axis='y', tight=True)
    subplot.set_yscale("log")
    subplot.set_title(channel_name)
    subplot.legend()
    
plt.show()

