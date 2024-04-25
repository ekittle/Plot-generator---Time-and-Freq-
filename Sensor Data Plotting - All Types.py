# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:21:54 2024

@author: EvanKittle
"""

import numpy as np

from matplotlib import pyplot as plt
from constants import CONVERSION
    
from dsp import notch_out_60Hz_harmonics, noise_density, butter_bandpass_filter, _butter_bandpass

temp = np.load(r"C:\Users\EvanKittle\Sonera Magnetics\Neuroscience Drive - Documents\Backup\Gesture\2024-03-08 15-41-51 Test Data\2024-03-08 15-42-23 Test Data.npz")
data = temp['data']
fs = temp['fs']
n_samples = 36720
sensors = temp['sensors']

filteredtemp = butter_bandpass_filter(data, 20, 200, 2000)
notchedtemp = notch_out_60Hz_harmonics(filteredtemp, 2000, 1000, notch_width=2)
f, freqdomain = noise_density(notchedtemp, fs, fs)

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
    
freqdomainmean = np.mean(freqdomain, axis=2)
for channel_number, channel_name in enumerate(sensors):
    subplot = sensordict[channel_name]
    datasubset = freqdomainmean[:, channel_number]
    subplot.loglog(f, datasubset/CONVERSION[channel_name], label=f"{channel_name}")
    subplot.set_ylim(1e-4, 1e0)
    subplot.set_xlim(10, 500)
    subplot.set_ylabel("pT")
    subplot.set_title(channel_name)
    subplot.legend()
    
plt.show()







#for each channel number, replace with corresponding value of active_opm list
#if DH, DJ, TDK, EMG then plot a subplot
#if only opms, only have one plot