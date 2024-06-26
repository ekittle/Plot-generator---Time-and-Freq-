# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:45:33 2024

@author: EvanKittle
"""

import numpy as np
import math
from scipy import signal
from matplotlib import pyplot as plt
from constants import CONVERSION, FS, FC_L, FC_H 
    
from dsp import notch_out_60Hz_harmonics, noise_density, butter_bandpass_filter, _butter_bandpass, noise_specgram

plt.close('all')

filtered_temp = butter_bandpass_filter(data, FC_L, FC_H, FS)
notched_temp = notch_out_60Hz_harmonics(filtered_temp, FS, FS/2, notch_width=2)
notched_temp = notched_temp[:,800:len(notched_temp)-800]
f, fft = noise_density(notched_temp, int(FS), FS)

unique_sensors = np.unique(sensors)

fig, (ax1, ax2, ax3) = plt.subplots(3, len(unique_sensors), height_ratios=(.25, .5, .25))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.25,hspace=0.25)

i = 0
sensordict = {}
for channel_name in unique_sensors:
    if len(unique_sensors) == 1:
        sensordict[channel_name] = [ax1, ax2, ax3]
    else:
        sensordict[channel_name] = [ax1[i], ax2[i], ax3[i]]
    i += 1

x = np.arange(len(notched_temp[1]/FS))
offset = 0

spectrograms = []
times = []
frequencies = None

freq_domain_mean = np.mean(fft, axis=2)
for channel_number, channel_name in enumerate(sensors):
    subplot_freq = sensordict[channel_name][0]
    subplot_time = sensordict[channel_name][1]
    subplot_spect = sensordict[channel_name][2]

    datasubset_freq = freq_domain_mean[10:500, channel_number]
    subplot_freq.loglog(f[10:500], datasubset_freq/CONVERSION[channel_name], label=f"{sensor_names[channel_number]}")
    subplot_freq.set_xlim(10, 500)
    subplot_freq.autoscale(enable=True, axis='y', tight=True)
    subplot_freq.set_yscale("log")
    subplot_freq.set_title(channel_name)
    subplot_freq.set_xlabel("Hz")
    subplot_freq.legend(bbox_to_anchor = (1.1, 0.3), loc='center right', fontsize="7")

    datasubset_time = notched_temp[channel_number,:]
    subplot_time.plot(x, datasubset_time + offset, label=f"{sensor_names[channel_number]}")
    subplot_time.get_yaxis().set_visible(False)
    subplot_time.set_xlim(0, len(datasubset_time))
    subplot_time.autoscale(enable=True, axis='y', tight=True)
    subplot_time.set_title(channel_name)
    offset += np.max(np.abs(datasubset_time))
    
    f_spect, t_spect, Sxx = signal.spectrogram(datasubset_time, FS)
    if frequencies is None:
        frequencies = f_spect
    times.append(t_spect)
    spectrograms.append(Sxx)

    mean_spectrogram = np.mean(spectrograms, axis=0)
    flim = [10, 200]
    inds = [np.where(frequencies <= flim[0])[0][0], np.where(frequencies >= flim[1])[0][0]]
    subplot_spect.set_ylim(10, 200)
    subplot_spect.set_ylabel("Hz")
    t_spect_mean = np.mean(times, axis=0)
    subplot_spect.pcolormesh(t_spect_mean, frequencies[inds[0]:inds[1]], mean_spectrogram[inds[0]:inds[1], :], shading='auto')
    
plt.show()