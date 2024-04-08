# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:09:16 2024

@author: EvanKittle
"""

import scipy
from scipy.signal import butter, iirnotch, iirpeak
from scipy import signal
from scipy.interpolate import interp1d

import numpy as np
import warnings

from matplotlib import pyplot as plt
    
def _iir_notch(data, notch_freq, q_factor, fs):
    b_notch, a_notch = iirnotch(notch_freq, q_factor, fs)
    out = signal.filtfilt(b_notch, a_notch, data)
    return out

def _iir_peak(data, notch_freq, q_factor, fs):
    b_peak, a_peak = iirpeak(notch_freq, q_factor, fs)
    out = signal.filtfilt(b_peak, a_peak, data)
    return out

def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b = butter(order, [low, high], btype="band", output="sos")
    return b

def butter_bandpass_filter(temp, lowcut, highcut, fs, order=5):   
    sos = _butter_bandpass(lowcut, highcut, fs, order)
    y = signal.sosfiltfilt(sos, temp)
    return y

def notch_out_60Hz_harmonics(x):
    for i in range(1,int(1000/60)):
        x = _iir_notch(x, 60.0*i, 60*i/2, 2000)
    return x

def noise_density(data, fs=2000, n_samples=6000, axis=None, detrend=False, window=None):
   data = np.array(data)
   if axis is None:
       if len(np.shape(data)) == 1:
           data = np.atleast_2d(data)
       if np.shape(data)[0] > np.shape(data)[1]:
           data = data.T
   elif axis == 0:
       data = data.T
   n_channels = np.shape(data)[0]
   runs = int(np.shape(data)[1] / n_samples)

   ffts = np.zeros((int(n_samples/2+1), n_channels, runs))
   for i in range(runs):
       for c in range(n_channels):
           temp = data[c, int(n_samples*i):int(n_samples*(i+1))]
           temp = temp[0:int(n_samples)]
           if detrend:
               temp = signal.detrend(temp)
           else:
               temp = temp - np.mean(temp)
           # if window is None:
           #     window = 1
           # else:
           #     window = np.hanning(window)
           ffts[:, c, i] = 2 / n_samples * abs(scipy.fft.rfft(temp))
           
   f = np.linspace(0, fs/2, int(n_samples/2+1))
           
   return f, ffts

temp = np.load(r"C:\Users\EvanKittle\Sonera Magnetics\Neuroscience Drive - Documents\Tone\2024-02-22 11-19-39 Clench Unclench Fatigue\2024-02-22 11-20-10 Clench Unclench Fatigue.npz")
data = temp['data']
         
plt.figure()
for i in range(23):
    temp = data[i,:]
    filteredtemp = butter_bandpass_filter(temp, 20, 200, 2000)
    notchedtemp = notch_out_60Hz_harmonics(filteredtemp)
    f, freqdomain = noise_density(notchedtemp)
    plt.loglog(f, np.mean(freqdomain, axis=(1, 2)), label=f'{i+1}')
    
# change plt.plot to plt.loglog to show the domain logarithmically

plt.xlabel('Frequency')
plt.ylabel('density')
plt.ylim(.000000001, .01)
plt.xlim(10, 1000)
plt.legend()
plt.show()