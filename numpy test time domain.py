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

temp = np.load(r"C:\Users\EvanKittle\Sonera Magnetics\Neuroscience Drive - Documents\Tone\2024-02-22 11-19-39 Clench Unclench Fatigue\2024-02-22 11-20-10 Clench Unclench Fatigue.npz")
x= (np.array([]))
y= (np.array([]))
plt.plot()
plt.title("Clench Unclench Fatigue")
plt.ylabel("amp")
plt.xlabel("seconds")
data = temp['data'] 
x=np.arange(60000)/2000

for i in range (23):
    temp = data[i,:]
    filteredtemp=butter_bandpass_filter(temp, 20, 200, 2000)
    notchedtemp = notch_out_60Hz_harmonics(filteredtemp)
    plt.plot(x, notchedtemp, label=f'{i+1}')
    
plt.legend("")
plt.ylim(.015, -.015)
plt.xlim(1, 1.03)
plt.legend()
plt.show()
 











