#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as ipd
import soundfile as sf

#These paths have interesting non-chewing noises that will likely trick a threshold model
path = r"C:\Users\jeffu\Documents\Recordings\05_20_2024\2024-05-16_16_55_29.wav"
#path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train\2024-05-16_11_35_39.wav"
#path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train\2024-05-19_09_06_15.wav"

#Clean
#path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train\2024-05-16_12_23_50.wav"
#path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train\2024-05-16_23_03_04.wav"
#path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train\2024-05-17_09_22_49.wav"
#path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train\2024-05-18_06_45_37.wav"
y,fs = sf.read(path)

y = y[int(5*fs):,0]
t = np.arange(len(y))/fs

plt.plot(t,y)
plt.show()

from scipy.signal import butter, filtfilt
def bandpass_filter(data,fs, lowcut=1000, highcut=17000, order=5, pad = 0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b,a,data, padlen=pad)
    return y
y=y/np.max(np.abs(y))
y = bandpass_filter(data=y,fs=fs)
plt.plot(t,y)

#threshes -0.01 < y < 0.02 ?
#filtered threshes: 0.3ish
# %% Moving Average

time_series = pd.Series(y)
smoothed_time_series1 = time_series.rolling(window=144, center=True).mean()

# %% Exponential Smoothing
from statsmodels.tsa.api import SimpleExpSmoothing
smoother = SimpleExpSmoothing(time_series)
smoothed_time_series = smoother.fit(smoothing_level=0.5, optimized=False).fittedvalues

# %%
fig, ax = plt.subplots(3,1, sharex=True, sharey=True)
ax[0].plot(t,y)
ax[0].set_title('Original Timeseries')
ax[1].plot(t,smoothed_time_series1)
ax[1].set_title('Moving Average')
ax[2].plot(t,smoothed_time_series)
ax[2].set_title('Exponential Smoothing')
fig.tight_layout()
# %%
len(y[:int(.01*fs)])
# %%
