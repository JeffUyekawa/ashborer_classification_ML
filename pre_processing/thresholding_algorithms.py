#%%
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import IPython.display as ipd 
import soundfile as sf
# %%
file_path = r"C:\Users\jeffu\Documents\Recordings\05_20_2024\2024-05-16_10_59_16.wav"

y, fs = sf.read(file_path)
y =y[:,0]
start = int(5*fs)
y = y[start:]
t = np.arange(len(y))/fs


plt.plot(t,y)
# %%
ipd.Audio(data=y, rate=fs)
# %%
from scipy.signal import stft

F, T, S = stft(y,fs)
# %%
amps = np.abs(S)
pows = librosa.power_to_db(amps)
librosa.display.specshow(pows, x_axis='time',y_axis='log')
# %%
nfft = 2048
window = np.kaiser(nfft, 5)
overlap = int(0.75*nfft)
plt.specgram(y, Fs=fs, NFFT = nfft,noverlap = overlap, window = window, scale='dB',)
# %%
