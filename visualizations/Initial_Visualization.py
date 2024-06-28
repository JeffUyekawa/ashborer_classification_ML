#%%
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
import torchaudio as ta
from scipy.signal import stft
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\pre_processing")

from LabelEvents import label_audio_events


path1=r"C:\Users\jeffu\Documents\Recordings\06_27_2024_R1\2024-06-24_18_35_57.wav"
y, fs = sf.read(path1)
y = y/np.max(np.abs(y))
sd.play(y,fs)
#%%
t = np.arange(len(y))/fs

plt.plot(t,y)

# %%
start_sec =4.0
dur_sec = 15.0
start_samp = int(np.floor(start_sec*fs))

x = y[start_samp:(start_samp+int(dur_sec*fs))]
x = x/np.max(np.abs(x))
t = np.arange(len(x))/fs
plt.plot(t,x)

sd.play(x,fs)
# %%
window = np.hamming(512)
noverlap = 256
nfft = 1024

F, T, S = spectrogram(x, fs, window = window, noverlap = noverlap, nfft = nfft)

# %%
for i in np.arange(int(x.shape[0]/fs)):
        
        start = i*fs
        end = (i+1)*fs
        clip = x[start:end]
        t = np.arange(len(clip))/fs
        if max(abs(clip)) > 0.02:
            title = "Chewing"
        else:
            title = "No Chewing"

        fig, ax = plt.subplots(1,2)
        fig.suptitle(title)
        ax[0].specgram(clip,Fs=fs,NFFT=nfft,noverlap=noverlap)
        ax[1].plot(t,clip)
        fig.tight_layout()
        
# %%
chew = x[int(8.5*fs):int(9.5*fs)]
num_clips = 10
for i in np.arange(num_clips):
     start = int(i*(fs/num_clips))
     end = int((i+1)*(fs/num_clips))
     clip = chew[start:end]
     t = np.arange(start,end)/(fs)
     plt.plot(t,clip)
     plt.show()
# %%
start = int(3.96*(fs/num_clips))
end = int(4*(fs/num_clips))
chew2 = chew[start:end]
t = np.arange(start,end)/fs
fig, ax = plt.subplots(1,2)
fig.suptitle('Chewing Event')
ax[0].specgram(chew2,Fs=fs,NFFT=nfft,noverlap=noverlap)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Frequency (Hz)')
ax[1].plot(t,chew2)
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Amplitude')
fig.tight_layout()
# %%
from matplotlib.ticker import FormatStrFormatter
start = int(8.88*fs)
end = int(8.92*fs)
tester = x[start:end]
t = np.arange(start,end)/fs

window = np.linspace(8.88,8.92,4)
fig, ax = plt.subplots(2,3)
for i, num in enumerate(window[:-1]):
    start = int(num*fs)
    end = int(window[i+1]*fs)
    clip = x[start:end]
    t = np.arange(start,end)/fs
    ax[0][i].specgram(clip,Fs=fs,NFFT=nfft,noverlap=noverlap) 
    ax[1][i].plot(t,clip)
    ax[0][i].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[1][i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
   
ax[1][1].set_xlabel('Time (s)')
ax[0][0].set_ylabel('Frequency (Hz)')
ax[1][0].set_ylabel('Amplitude')
ax[0][0].set_title('No Chewing')
ax[0][1].set_title('Chewing')
ax[0][2].set_title('No Chewing')

fig.suptitle('Comparison of Chewing and Nonchewing Events')
fig.set_figwidth(10)
fig.tight_layout()

# %%
