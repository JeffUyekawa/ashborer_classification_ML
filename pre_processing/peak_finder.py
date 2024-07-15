#%%
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

#These paths have interesting non-chewing noises that will likely trick a threshold model
path = r"C:\Users\jeffu\Documents\Recordings\05_20_2024\2024-05-16_16_55_29.wav"
#path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train\2024-05-16_11_35_39.wav"
#path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train\2024-05-19_09_06_15.wav"

#Clean
#path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train\2024-05-16_12_23_50.wav"
#path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train\2024-05-16_23_03_04.wav"
#path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train\2024-05-17_09_22_49.wav"
#path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train\2024-05-18_06_45_37.wav"


# Apply band-pass filter


def bandpass_filter(data,fs, lowcut=1000, highcut=12000, order=5, pad = 0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b,a,data, padlen=pad)
    return y
def get_peak_times(peaks, window,fs):
    final = []
    buffer = int(window*fs)
    for i,peak in enumerate(peaks):
        final.append(np.arange(peak-buffer,peak+buffer))
    return final

def plot_peaks(clip, fs):
    peaks, _ = find_peaks(clip, height=0.2, distance=int(0.1 * fs))
    peak_times = get_peak_times(peaks,0.01,fs)
    t = np.arange(len(clip))/fs
    plt.plot(t, clip)
    for times in peak_times:
        plt.plot(t[times], clip[times], color = 'r')
    plt.show()

def detect_peaks(clip,fs):
    peaks, _ = find_peaks(clip, height=0.2, distance=int(0.1 * fs))
    return len(peaks)
y, fs = sf.read(path)
y = y[:,0]/np.max(np.abs(y))

clip = y[:int(6*fs)]
plot_peaks(clip,fs)
detect_peaks(clip,fs)



# %%
from scipy.signal import stft 
import librosa

path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train\2024-05-16_12_23_50.wav"
y,fs = sf.read(path)
y = y[:,0]
start = int(8.2*fs)
end = int(8.4*fs)
t = np.arange(len(y))/fs
clip = y[start:end]
t_clip = t[start:end]
nfft = 512
window = np.hamming(nfft)

plt.plot(t_clip,clip)
#%%
t,f,spec = stft(clip,fs, window='hann',nperseg = 512, noverlap = 256)
spec = librosa.power_to_db(spec)
librosa.display.specshow(spec, x_axis='time', y_axis='log', sr=fs)
plt.colorbar()

#%%
ms = librosa.feature.melspectrogram(y=y,sr=fs,window='hann', n_mels=13)
log_mel = librosa.power_to_db(ms)
librosa.display.specshow(log_mel,x_axis='time',sr=fs,y_axis='mel')


# %%
import IPython.display as ipd
ipd.Audio(path)

# %%
