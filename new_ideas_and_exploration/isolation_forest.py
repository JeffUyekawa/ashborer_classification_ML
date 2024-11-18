#%%
import numpy as np
import matplotlib.pyplot as plt
import torchaudio as ta
from IPython.display import Audio, display 

AUDIO_PATH = r"C:\Users\jeffu\Documents\Recordings\new_test\2024-06-24_15_16_20.wav"

y, fs = ta.load(AUDIO_PATH)
t = np.arange(y.shape[1])/fs
# %%
from sklearn.ensemble import IsolationForest
n_fft = 256
hop_length = n_fft//4
trans = ta.transforms.Spectrogram(n_fft =n_fft, hop_length = hop_length, power = 1)
spec = trans(y)
spec = ta.transforms.AmplitudeToDB()(spec)
spec = spec.squeeze(0) #Remove channel dimension
spec = spec.numpy()
#Transpose to search for anomalous time bins
spec = spec.T #(time_bins, frequency_bins)
X = spec.reshape(-1, spec.shape[1])
iso = IsolationForest(contamination = 0.002, random_state= 13)
iso.fit(X)
anomaly_labels = iso.predict(X)


# %%
# Reshape anomaly labels to match the time bins in the spectrogram
anomaly_labels = anomaly_labels.reshape(spec.shape[0])

# Map time bins to actual time in the original audio
time_bin_size = hop_length / fs  # Time duration represented by each spectrogram time bin
anomaly_times = np.where(anomaly_labels == -1)[0] * time_bin_size  # Convert bin indices to seconds

# Plot the original audio waveform and highlight anomalies
plt.figure(figsize=(14, 6))
plt.plot(t, y[0], label='Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Signal with Anomalies Highlighted')

# Highlight the anomaly regions
for anomaly_time in anomaly_times:
    plt.axvspan(anomaly_time, anomaly_time + time_bin_size, color='red', alpha=0.5, label='Anomaly')

# Remove duplicate labels in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()
# %%
