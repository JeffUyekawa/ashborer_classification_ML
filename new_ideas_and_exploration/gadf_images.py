#%%
import numpy as np
import matplotlib.pyplot as plt

# Function to normalize the time series to the range [-1, 1]
def normalize_time_series(ts):
    return 2 * (ts - np.min(ts)) / (np.max(ts) - np.min(ts)) - 1

# Function to compute the Gramian Angular Difference Field (GADF)
def compute_gadf(ts):
    # Normalize the time series
    normalized_ts = normalize_time_series(ts)
    
    # Transform normalized data using the angular cosine function
    phi = np.arccos(normalized_ts)
    
    # Create the GADF matrix
    gadf = np.cos(phi[:, None] + phi[None, :])
    
    return gadf

# Sample time series data
ts = np.sin(np.linspace(0, 2 * np.pi, 100))

# Compute the GADF
gadf_matrix = compute_gadf(ts)

# Plot the GADF
plt.imshow(gadf_matrix, cmap='rainbow', origin='upper')
plt.title("Gramian Angular Difference Field")
plt.colorbar()
plt.show()

# %%
import torchaudio as ta
AUDIO_PATH = r"C:\Users\jeffu\Documents\Recordings\new_test\2024-06-24_15_16_20.wav"

y, fs = ta.load(AUDIO_PATH)
t = np.arange(y.shape[1])/fs

start = int(6.615*fs)
end = int(6.625*fs)
clip = y[0,start:end].numpy()
plt.plot(t[start:end],y[0,start:end].numpy())
# %%
gadf_matrix = compute_gadf(clip)

# Plot the GADF
plt.imshow(gadf_matrix, cmap='rainbow', origin='upper')
plt.title("Gramian Angular Difference Field")
plt.colorbar()
plt.show()
# %%
start = int(23.275*fs)
end = int(23.3*fs)
clip = y[0,start:end].numpy()
plt.plot(t[start:end],y[0,start:end].numpy())
# %%
gadf_matrix = compute_gadf(clip)

# Plot the GADF
plt.imshow(gadf_matrix, cmap='rainbow', origin='upper')
plt.title("Gramian Angular Difference Field")
plt.colorbar()
plt.show()
# %%
