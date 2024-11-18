#%%
import pandas as pd
df = pd.read_csv(r"C:\Users\jeffu\Downloads\labeled_data.csv")
df1 = pd.read_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\lab_recordings.csv")

df = pd.concat([df,df1])
# %%
import os
train_path = r"C:\Users\jeffu\Documents\Recordings\new_training"
test_path = r"C:\Users\jeffu\Documents\Recordings\new_test"

train_files = os.listdir(train_path)
test_files = os.listdir(test_path)

df_train = df[df['File'].isin(train_files)]
df_test = df[df['File'].isin(test_files)]

df_train.to_csv(r"C:\Users\jeffu\Documents\Recordings\new_training_data.csv", index = False)
df_test.to_csv(r"C:\Users\jeffu\Documents\Recordings\new_test_data.csv",index= False)
# %%
df_overlap = df_test[df_test['File'].isin(train_files)]
# %%

import torchaudio as ta
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display

y, fs = ta.load(r"C:\Users\jeffu\Documents\Recordings\08_28_2024_LAB_LargeLog\TASCAM_1582.wav")
y = y[:,:2880000]
t = np.arange(y.shape[1])/fs
thresh = y.mean() + 2*y.std()
plt.plot(t,y[1].numpy())
plt.hlines(thresh.item(),0,30,'r')
display(Audio(y[1].numpy(),rate=fs))


# %%
from scipy.signal import butter, filtfilt
import torch
def bandpass_filter(data,fs, lowcut=1000, highcut=12000, order=5, pad = 0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b,a,data, padlen=pad)
    return y
y, fs = ta.load(r"C:\Users\jeffu\Documents\Recordings\08_28_2024_LAB_LargeLog\2024-08-28_09_34_23.wav")
y = bandpass_filter(y,fs)
y= torch.from_numpy(y.astype('f'))
t = np.arange(y.shape[1])/fs
thresh = y.mean() + 4*y.std()
thresh = thresh.item()
plt.plot(t,y[0].numpy())
plt.hlines(thresh,0,30,'r')
display(Audio(y[0].numpy(),rate=fs))
# %%
y, fs = ta.load(r"C:\Users\jeffu\Documents\Recordings\08_28_2024_LAB_LargeLog\2024-08-28_10_49_35.wav")
t = np.arange(y.shape[1])/fs
thresh = y.mean() + 2*y.std()
thresh = thresh.item()
plt.plot(t,y[0].numpy())
plt.hlines(thresh,0,30,'r')
display(Audio(y[0].numpy(),rate=fs))
# %%
trans = ta.transforms.Spectrogram(n_fft = 4800, hop_length = int(0.25*4800), power = 1)
spec = trans(y)
plt.imshow(spec[0])

# %%
