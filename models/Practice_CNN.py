#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

samplerate, data = read(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\wav_files\StarWars60.wav")
# %%
duration = len(data)/samplerate
time = np.arange(0,duration,1/samplerate)

plt.plot(time,data)

# %%
import torchaudio
str(torchaudio.get_audio_backend())
#%%
waveform, sample_rate = torchaudio.load(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\wav_files\StarWars60.wav")
def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
plot_waveform(waveform, sample_rate)
# %%
type(waveform)
# %%
import torchaudio as ta
from torch.utils.data import Dataset,DataLoader
import numpy as numpy
import os
import subprocess
import tqdm as tqdm
import json

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'
# %%

path=r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\practice_data\public_dataset"
for file in os.listdir(path):
    if file.endswith('webm'):
        subprocess.run(['ffmpeg','-i',os.path.join(path,file),'test_df/'+file[:-5]+'.wav'])
#%%
name_set=set()
for file in os.listdir(path):
    if file.endswith('wav'):
        name_set.add(file)
print(len(name_set))

t=os.path.join(path,list(name_set)[0])
label_path=path
fname=t[8:-4]
l=os.path.join(label_path,fname+'.json')
print(fname)
with open(l,'r') as f:
    content=json.loads(f.read())
print(content)

signal,sr=ta.load(t)
# %%
