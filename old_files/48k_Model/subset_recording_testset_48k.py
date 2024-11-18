#%%
import numpy as np
import torch
import torchaudio as ta
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import random 
import shutil
import os
import sys
from scipy.signal import find_peaks, butter, filtfilt
from IPython.display import clear_output
test_path = r"C:\Users\jeffu\Documents\Recordings\test_set"
#%%
def bandpass_filter(data,fs, lowcut=8000, highcut=30000, order=5, pad = 0):
        nyquist = 0.5 * fs
        highcut = (fs-fs/4)//2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b,a,data, padlen=pad)
        return y
def verify_event(y, fs, start, end):
    clear_output()
    buff_start = max(start - 24000, 0)
    buff_end = min(end + 24000, 1440000)

    t = np.arange(y.shape[1]) / fs

    clip = y[0,buff_start:buff_end].numpy()
    t_clip = t[buff_start:buff_end]

    event = y[0,start:end].numpy()
    t_event = t[start:end]
   

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(t_clip, clip)
    ax[0].plot(t_event, event, 'r')
    ax[1].plot(t_event, event)
    
    start_time = start/fs
    end_time = end/fs
    print(f'Clip from {start_time} to {end_time} seconds')
    plt.show()
    check = 3
    while check == 3:
        sd.play(y[0,buff_start:buff_end].numpy(), fs)
        try:
            
            user_input = input('0: no event, 1: event, 2:False Pos, 3: Replay ')
            check = int(user_input)
            if check not in [0, 1, 2, 3]:
                print("Invalid input. Please enter 0, 1, or 2.")
                check = 3  # Reset check to keep the loop running
        except ValueError:
            print("Invalid input. Please enter a number.")
            check = 3  # Reset check to keep the loop running
    return check
#%%
file_names = []
start_times = []
end_times = []
auto_label = []
rolls=[]
noise = []
win_time = 0.05
for k,file in enumerate(os.listdir(test_path)):
    full_path = os.path.join(test_path,file)
    y, fs = ta.load(full_path)
    if y.shape[0] > 1:
        y = y[0,:].reshape(1,-1)
    if fs != 48000:
            resampler = ta.transforms.Resample(fs,48000)
            y = resampler(y)
            fs = 48000
    else:
        start = int(5*fs)
        y = y[:,start:]
    y = y/y.max()
    filt = bandpass_filter(y,fs)
    filt = torch.from_numpy(filt.astype('f'))
    thresh = np.mean(filt[0,:].numpy()) + 6*np.std(filt[0,:].numpy())
    for i in np.arange(int(y.shape[1]/(fs*win_time))):
        start = int(i*(fs*win_time))
        end = int((i+1)*fs*win_time)
        clip = filt[0,start:end].numpy()
        if np.max(clip) > thresh:
            check = verify_event(y,fs,start,end)
            if check == 0:
                file_names.append(file)
                start_times.append(start)
                end_times.append(end)
                rolls.append(0)
                auto_label.append(check)
                noise.append(0)
            else:
                if check != 1:
                    check = 0
                file_names.append(file)
                start_times.append(start)
                end_times.append(end)
                rolls.append(0)
                auto_label.append(check)
                noise.append(0)
    
       
        else:
            file_names.append(file)
            start_times.append(start)
            end_times.append(end)
            rolls.append(0)
            auto_label.append(0)
            noise.append(0)


# %%
import pandas as pd
labeled_data = {'File': file_names, 'Start': start_times, 'End': end_times, 'Roll Amount': rolls, 'Noise': noise, 'Label': auto_label}
df = pd.DataFrame(labeled_data)

#%%
df.to_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\test_set_48k",index=False)
#%%
import torchaudio as ta
import numpy as np
import matplotlib.pyplot as plt
y, fs = ta.load(r"C:\Users\jeffu\Documents\Recordings\test_set\2024-05-20_10_59_37.wav")
y = y[0,:].reshape(1,-1)
resampler = ta.transforms.Resample(fs,96000)
y_resamp = resampler(y)
fs_resamp = 96000
thresh = y.mean() + 3*y.std()


t = np.arange(y.shape[1])/fs
t_resamp = np.arange(y_resamp.shape[1])/fs_resamp

fig, ax = plt.subplots(2,1)
ax[0].plot(t,y[0].numpy())
ax[1].plot(t_resamp,y_resamp[0].numpy())
ax[1].hlines(thresh.item(),0,30)
plt.show()
# %%
from IPython.display import Audio
Audio(y_resamp[0].numpy(),rate=fs_resamp)
# %%
import torch
from IPython.display import Audio, display
for file in os.listdir(test_path):
    path = os.path.join(test_path,file)
    y,fs = ta.load(path)
    y = y/y.max()
    y = _bandpass_filter(y,fs)
    y = torch.from_numpy(y.astype('f'))
    t = np.arange(y.shape[1])/fs
    thresh = y.mean() + 6*y.std()
    plt.plot(t,y[0].numpy())
    plt.hlines(thresh.item(),0,30, color='r')
    plt.show()
    display(Audio(y[0].numpy(),rate =fs))

# %%
