#%%
import numpy as np
import torchaudio as ta
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import os
import sys
from scipy.signal import find_peaks, butter, filtfilt
# Define a function that avoids bad files, then selects a random subset of .wav recordings for the purpose of testing. 
#%%
# Insert sys path to load label_audio_events python script
sys.path.insert(1, r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\pre_processing")
train_path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train"
test_path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_test"
val_path = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\Datasets\validation_recordings"
from scipy.signal import butter, filtfilt
import librosa

def bandpass_filter(data,fs, lowcut=1000, highcut=12000, order=5, pad = 0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b,a,data, padlen=pad)
    return y

def verify_event(y,fs, start,end):
    buff_start = max(start - 24000,0)
    buff_end = min(end + 24000, 1440000)

    filt = bandpass_filter(data=y,fs=fs)
    t = np.arange(len(y))/fs

    clip = filt[buff_start:buff_end]
    t_clip = t[buff_start:buff_end]

    event = filt[start:end]
    t_event = t[start:end]

    plt.plot(t_clip,clip)
    plt.plot(t_event,event, 'r')
    plt.show()
    check = 2
    while check ==2:
        sd.play(y[buff_start:buff_end],fs)
        check = int(input('0: no event, 1: event, 2:replay'))
    return check

file_names = []
start_times = []
end_times = []
auto_label = []
rolls=[]
thresh = 0.2
win_time = 0.05
for i,file in enumerate(os.listdir(val_path)):
    full_path = os.path.join(val_path,file)
    y, fs = sf.read(full_path)
    if y.shape[1] > 1:
        y = y[:,0]
    y = y/np.max(np.abs(y))
    filt = bandpass_filter(y,fs=fs)
    for i in np.arange(int(len(y)/(fs*win_time))):
        start = int(i*(fs*win_time))
        end = int((i+1)*fs*win_time)
        clip = np.array(filt[start:end])
        if np.max(np.abs(clip)) > thresh:
            check = verify_event(y,fs,start,end)
            file_names.append(file)
            start_times.append(start)
            end_times.append(end)
            rolls.append(0)
            auto_label.append(check)
        else:
            file_names.append(file)
            start_times.append(start)
            end_times.append(end)
            rolls.append(0)
            auto_label.append(0)



# %%
import pandas as pd
labeled_data = {'File': file_names, 'Start': start_times, 'End': end_times, 'Roll Amount': rolls, 'Label': auto_label}
df = pd.DataFrame(labeled_data)

#%%
df.to_csv(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\Datasets\data_for_validation.csv",index=False)


# %%
