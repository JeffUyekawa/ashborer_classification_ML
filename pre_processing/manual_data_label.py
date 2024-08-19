#%%
import numpy as np
import torchaudio as ta
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import random 
import shutil
import os
import sys
from scipy.signal import find_peaks, butter, filtfilt
import time

# Insert sys path to load label_audio_events python script
sys.path.insert(1, r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\pre_processing")
train_path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train_96k"
test_path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_test_96k"

from IPython.display import clear_output

def bandpass_filter(data,fs, lowcut=1000, highcut=12000, order=5, pad = 0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b,a,data, padlen=pad)
    return y

def first_pass(y, fs, start, end):
    
    t = np.arange(len(y)) / fs
    clip = y[start:end]
    t_clip = t[start:end]
    clear_output(wait=True)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(t, y)
    ax[0].plot(t_clip, clip, 'r')
    ax[1].plot(t_clip, clip)
    plt.show()
    check = 2
    while check == 2:
        sd.play(y[start:end], fs)
        time.sleep(0.1)
        try:
            user_input = input('0: no event, 1: event, 2:replay, 9: No events remain ')
            
            check = int(user_input)
            if check not in [0, 1, 2, 9]:
                print("Invalid input. Please enter 0, 1, or 2.")
                check = 2  # Reset check to keep the loop running
        except ValueError:
            print("Invalid input. Please enter a number.")
            check = 2  # Reset check to keep the loop running
    plt.close("all")
    return check

def verify_event(y, fs, start, end, event_start, event_end):
    t = np.arange(len(y)) / fs

    clip = y[start:end]
    t_clip = t[start:end]

    event = filt[event_start:event_end]
    t_event = t[event_start:event_end]
    clear_output(wait=True)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(t_clip, clip)
    ax[0].plot(t_event, event, 'r')
    ax[1].plot(t_event, event)
    plt.show()
    time.sleep(0.1)
    check = 3
    while check == 3:
        try:
            user_input = input('0: no event, 1: Chew event, 2: False Event ')
            check = int(user_input)
            if check not in [0, 1, 2, 9]:
                print("Invalid input. Please enter 0, 1, or 2.")
                check = 3  # Reset check to keep the loop running
        except ValueError:
            print("Invalid input. Please enter a number.")
            check = 3  # Reset check to keep the loop running
    plt.close("all")
    return check

file_names = []
start_times = []
end_times = []
auto_label = []
rolls=[]
thresh = 0.19
win_time = 0.025

for k,file in enumerate(os.listdir(test_path)):
    full_path = os.path.join(test_path,file)
    y, fs = sf.read(full_path)
    y = y/np.max(np.abs(y))
    filt = y
    check = ''
    for i in np.arange(int(len(y)/(fs))):
        start = int(i*(fs))
        end = int((i+1)*fs)
        clip = np.array(filt[start:end])
        event_check = ''
        if check == 9:
            for j in range(int(len(clip)/(fs*win_time))):  
                file_names.append(file)
                jump = int(fs*win_time)
                event_start = start + j*jump
                event_end = event_start + jump
                start_times.append(event_start)
                end_times.append(event_end)
                rolls.append(0)
                auto_label.append(0)
            continue
        else:
            
            check = first_pass(filt,fs,start,end)
            if check == 0 or check == 9:
                for j in range(int(len(clip)/(fs*win_time))):
                    file_names.append(file)
                    jump = int(fs*win_time)
                    event_start = start + j*jump
                    event_end = event_start + jump
                    start_times.append(event_start)
                    end_times.append(event_end)
                    rolls.append(0)
                    auto_label.append(0)
            else:
                for j in range(int(len(clip)/(fs*win_time))):
                    jump = int(fs*win_time)
                    event_start = start + j*jump
                    event_end = event_start + jump
                    if event_check == 9:
                        file_names.append(file)
                        start_times.append(event_start)
                        end_times.append(event_end)
                        rolls.append(0)
                        auto_label.append(0)
                        continue
                    else:
                        event_check = verify_event(filt, fs,start,end, event_start,event_end)
                        if event_check == 0 or event_check ==9:
                            file_names.append(file)
                            start_times.append(event_start)
                            end_times.append(event_end)
                            rolls.append(0)
                            auto_label.append(0)
                        else:
                            if event_check == 2:
                                event_check = 0
                            for k in range(5*48+1):
                                file_names.append(file)
                                start_times.append(event_start)
                                end_times.append(event_end)
                                rolls.append(k*10)
                                auto_label.append(event_check)


#%%
import pandas as pd
labeled_data = {'File': file_names, 'Start': start_times, 'End': end_times, 'Roll Amount': rolls, 'Label': auto_label}
df = pd.DataFrame(labeled_data)
num_samps = df.shape[0]
num_pos = df[df['Label'].eq(1)].shape[0]
N = df.shape[0]-2*df[df['Label'].eq(1)].shape[0]
if N > 0:
    df1 = df.drop(df[df['Label'].eq(0)].sample(N).index)
    df1.reset_index(drop=True,inplace=True)
    print('Dropping negatives')
else:
    df.reset_index(drop=True, inplace=True)
    df1=df
    print('No need to drop')
#%%
df1.to_csv(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\Datasets\val_data_info.csv",index=False)


# %%
check_df = df[df['Roll Amount']==0]

grouped = check_df.groupby('File')


# %%
for file, group in grouped:
    full_path = os.path.join(test_path,file)
    y, fs = sf.read(full_path)
    t = np.arange(len(y))/fs
    plt.plot(t,y)
    pos_df = group[group['Label']==1]
    times = []
    for i, row in pos_df.iterrows():
        start = row['Start']
        end = row['End']
        t_clip = t[start:end]
        clip = y[start:end]
        plt.plot(t_clip,clip,'r')
    plt.show()
    

# %%
