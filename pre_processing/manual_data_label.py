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
# Define a function that avoids bad files, then selects a random subset of .wav recordings for the purpose of testing. 
def select_random_files(source_folder, bad_files, destination_folder, num_files):
    # Get a list of all files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f)) and (f not in bad_files)]
    
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Select a random subset of the files
    selected_files = random.sample(files, num_files)
    
    # Copy the selected files to the destination folder
    for file_name in selected_files:
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        shutil.copy(source_file, destination_file)
    
    print(f"Selected {num_files} files from {source_folder} and copied them to {destination_folder}")

#path=r"C:\Users\jeffu\Documents\Recordings\05_20_2024"
path = r"C:\Users\jeffu\Documents\Recordings\05_24_2024"
#List of recordings of low quality discovered with 2-means clustering

bad_recordings = ['2024-05-16_15_49_06.wav',
 '2024-05-17_02_42_21.wav',
 '2024-05-17_05_28_52.wav',
 '2024-05-17_12_43_02.wav',
 '2024-05-17_18_33_25.wav',
 '2024-05-18_01_29_23.wav',
 '2024-05-18_04_57_28.wav',
 '2024-05-19_02_59_33.wav',
 '2024-05-19_13_32_40.wav',
 '2024-05-19_17_28_28.wav']



target = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\Datasets\recordings_for_test"
target1 = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train"
#%%
#Run cell to select random files for test
#select_random_files(path,bad_recordings,target,5)
#select_random_files(path,bad_recordings,target1,20)
#%%
# Insert sys path to load label_audio_events python script
sys.path.insert(1, r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\pre_processing")
train_path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train"
test_path = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\Datasets\recordings_for_test"
from scipy.signal import butter, filtfilt
import librosa
import time
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
win_time = 0.05

for k,file in enumerate(os.listdir(test_path)):
    full_path = os.path.join(test_path,file)
    y, fs = sf.read(full_path)
    if y.shape[1] > 1:
        y = y[:,0]
    y = y/np.max(np.abs(y))
    filt = bandpass_filter(y,fs=fs)
    check = ''
    for i in np.arange(int(len(y)/(fs))):
        start = int(i*(fs))
        end = int((i+1)*fs)
        clip = np.array(filt[start:end])
        event_check = ''
        if check == 9:
            for j in range(int(len(clip)/(fs*win_time))):
                if j%5==0:
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
                    if j%5==0:
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


# %%
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
df_pos = df1[df1.Label.eq(1)]
df_neg = df1[df1.Label.eq(0)]
df_pos = df_pos[df_pos['Roll Amount'].eq(0)]
df_neg = df_neg[df_neg['Roll Amount'].eq(0)]
pos_maxes = []
neg_maxes = []


for i,row in df_pos.iterrows():
    file = row['File']
    start = row['Start']
    end = row['End']
    path = os.path.join(train_path,file)
    y, fs = sf.read(path)
    y = y[:,0]
    y = y/np.max(np.abs(y))
    y = bandpass_filter(y,fs)
    clip = y[start:end]
    pos_maxes.append(np.max(np.abs(clip)))

for i,row in df_neg.iterrows():
    file = row['File']
    start = row['Start']
    end = row['End']
    path = os.path.join(train_path,file)
    y, fs = sf.read(path)
    y = y[:,0]
    y = y/np.max(np.abs(y))
    y = bandpass_filter(y,fs)
    clip = y[start:end]
    neg_maxes.append(np.max(np.abs(clip)))

# %%
np.min(pos_maxes)
# %%
np.average(neg_maxes)
# %%
np.max(neg_maxes)
# %%
plt.hist(pos_maxes, bins= 100)
# %%
plt.hist(neg_maxes)
# %%
