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

path=r"C:\Users\jeffu\Documents\Recordings\07_26_2024_LAB"
bad_recordings =[]
target = r"C:\Users\jeffu\Documents\Recordings\test_set"
target1 = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train"
#%%
#Run cell to select random files for test
select_random_files(path,bad_recordings,target,2)
#%%
# Insert sys path to load label_audio_events python script
sys.path.insert(1, r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\pre_processing")
train_path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train"
test_path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_test"
from scipy.signal import butter, filtfilt
import librosa

def bandpass_filter(data,fs, lowcut=1000, highcut=12000, order=5, pad = 0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b,a,data, padlen=pad)
    return y

def verify_event(y, fs, start, end):
    buff_start = max(start - 24000, 0)
    buff_end = min(end + 24000, 1440000)

    filt = bandpass_filter(data=y, fs=fs)
    t = np.arange(len(y)) / fs

    clip = filt[buff_start:buff_end]
    t_clip = t[buff_start:buff_end]

    event = filt[start:end]
    t_event = t[start:end]
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(t_clip, clip)
    ax[0].plot(t_event, event, 'r')
    ax[1].plot(t_event, event)
    plt.show()
    check = 2
    while check == 2:
        sd.play(y[buff_start:buff_end], fs)
        try:
            user_input = input('0: no event, 1: event, 2:replay: ')
            check = int(user_input)
            if check not in [0, 1, 2]:
                print("Invalid input. Please enter 0, 1, or 2.")
                check = 2  # Reset check to keep the loop running
        except ValueError:
            print("Invalid input. Please enter a number.")
            check = 2  # Reset check to keep the loop running
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
    for i in np.arange(int(len(y)/(fs*win_time))):
        start = int(i*(fs*win_time))
        end = int((i+1)*fs*win_time)
        clip = np.array(filt[start:end])
        if np.max(np.abs(clip)) > thresh:
            check = verify_event(y,fs,start,end)
            for j in range(5*48+1):
                file_names.append(file)
                start_times.append(start)
                end_times.append(end)
                rolls.append(j*10)
                auto_label.append(check)
        else:
            if i%2==0: 
                file_names.append(file)
                start_times.append(start)
                end_times.append(end)
                rolls.append(0)
                auto_label.append(0)



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
