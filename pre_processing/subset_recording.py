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

path=r"C:\Users\jeffu\Documents\Recordings\05_20_2024"

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

target = r"C:\Users\jeffu\Documents\Recordings\recordings_for_test"
target1 = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train"
#%%
#Run cell to select random files for test
select_random_files(path,bad_recordings,target,20)
select_random_files(path,bad_recordings,target1,20)
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

file_names = []
start_times = []
end_times = []
auto_label = []
rolls=[]
thresh = 0.3
win_time = 0.01
for file in os.listdir(test_path):
    full_path = os.path.join(test_path,file)
    y, fs = sf.read(full_path)
    if y.shape[1] > 1:
        y = y[:,0]
    y = y/np.max(np.abs(y))
    y = bandpass_filter(y,fs=fs)
    for i in np.arange(int(len(y)/(fs*win_time))):
        start = int(i*(fs*win_time))
        end = int((i+1)*fs*win_time)
        clip = np.array(y[start:end])
        for j in range(49):
            new = np.roll(clip,j*10)
            file_names.append(file)
            start_times.append(start)
            end_times.append(end)
            rolls.append(j*48)
            if (np.max(np.abs(new)) > thresh):
                auto_label.append(1)
            else:
                auto_label.append(0)



# %%
import pandas as pd
labeled_data = {'File': file_names, 'Start': start_times, 'End': end_times, 'Roll Amount': rolls, 'Label': auto_label}
df = pd.DataFrame(labeled_data)
N = df.shape[0]-2*df[df['Label'].eq(1)].shape[0]
df1 = df.drop(df[df['Label'].eq(0)].sample(N).index)
df1.reset_index(drop=True,inplace=True)
#%%
df1.to_csv(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\Datasets\val_data_info.csv",index=False)

# %%
