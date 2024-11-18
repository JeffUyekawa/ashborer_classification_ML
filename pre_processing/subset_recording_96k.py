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
from scipy.signal import butter, filtfilt
from IPython.display import clear_output
# Define a function that avoids bad files, then selects a random subset of .wav recordings for the purpose of testing. 
def select_random_files(parent_folder, test_set, destination_folder, num_files, train_set = []):
    # Get a list of all files in the source folder
    files = []
    for source_folder in os.listdir(parent_folder):
        source_path = os.path.join(parent_folder,source_folder)
        files = files + [os.path.join(source_path,f) for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f)) and (f not in test_set) and (f not in train_set)]
    print(f'Total files {len(files)}')
    # Select a random subset of the files
    selected_files = random.sample(files, num_files)
    print(f'Selected {len(selected_files)} files')
    # Copy the selected files to the destination folder
    for file in selected_files:
        print(f'File: {file}')
        files = file.split('\\')
        file_name = files[-1]
        print(f'File Name: {file_name}')
        destination_file = os.path.join(destination_folder, file_name)
        shutil.copy(file, destination_file)
    
    print(f"Selected {num_files} files from {source_folder} and copied them to {destination_folder}")

parent_folder = r"C:\Users\jeffu\Documents\Recordings\all_recordings"
train_target = r"C:\Users\jeffu\Documents\Recordings\one_second_training"
test_target = r"C:\Users\jeffu\Documents\Recordings\one_second_validation"
test_set = r"C:\Users\jeffu\Documents\Recordings\test_set"
#%%
#Run cell to select random files for test
select_random_files(parent_folder,test_set,test_target, 15, train_set=train_target)
#select_random_files(path,bad_recordings,target,5, target1)
#%%
# Insert sys path to load label_audio_events python script
sys.path.insert(1, r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\pre_processing")
train_path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train_96k"
test_path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_test_96k"




def verify_event(y, fs, start, end):
    clear_output()
    buff_start = max(start - 48000, 0)
    buff_end = min(end + 48000, 2880000)

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

file_names = []
start_times = []
end_times = []
auto_label = []
rolls=[]
noise = []
win_time = 0.025
for k,file in enumerate(os.listdir(train_path)):
    full_path = os.path.join(train_path,file)
    y, fs = ta.load(full_path)
    if y.shape[0] > 1:
        y = y[0,:]
    y = y/y.max()
    thresh = np.mean(y[0,:].numpy()) + 5*np.std(y[0,:].numpy())
    for i in np.arange(int(y.shape[1]/(fs*win_time))):
        start = int(i*(fs*win_time))
        end = int((i+1)*fs*win_time)
        clip = y[0,start:end].numpy()
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

                idx = np.argmax(clip)
                right_bumper = (2300-idx)//10
                for j in range(241-13):
                    if j >= right_bumper:
                        j = j+13
                    file_names.append(file)
                    start_times.append(start)
                    end_times.append(end)
                    rolls.append(j*10)
                    auto_label.append(check)
                    noise.append(0)
    
       
        else:
            for n in range(0,-12,-1):
                file_names.append(file)
                start_times.append(start)
                end_times.append(end)
                rolls.append(0)
                auto_label.append(0)
                noise.append(n)



# %%
import pandas as pd
labeled_data = {'File': file_names, 'Start': start_times, 'End': end_times, 'Roll Amount': rolls, 'Noise': noise, 'Label': auto_label}
df = pd.DataFrame(labeled_data)
num_samps = df.shape[0]
num_pos = df[df['Label'].eq(1)].shape[0]
N = df.shape[0]-2*df[df['Label'].eq(1)].shape[0]
M = df.shape[0]-2*df[df['Label'].eq(0)].shape[0]
if N > 0:
    df2 = df[df['Label'].eq(1)].sample(N, replace=True)
    df1 = pd.concat([df,df2])
    df1.reset_index(drop=True,inplace=True)
    print('Adding Positives')
elif M > 0:
    df2 = df[df['Label'].eq(0)].sample(M, replace=True)
    df1 = pd.concat([df,df2])
    df1.reset_index(drop=True,inplace=True)
    print('Adding Negatives')
else:
    df.reset_index(drop=True, inplace=True)
    df1=df
    print('No need to drop')
#%%
df1.to_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\training_recordings_96k_filtered.csv",index=False)


