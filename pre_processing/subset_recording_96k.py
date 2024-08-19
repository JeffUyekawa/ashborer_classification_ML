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
from IPython.display import clear_output
# Define a function that avoids bad files, then selects a random subset of .wav recordings for the purpose of testing. 
def select_random_files(source_folder, bad_files, destination_folder, num_files, train_set = []):
    # Get a list of all files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f)) and (f not in bad_files) and (f not in train_set)]
    
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

path=r"C:\Users\jeffu\Documents\Recordings\06_27_2024_R1"

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

target = r"C:\Users\jeffu\Documents\Recordings\recordings_for_test_96k"
target1 = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train_96k"
#%%
#Run cell to select random files for test
#select_random_files(path,bad_recordings,target,20)
select_random_files(path,bad_recordings,target,5, target1)
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


# %%
y, fs = ta.load(r"C:\Users\jeffu\Documents\Recordings\07_26_2024_LAB\2024-07-26_09_27_55.wav")
# %%
t = np.arange(y.shape[1])/fs
plt.plot(t,y[0,:].numpy())
# %%
start = int(16.975*fs)
end = int(17*fs)

clip = y[0,start:end].numpy()
t_clip = t[start:end]

plt.plot(t_clip,clip)
# %%
from sklearn.preprocessing import MinMaxScaler
trans = ta.transforms.MelSpectrogram(sample_rate = fs, n_fft = 64, hop_length= 16)
spec = y[:,start:end]/y[:,start:end].max()
spec = trans(spec)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(spec[0])
plt.imshow(spec[0])
plt.show()
plt.imshow(scaled)
# %%
for file in os.listdir(test_path):
    path = os.path.join(test_path, file)
    y, fs = sf.read(path)
    t = np.arange(len(y))/fs
    plt.plot(t,y)
    plt.title(f'Filename: {file}')
    plt.show()
# %%
