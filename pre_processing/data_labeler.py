#%%
import numpy as np
import torchaudio as ta
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import os
import sys
from scipy.signal import butter, filtfilt
from IPython.display import clear_output
import pandas as pd
import torch
#%%
train_path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train_96k"
test_path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_test_96k"
path = r"C:\Users\jeffu\Documents\Recordings\test_set"
temp = r"C:\Users\jeffu\Documents\Recordings\test_set_labels.csv"

def bandpass_filter(data,fs, lowcut=5000, highcut=30000, order=5, pad = 0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b,a,data, padlen=pad)
    return y


def verify_event(audio_path, entry):
    clear_output()
    #Prepare clip for plotting and playing
    full_path = os.path.join(audio_path, entry['File'])
    y, fs = ta.load(full_path)
    if fs != 96000:
        resampler = ta.transforms.Resample(fs,96000)
        y = y[:,int(5*fs):]
        y = resampler(y)
        y = y[0,:].reshape(1,-1)
        fs = 96000
    if y.shape[0] > 1:
        y = y[1,:].reshape(1,-1)
        end = int(30*fs)
        y=y[:,:end]
    
    y = y/y.max()
    start = entry['Start']
    end = entry['End']
    #create 2 second clip to play for verification
    buff_start = max(start - 96000, 0)
    buff_end = min(end + 96000, 2880000)
    t = np.arange(y.shape[1]) / fs
    clip = y[0,buff_start:buff_end].numpy()
    t_clip = t[buff_start:buff_end]
    event = y[0,start:end].numpy()
    t_event = t[start:end]
   
    #plot 2 second clip and 25ms event
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(t_clip, clip)
    ax[0].plot(t_event, event, 'r')
    ax[1].plot(t_event, event)
    
    start_time = start/fs
    end_time = end/fs
    print(f'Clip from {start_time} to {end_time} seconds')
    plt.show()
    #While loop to allow user to replay clip
    check = 2
    while check == 2:
        sd.play(y[0,buff_start:buff_end].numpy(), fs)
        try:
            #Prompt user to verify potential events
            user_input = input('0: no event, 1: event, 2:Replay, 3: Go back to previous ')
            check = int(user_input)
            if check not in [0, 1, 2, 3]:
                print("Invalid input. Please enter 0, 1, 2, or 3.")
                check = 2  # Reset check to keep the loop running
        except ValueError:
            print("Invalid input. Please enter a number.")
            check = 2  # Reset check to keep the loop running
    return check

def prepare_data(audio_path):
    '''
    Goes through all of the audio clips in a folder and prepares a dataframe. Labels are left null if above a threshold. 
    '''
    file_names = []
    start_times = []
    end_times = []
    auto_label = []
    rolls=[]
    noise = []
    win_time = 0.025
    for k,file in enumerate(os.listdir(audio_path)):
        full_path = os.path.join(audio_path,file)
        y, fs = ta.load(full_path)
        if fs != 96000:
            resampler = ta.transforms.Resample(fs,96000)
            y = y[0,int(5*fs):].reshape(1,-1)
            y = resampler(y)
            fs = 96000
        if y.shape[0] > 1:
            y = y[1,:].reshape(1,-1)
            end = int(30*fs)
            y=y[:,:end]
        y = y/y.max()
        #y = bandpass_filter(y,fs)
        #y=torch.from_numpy(y.astype('f'))
        if file == '2024-07-26_09_27_55.wav':
            factor = 0.5
        elif file == '2024-07-25_09_51_20.wav':
            factor = 2.5
        else:
            factor = 6
        thresh = y.mean() + factor*y.std()
        thresh = thresh.item()
        for i in np.arange(int(y.shape[1]/(fs*win_time))):
            start = int(i*(fs*win_time))
            end = int((i+1)*fs*win_time)
            clip = y[0,start:end].numpy()
            if np.max(clip) > thresh:
                idx = np.argmax(clip)
                right_bumper = (2300-idx)//48
                for j in range(50-4):
                    if j >= right_bumper:
                        j = j+4
                    file_names.append(file)
                    start_times.append(start)
                    end_times.append(end)

                    rolls.append(j*48)
                    auto_label.append(None)
                    noise.append(0)
            else:
                for n in range(0,-12,-1):
                    file_names.append(file)
                    start_times.append(start)
                    end_times.append(end)
                    rolls.append(0)
                    auto_label.append(0)
                    noise.append(n)   
    labeled_data = {'File': file_names, 'Start': start_times, 'End': end_times, 'Roll Amount': rolls, 'Noise': noise, 'Label': auto_label}
    df = pd.DataFrame(labeled_data)
    return df

def check_all_events(df, path):
    #Create a dataframe of events to check manually
    new_df = df[df['Label'].isnull()]
    #Iterate through only one instance of each potential event 
    grouped = list(new_df.groupby(by=['File', 'Start']))
    #Initialize group index
    i = 0
    while i < len(grouped):
        (file, start), group = grouped[i]
        entry = group.iloc[0]
        check = verify_event(path, entry)
        # Allow option to go back to previous if mistake is made
        if check == 3:
            if i > 0:
                i -= 1  
                print("Going back to the previous event")
            else:
                print("This is the first group, can't go back further.")
        else:
            # Update the DataFrame with the check value
            df.loc[(df['File'] == file) & (df['Start'] == start), 'Label'] = check
            i += 1  # Move to the next event
#%% 
if __name__ == "__main__":
    df = prepare_data(path)

    check_all_events(df, path)

    num_samps = df.shape[0]
    num_pos = df[df['Label'].eq(1)].shape[0]
    N = df.shape[0]-2*df[df['Label'].eq(1)].shape[0]
    M = df.shape[0]-2*df[df['Label'].eq(0)].shape[0]
    '''if N > 0:
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
        print('No need to drop')'''

    df.to_csv(temp,index=False)
#%%
df = pd.read_csv(r"C:\Users\jeffu\Documents\Recordings\new_training_data.csv")
# %%
shared = list(df1['File'].unique())
# %%
new_df = df.drop(df[df['File'].isin(shared)].index, axis = 0)

# %%
df.shape
# %%
new_df.shape
# %%
df2 = pd.concat([new_df,df1],axis=0)
# %%
df2.shape
# %%
new_df.shape[0] + df1.shape[0]
# %%
df2.to_csv(r"C:\Users\jeffu\Documents\Recordings\new_training_data.csv", index = False)
# %%
