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
import pickle
import time
#%%
train_path = r"C:\Users\jeffu\Documents\Recordings\one_second_training"
test_path = r"C:\Users\jeffu\Documents\Recordings\one_second_validation"
test_set = r"C:\Users\jeffu\Documents\Recordings\test_set"
save_state_path = r"C:\Users\jeffu\Documents\Recordings\save_state.pkl"



def verify_event(y, fs, start, end):
    clear_output()
    t = np.arange(y.shape[1])/fs
    event = y[0,start:end].numpy()
    t_event = t[start:end]
   
    #plot 2 second clip and 25ms event
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(t, y[0].numpy())
    ax[0].plot(t_event, event, 'r')
    ax[1].plot(t_event, event)
    
    start_time = start/fs
    end_time = end/fs
    print(f'Clip from {start_time} to {end_time} seconds')
    plt.show()
    #While loop to allow user to replay clip
    check = 3
    while check == 3:
        sd.play(y[0,start:end].numpy(), fs)
        try:
            #Prompt user to verify potential events
            user_input = input('0: no event, 1: event, 2: False Event, 3:Replay, 9: Save & Quit')
            check = int(user_input)
            if check not in [0, 1, 2, 3,9]:
                print("Invalid input. Please enter 0, 1,2, 3, or 9")
                check = 3 # Reset check to keep the loop running
        except ValueError:
            print("Invalid input. Please enter a number.")
            check = 3  # Reset check to keep the loop running
    return check
def save_state(state, save_path = save_state_path):
    """
    Save the current state of the lists to a pickle file.
    """
    
    with open(save_path, 'wb') as f:
        pickle.dump(state, f)
    print(f"Progress saved to {save_path}. You can resume later.")

# Function to load saved progress
def load_state(save_path = save_state_path):
    """
    Load the saved lists from the pickle file, if it exists.
    """
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            state = pickle.load(f)
        print(f"Resuming from saved state in {save_path}")
    else:
        # Initialize empty lists if no save state exists
        state = {
            'file_names': [], 'start_times': [], 'end_times': [], 'auto_label': [], 'rolls': [], 'index':0
        }
        print("No saved state found. Starting fresh.")
    return state


def prepare_data(audio_path):
    '''
    Goes through all of the audio clips in a folder and prepares a dataframe. Labels are left null if above a threshold. 
    '''
   
    state = load_state()
    time.sleep(3)
    win_time = 2.0

    for k,file in enumerate(os.listdir(audio_path)):
        if file in state['file_names']:
            continue
        full_path = os.path.join(audio_path,file)
        y, fs = ta.load(full_path)
        if fs != 96000:
            resampler = ta.transforms.Resample(fs,96000)
            y = y[0,int(4*fs):].reshape(1,-1)
            y = resampler(y)
            fs = 96000
        if y.shape[0] > 1:
            y = y[1,:].reshape(1,-1)
            end = int(30*fs)
            y=y[:,:end]
        y = y/y.max()
        
        for i in np.arange(int(y.shape[1]/(fs*win_time))):
            if i < state['index']:
                continue
            state['index'] = 0
            start = int(i*(fs*win_time))
            end = int((i+1)*fs*win_time)
            check = verify_event(y,fs,start,end)
            if check == 9:
                state['index'] = i
                save_state(state)
                return
                
            elif check == 0:
                state['file_names'].append(file)
                state['start_times'].append(start)
                state['end_times'].append(end)
                state['auto_label'].append(check)
                state['rolls'].append(0)
            else:
                for j in range(0,96000,96000//40):
                    state['file_names'].append(file)
                    state['start_times'].append(start)
                    state['end_times'].append(end)
                    state['auto_label'].append(check)
                    state['rolls'].append(j)
           
    labeled_data = {'File': state['file_names'], 'Start': state['start_times'], 'End': state['end_times'], 'Roll Amount': state['rolls'], 'Label': state['auto_label']}
    df = pd.DataFrame(labeled_data)
    return df


#%% 
if __name__ == "__main__":
    df = prepare_data(test_set)
#%%
    df.loc[df.Label.eq(2),'Label'] = 0
    Pos = df[df.Label.eq(1)].shape[0]
    Neg = df[df.Label.eq(0)].shape[0]
    if  Pos > Neg:
        df2 = df[df['Label'].eq(0)].sample(Pos-Neg, replace=True)
        df1 = pd.concat([df,df2])
        df1.reset_index(drop=True,inplace=True)
        print('Adding Negatives')
    elif Neg > Pos:
        df2 = df[df['Label'].eq(1)].sample(Neg-Pos, replace=True)
        df1 = pd.concat([df,df2])
        df1.reset_index(drop=True,inplace=True)
        print('Adding Positives')
    else:
        df.reset_index(drop=True, inplace=True)
        df1=df
        print('No need to drop')

    

# %%
    df1.to_csv(r"C:\Users\jeffu\Documents\Recordings\one_second_test_set.csv",index = False)
# %%
