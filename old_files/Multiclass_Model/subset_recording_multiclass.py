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
import time
import librosa 
import pickle 
import pandas as pd

# Insert sys path to load label_audio_events python script
train_path = r"C:\Users\jeffu\Documents\Recordings\new_training"
test_path = r"C:\Users\jeffu\Documents\Recordings\new_test"
train_output = r"C:\Users\jeffu\Documents\Recordings\multiclass_training_data.csv"
test_output = r"C:\Users\jeffu\Documents\Recordings\multiclass_test_data.csv"


save_path = r"C:\Users\jeffu\Documents\Recordings\current_labeling_state.pkl"
# Define a function that avoids bad files, then selects a random subset of .wav recordings for the purpose of testing. 

def save_state(df,current_index):
  state = {'df': df, 'index': current_index}
  with open(save_path, 'wb') as f:
    pickle.dump(state,f)
  print(f'Progress Saved. You can close the window and resume from clip {current_index + 1} next time.')


def load_state():
  if os.path.exists(save_path):
    with open(save_path, 'rb') as f:
      state = pickle.load(f)
    return True, state['df'], state['index']
  else:
    return False, None, None

def verify_event(audio_path, entry, total, current):
    #Prepare clip for plotting and playing
    full_path = os.path.join(audio_path, entry['File'])
    y, fs = librosa.load(full_path,sr=96000)
    if len(y.shape) > 1:
       y= y[1,:]
    y = y/np.max(y)
    start = entry['Start']
    end = entry['End']
    #create 2 second clip to play for verification
    buff_start = max(start - 96000, 0)
    buff_end = min(end + 96000, 2880000)
    t = np.arange(len(y)) / fs
    clip = y[buff_start:buff_end]
    t_clip = t[buff_start:buff_end]
    event = y[start:end]
    t_event = t[start:end]
    #While loop to allow user to replay clip
    check = 4
    while check == 4:
        try:
            #plot 2 second clip and 25ms event
            clear_output(wait=True)
            fig, ax = plt.subplots(1, 2)
            ax[0].plot(t_clip, clip)
            ax[0].plot(t_event, event, 'r')
            ax[0].set_title('Full Audio Clip')
            ax[1].plot(t_event, event)
            ax[1].set_title('Zoomed in 25ms Event')

            start_time = start/fs
            end_time = end/fs
            print(f'Clip {current+1} of {total}')
            plt.show()
            #Create play button and automatically play
            sd.play(y[buff_start:buff_end],samplerate = fs)

            #Prompt user to verify potential events
            #HTML used to fix issue with user input box being too long
            #display(HTML("<style>input { width: 100px !important; }</style>"))
            user_input = input('0: no event, 1: event, 2: False Positive, 3: Go back to previous, 4: Replay, 9: Save & Quit | ')
            check = int(user_input)

            if check not in [0, 1, 2, 3, 4, 9]:
                print("Invalid input. Please enter 0, 1, 2, 3, 4 or 9.")
                check = 4  # Reset check to keep the loop running
        except ValueError:
            print("Invalid input. Please enter a number.")
            check = 4  # Reset check to keep the loop running
    return check

def prepare_data(audio_path):
    '''
    Goes through all of the audio clips in a folder and prepares a dataframe. Labels are left null if above a threshold.
    '''
    load_check, df, index = load_state()
    if load_check:
      print('Loading dataset from last checkpoint.')
      time.sleep(2)
      return df, index
    else:
      print('Beginning to prepare the dataset.')
      file_names = []
      start_times = []
      end_times = []
      auto_label = []
      rolls=[]
      noise = []
      win_time = 0.025
      for k,file in enumerate(os.listdir(audio_path)):
          full_path = os.path.join(audio_path,file)
          y, fs = librosa.load(full_path,sr=96000)
          y = y/np.max(y)
          if file in ['2024-08-28_09_48_55.wav','2024-08-28_10_09_52.wav']:
            factor = 2.0
          else:
            factor = 5.0
          thresh = np.mean(y) + factor*np.std(y)
          for i in np.arange(int(len(y)/(fs*win_time))):
              start = int(i*(fs*win_time))
              end = int((i+1)*fs*win_time)
              clip = y[start:end]
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
                file_names.append(file)
                start_times.append(start)
                end_times.append(end)
                rolls.append(0)
                auto_label.append(0)
                noise.append(0)
      labeled_data = {'File': file_names, 'Start': start_times, 'End': end_times, 'Roll Amount': rolls, 'Noise': noise, 'Label': auto_label}
      df = pd.DataFrame(labeled_data)
      print('Dataset prepared. Ready for labeling.')
      time.sleep(2)
      return df, 0

def check_all_events(audio_path):
    df, current = prepare_data(audio_path)

    #Create a dataframe of events to check manually
    new_df = df[df['Label'].isnull()]
    #Iterate through only one instance of each potential event
    grouped = list(new_df.groupby(by=['File', 'Start']))
    i = 0
    while i  < len(grouped):
        (file, start), group = grouped[i]
        entry = group.iloc[0]
        check = verify_event(audio_path, entry, len(grouped)+current, i+current)
        # Allow option to go back to previous if mistake is made
        if check == 3:
            if i > 0:
                i -= 1
                print("Going back to the previous event")
            else:
                print("This is the first group, can't go back further.")
        elif check == 9:
          save_state(df,i+current)
          break
        else:
            # Update the DataFrame with the check value
            df.loc[(df['File'] == file) & (df['Start'] == start), 'Label'] = check
            i += 1  # Move to the next event
    if check ==9:
      pass
    else:
      print('labeling complete')
      if os.path.exists(save_path):
        print('Deleting temporary save state')
        os.remove(save_path)
    return df
#%%
if __name__ == '__main__':
   df = check_all_events(test_path)
#%%
M = df[df.Label.eq(0)].shape[0]
N = df[df.Label.eq(1)].shape[0]
K = df[df.Label.eq(2)].shape[0]
print(M, N, K)


   
# %%
#Up sample
goal = max(M,N,K)
add_df = df.copy()

for i in range(3):
   J = df[df.Label.eq(i)].shape[0]
   if J < goal:
      df1 = add_df[add_df.Label.eq(i)].sample(goal-J, random_state = 13, replace = True)
      add_df = pd.concat([add_df,df1],axis = 0)
      add_df.reset_index(inplace = True, drop = True)

# %%
M = add_df[add_df.Label.eq(0)].shape[0]
N = add_df[add_df.Label.eq(1)].shape[0]
K = add_df[add_df.Label.eq(2)].shape[0]
print(M, N, K)
# %%
add_df.to_csv(r"C:\Users\jeffu\Documents\Recordings\multiclass_test_upsampled.csv", index = False)
# %%
M = df[df.Label.eq(0)].shape[0]
N = df[df.Label.eq(1)].shape[0]
K = df[df.Label.eq(2)].shape[0]
print(M,N,K)
goal = min(M,N,K)
drop_df = df.copy()

for i in range(3):
   J = drop_df[drop_df.Label.eq(i)].shape[0]
   if J > goal:
      Q = J-goal
      drop_rows = drop_df[drop_df.Label.eq(i)].sample(Q, random_state=13)
      drop_df.drop(drop_rows.index, inplace = True)
      drop_df.reset_index(inplace = True, drop = True)


M = drop_df[drop_df.Label.eq(0)].shape[0]
N = drop_df[drop_df.Label.eq(1)].shape[0]
K = drop_df[drop_df.Label.eq(2)].shape[0]
print(M, N, K)
#%%
drop_df.to_csv(r"C:\Users\jeffu\Documents\Recordings\multiclass_test_dropped.csv", index = False)
# %%
