#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
import torchaudio as ta


 
def filter_labeled_data(in_path, out_path):
    df = pd.read_csv(in_path)
    filtered_df = df.drop_duplicates(subset=['File','Start'], keep = 'first')
    path = out_path
    filtered_df.to_csv(path, index = False)
    return path
train_path = r"C:\Users\jeffu\Documents\Recordings\time_series_training.csv"
test_path = r"C:\Users\jeffu\Documents\Recordings\time_series_test.csv"
train_temp = r"C:\Users\jeffu\Documents\Recordings\temp_path_train.csv"
test_temp = r"C:\Users\jeffu\Documents\Recordings\temp_path_test.csv"
TRAIN_ANNOTATION = filter_labeled_data(train_path, train_temp)
VAL_ANNOTATION = filter_labeled_data(test_path, test_temp)
TRAIN_AUDIO = r"C:\Users\jeffu\Documents\Recordings\new_training"
VAL_AUDIO = r"C:\Users\jeffu\Documents\Recordings\time_series_testset"

df = pd.read_csv(TRAIN_ANNOTATION)
df = df[df.Label==1]
pos_df = df.sample(n=10, random_state=11)
pos_df

fig, ax = plt.subplots(11,1, figsize = (10,30))
buff = 96//2
peak_ts = []
ax[0].set_title('Average Waveform of 10 Positive Events')
for j,(i, row) in enumerate(pos_df.iterrows()):
    audio_path = os.path.join(TRAIN_AUDIO,row['File'])
    y, fs = ta.load(audio_path)
    y = y[0].numpy()
    start = row['Start']
    end = row['End']
    y=y[start:end]
    y = y/y.max()
    max_idx = np.argmax(y)
    start = max(0,max_idx - buff)
    left = False
    right = False
    if start ==0:
        left = True
    end = min(max_idx + buff, len(y))
    if end == len(y):
        right = True
    clip = y[start:end]
    
    
    pad_len = 96-len(clip)
    zero_pad = np.zeros(pad_len)
    if left:
        clip = np.concatenate([zero_pad,clip])
    elif right:
        clip = np.concatenate([clip,zero_pad])
    t = np.arange(len(clip))/fs
    ax[j].plot(t,clip)
    peak_ts.append(clip)

peak_ts = np.array(peak_ts)
avg_ts = np.mean(peak_ts,axis=0)
ax[10].plot(t, avg_ts)
ax[10].set_title('Average Waveform')
fig.tight_layout()



plt.show()
# %%
