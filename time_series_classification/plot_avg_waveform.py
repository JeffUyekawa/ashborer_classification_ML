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
pos_df = df.sample(n=10, random_state=31)
pos_df

fig, ax = plt.subplots(11,1, figsize = (10,30), sharex=True)
buff = 96
peak_ts = []
ax[0].set_title('Average Waveform of 10 Positive Events')
for j,(i, row) in enumerate(pos_df.iterrows()):
    audio_path = os.path.join(TRAIN_AUDIO,row['File'])
    y, fs = ta.load(audio_path)
    if row['File'][0] == 'T':
        y = y[1].numpy()
    else:
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
    pad_len = 2*buff-len(clip)
    zero_pad = np.zeros(pad_len)
    if left:
        clip = np.concatenate([zero_pad,clip])
    elif right:
        clip = np.concatenate([clip,zero_pad])
    t = np.arange(len(clip))/fs*1000
    ax[j].plot(t,clip)
    peak_ts.append(clip)

peak_ts = np.array(peak_ts)
avg_ts = np.mean(peak_ts,axis=0)
ax[10].plot(t, avg_ts)
ax[10].set_title('Average Waveform')
ax[10].set_xlabel('Time (ms)')
fig.tight_layout()



plt.show()
# %%

audio_path = os.path.join(TRAIN_AUDIO,pos_df.iloc[9].File)
y, fs = ta.load(audio_path)
y = y[0].numpy()
start = pos_df.iloc[9].Start
end = pos_df.iloc[9].End
clip = y[start:end]
# %%
t = np.arange(len(y))/fs
plt.plot(t[start:end],clip)
# %%
start = int(19.042*fs)
end = int(19.044*fs)
plt.plot(t[start:end],y[start:end])
# %%
from IPython.display import Audio, display
Audio(y, rate = fs)
# %%
for j,(i, row) in enumerate(pos_df.iterrows()):
    audio_path = os.path.join(TRAIN_AUDIO,row['File'])
    y, fs = ta.load(audio_path)
    if row['File'][0] == 'T':
        y = y[1].numpy()
    else:
        y = y[0].numpy()
    y = y/y.max()
    start = row['Start']
    end = row['End']
    print(start)
    print(end)
    print(play_start)
    print(play_end)
    play_start = max(start - 96000, 0)
    play_end = min(end + 96000, len(y))
    clip=y[start:end]
    clip = clip/clip.max()
    max_idx = np.argmax(clip)
    start = max(0,max_idx - buff)
    left = False
    right = False
    if start ==0:
        left = True
    end = min(max_idx + buff, len(clip))
    if end == len(clip):
        right = True
    clip = clip[start:end]
    pad_len = 2*buff-len(clip)
    zero_pad = np.zeros(pad_len)
    if left:
        clip = np.concatenate([zero_pad,clip])
    elif right:
        clip = np.concatenate([clip,zero_pad])
    t = np.arange(len(clip))/fs*1000
    plt.plot(t,clip)
    plt.show()
    display(Audio(y[play_start:play_end], rate = fs))
# %%
