#%%
import pandas as pd
import numpy as np
import torchaudio as ta
import matplotlib.pyplot as plt
import os
import time
from IPython.display import display, Audio, clear_output

df = pd.read_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\data_for_validation.csv")
AUDIO_PATH = r"C:\Users\jeffu\Documents\Recordings\recordings_for_labeling"

grouped = df.groupby('File')
dfs = []
for j, (file, group) in enumerate(grouped):
    path = os.path.join(AUDIO_PATH,file)
    y, fs = ta.load(path)
    if fs != 96000:
            start = int(5*fs)
            y = y[0,start:].reshape(1,-1)
            resampler = ta.transforms.Resample(fs,96000)
            y = resampler(y)
            fs = 96000
    if y.shape[0] > 1:
        y = y[1,:].reshape(1,-1)
    t = np.arange(y.shape[1])/fs
    
    pos = group[(group['Prediction']==1)&(group['Label']==0)]
    neg = group[(group['Prediction']==0)&(group['Label']==1)]

    validation_df = pd.concat([pos,neg], axis = 0)
    dfs.append(validation_df)

final_df = pd.concat(dfs, axis = 0)

#%%
for k, (i, row) in enumerate(final_df.iterrows()):
    clear_output()
    print(f'Clip {k+1} of {final_df.shape[0]}')
    fig, ax = plt.subplots(2,2)
    start = row['Start']
    end = row['End']
    trans = ta.transforms.Spectrogram(n_fft = 128, hop_length = 32, power = 1)
    spec = trans(y[:,start:end])
    play_start = max(0,int(start - 96000))
    play_end = min(int(end + 96000), y.shape[1])
    ax[0,0].plot(t,y[0].numpy())
    ax[0,0].plot(t[start:end],y[0,start:end].numpy(),color='r')
    ax[0,0].set_title('Full 30s Audio')
    ax[0,1].plot(t[start:end],y[0,start:end].numpy(),color='r')
    ax[0,1].set_title('25ms Event')
    ax[1,1].imshow(spec[0])
    ax[1,1].set_title('Spectrogram of 25ms Event')
    ax[1,0].plot(t[play_start:play_end], y[0,play_start:play_end].numpy())
    ax[1,0].plot(t[start:end],y[0,start:end].numpy(),color = 'r')
    ax[1,0].set_title('2 Second Audio Clip Played')
    fig.tight_layout()
    plt.show()
    print(f"Original: {row['Label']}, Predicted: {row['Prediction']}")
    

    display(Audio(y[0,play_start:play_end].numpy(), rate = fs))
    time.sleep(0.25)
    check = int(input('0: no event, 1:event'))
# %%
