#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import  DataLoader
import torchaudio as ta
import os
import sys
from time import time
from sklearn.metrics import log_loss, accuracy_score

class CNNNetwork_2D(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=3072,out_features=128)
        self.linear2=nn.Linear(in_features=128,out_features=1)
        self.output=nn.Sigmoid()
    
    def forward(self,input_data):
        x=self.conv1(input_data)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.flatten(x)
        x=self.linear1(x)
        logits=self.linear2(x)
        output=self.output(logits)
        
        return output

class CNNNetwork_1D(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv5=nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=19456,out_features=64)
        self.linear2=nn.Linear(in_features=64,out_features=1)
        self.output=nn.Sigmoid()
    
    def forward(self,input_data):
        x=self.conv1(input_data)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.flatten(x)
        x=self.linear1(x)
        logits=self.linear2(x)
        output=self.output(logits)
        
        return output

sys.path.insert(1, r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\pre_processing")
from custom_dataset_class import borer_data

model_1D = CNNNetwork_1D()
model_2D = CNNNetwork_2D()
dict_1 = r"C:\Users\jeffu\Downloads\1DAshBorercheckpoint.pt"
dict_2 = r"C:\Users\jeffu\Downloads\2DAshBorercheckpoint.pt"

model_1D.load_state_dict(torch.load(dict_1))
model_2D.load_state_dict(torch.load(dict_2))

ANNOTATIONS_FILE = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\Datasets\data_for_validation.csv"
AUDIO_DIR = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\Datasets\validation_recordings"

df = pd.read_csv(ANNOTATIONS_FILE)

dataset_1D = borer_data(ANNOTATIONS_FILE,AUDIO_DIR,mel_spec=False)
dataset_2D = borer_data(ANNOTATIONS_FILE,AUDIO_DIR,mel_spec = True)

loader_1D = DataLoader(dataset_1D,
                       batch_size=1,
                       shuffle=False)
loader_2D = DataLoader(dataset_2D,
                       batch_size=1,
                       shuffle=False)
#%%
model_1D.eval()
model_2D.eval()
outputs_2D = []
outputs_1D=[]
with torch.no_grad():
    for i, (inputs, labels) in enumerate(loader_2D):
        preds = model_2D(inputs)
        guess = (preds>0.5)*1
        outputs_2D.append(guess.item())

with torch.no_grad():
    for i, (inputs, labels) in enumerate(loader_1D):
        preds = model_1D(inputs)
        guess = (preds>0.5)*1
        outputs_1D.append(guess.item())

# %%
df['2D Predictions'] = outputs_2D
df['1D Predictions'] = outputs_1D
#%%
df.to_csv(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\visualizations\model_results.csv")
# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_test = df['Label']
predictions = df['2D Predictions']
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.show()
# %%
from scipy.signal import butter, filtfilt
def bandpass_filter(data,fs, lowcut=1000, highcut=12000, order=5, pad = 0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b,a,data, padlen=pad)
    return y
df = pd.read_csv(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\visualizations\model_results.csv")
import soundfile as sf
from IPython.display import Audio, display
import sounddevice as sd
def verify_event(df):
    
    for i, row in df.iterrows():
        start = row['Start']
        end = row['End']
        file = row['File']
        label = row['Label']
        pred = row['2D Predictions']
        path = os.path.join(AUDIO_DIR,file)
        y, fs = sf.read(path)

        if label == pred:
            continue

        else:
            y, fs = sf.read(path)
            y = y[:,0]
            y = y/np.max(np.abs(y))
            y = bandpass_filter(y,fs)


        if label == 1:
        

            buff_start = max(start - 24000,0)
            buff_end = min(end + 24000, 1440000)

            filt = bandpass_filter(data=y,fs=fs)
            t = np.arange(len(y))/fs

            clip = filt[buff_start:buff_end]
            t_clip = t[buff_start:buff_end]

            event = filt[start:end]
            t_event = t[start:end]
            plt.figure()
            fig1, ax1 = plt.subplots(1,2)
            ax1[0].plot(t_clip,clip)
            ax1[0].plot(t_event,event, 'r')
            ax1[1].plot(t_event,event)
            fig1.tight_layout()
            plt.show()

            check = 2
            while check ==2:
                sd.play(filt[buff_start:buff_end],fs)
                check = int(input('0: no event, 1: event, 2:replay'))
            if check == 0:
                df.at[i,'Label'] = 0
        elif label == 0:
            buff_start = max(start - 24000,0)
            buff_end = min(end + 24000, 1440000)

            filt = bandpass_filter(data=y,fs=fs)
            t = np.arange(len(y))/fs

            clip = filt[buff_start:buff_end]
            t_clip = t[buff_start:buff_end]

            event = filt[start:end]
            t_event = t[start:end]
            plt.figure()
            fig1, ax1 = plt.subplots(1,2)
            ax1[0].plot(t_clip,clip)
            ax1[0].plot(t_event,event, 'r')
            ax1[1].plot(t_event,event)
            fig1.tight_layout()
            plt.show()

            check = 2
            while check ==2:
                sd.play(filt[buff_start:buff_end],fs)
                check = int(input('0: no event, 1: event, 2:replay'))
            if check == 1:
                df.at[i,'Label']= 1
    return df

df1 = verify_event(df)       
#%%

for i, file in enumerate(os.listdir(AUDIO_DIR)):
    audio_df = df1[df1['File']==file]
    full_path = os.path.join(AUDIO_DIR,file)
    y, fs = sf.read(full_path)
    y = y[:,0]
    t = np.arange(len(y))/fs
    filt = y/np.max(np.abs(y))
    filt = bandpass_filter(filt,fs)
    display(Audio(y,rate=fs))

    plt.figure()
    fig,ax = plt.subplots(1,1)
    ax.plot(t,filt)
    pos_df = audio_df[audio_df['2D Predictions']==1]

    used_colors=[]

    for j, row in audio_df.iterrows():
        start = row['Start']
        end = row['End']
        clip = filt[start:end]
        t_clip = t[start:end]
        
        if (row['Label']==1) and (row['2D Predictions']==0):
    
            if 'black' not in used_colors:
                ax.plot(t_clip,clip, 'black', label='False Negative')
                used_colors.append('black')
            else:
                ax.plot(t_clip,clip,'black')
        elif (row['Label']==1) and (row['2D Predictions']==1):
            if 'red' not in used_colors:
                ax.plot(t_clip,clip, 'red', label='True Positive')
                used_colors.append('red')
            else:
                ax.plot(t_clip,clip,'red')
        elif (row['Label']==0) and (row['2D Predictions']==1):
            if 'orange' not in used_colors:
                ax.plot(t_clip,clip, 'orange', label='False Positive')
                used_colors.append('orange')
            else:
                ax.plot(t_clip,clip,'orange')
            

    ax.set_title(f'Clip #{i+1} | 2D Model')
    ax.legend()
    fig.tight_layout()
    plt.show()

# %%

# %%
