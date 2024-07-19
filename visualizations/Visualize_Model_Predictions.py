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

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
original_weights = model.conv1.weight.data
new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
new_conv1.weight.data = original_weights.mean(dim=1, keepdim=True)
model.conv1 = new_conv1
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

class ResNetWithSigmoid(nn.Module):
    def __init__(self, base_model):
        super(ResNetWithSigmoid, self).__init__()
        # Copy the ResNet model
        self.resnet = base_model
        
        # Define a sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass input through ResNet model
        x = self.resnet(x)
        
        # Apply sigmoid activation
        x = self.sigmoid(x)
        
        return x

sys.path.insert(1, r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\pre_processing")
from custom_dataset_class import borer_data

model_1D = CNNNetwork_1D()
model_2D = CNNNetwork_2D()
model_resnet = ResNetWithSigmoid(model)
dict_1 = r"C:\Users\jeffu\Downloads\1DAshBorercheckpoint.pt"
dict_2 = r"C:\Users\jeffu\Downloads\2DAshBorercheckpoint.pt"
dict_3 = r"C:\Users\jeffu\Downloads\ResnetAshBorercheckpoint.pt"

model_1D.load_state_dict(torch.load(dict_1))
model_2D.load_state_dict(torch.load(dict_2))
model_resnet.load_state_dict(torch.load(dict_3))

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
model_resnet.eval()
model_2D.eval()
outputs_2D = []
outputs_resnet=[]
with torch.no_grad():
    for i, (inputs, labels) in enumerate(loader_2D):
        preds = model_2D(inputs)
        guess = (preds>0.5)*1
        outputs_2D.append(guess.item())

with torch.no_grad():
    for i, (inputs, labels) in enumerate(loader_2D):
        preds = model_resnet(inputs)
        guess = (preds>0.5)*1
        outputs_resnet.append(guess.item())

# %%
df['2D Predictions'] = outputs_2D
df['Resnet Predictions'] = outputs_resnet
#%%
df.to_csv(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\visualizations\model_results.csv")
# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_test = df['Label']
predictions = df['Resnet Predictions']
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
#%%
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
    audio_df = df[df['File']==file]
    full_path = os.path.join(AUDIO_DIR,file)
    y, fs = sf.read(full_path)
    y = y[:,0]
    t = np.arange(len(y))/fs
    filt = y/np.max(np.abs(y))
    filt = bandpass_filter(filt,fs)
    display(Audio(y,rate=fs))

    plt.figure()
    fig,ax = plt.subplots(2,1)
    ax[0].plot(t,filt)
    ax[1].plot(t,filt)
    pos_df = audio_df[audio_df['2D Predictions']==1]

    for k, mod_name in zip([0,1],['2D Predictions', 'Resnet Predictions']):

        used_colors=[]

        for j, row in audio_df.iterrows():
            start = row['Start']
            end = row['End']
            clip = filt[start:end]
            t_clip = t[start:end]
            
            if (row['Label']==1) and (row[mod_name]==0):
        
                if 'black' not in used_colors:
                    ax[k].plot(t_clip,clip, 'black', label='False Negative')
                    used_colors.append('black')
                else:
                    ax[k].plot(t_clip,clip,'black')
            elif (row['Label']==1) and (row[mod_name]==1):
                if 'red' not in used_colors:
                    ax[k].plot(t_clip,clip, 'red', label='True Positive')
                    used_colors.append('red')
                else:
                    ax[k].plot(t_clip,clip,'red')
            elif (row['Label']==0) and (row[mod_name]==1):
                if 'orange' not in used_colors:
                    ax[k].plot(t_clip,clip, 'orange', label='False Positive')
                    used_colors.append('orange')
                else:
                    ax[k].plot(t_clip,clip,'orange')
            

    ax[k].set_title(f'Clip #{i+1} | {mod_name}')
    ax[k].legend()
    fig.tight_layout()
    plt.show()

# %%
path = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\Datasets\validation_recordings - Copy\Clip 2.wav"
audio_df = df[df.File=='2024-05-20_10_59_37.wav']
y, fs = sf.read(path)
y = y[:,0]
t = np.arange(len(y))/fs

start_false = int(5.77*fs)
end_false = int(5.82*fs)

true_start = int(3.85*fs)
true_end = int(3.9*fs)

t_false , y_false= t[start_false:end_false],y[start_false:end_false]
t_true, y_true = t[true_start:true_end],y[true_start:true_end]


# %%
plt.plot(t,y)
# %%
diff = np.diff(y)
diff = diff/np.max(np.abs(diff))
plt.plot(t[1:],np.abs(diff))
# %%
Audio(y,rate=fs)
# %%
