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
# %%
sys.path.insert(1, r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\pre_processing")
from custom_dataset_class import borer_data

model_1D = CNNNetwork_1D()
model_2D = CNNNetwork_2D()
dict_1 = r"C:\Users\jeffu\Downloads\1DAshBorercheckpoint.pt"
dict_2 = r"C:\Users\jeffu\Downloads\2DAshBorercheckpoint.pt"

model_1D.load_state_dict(torch.load(dict_1))
model_2D.load_state_dict(torch.load(dict_2))
# %%
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

model_1D.eval()
model_2D.eval()
outputs_2D = []
with torch.no_grad():
    for i, (inputs, labels) in enumerate(loader_2D):
        preds = model_2D(inputs)
        guess = (preds>0.5)*1
        outputs_2D.append(guess.item())

# %%
df['2D Predictions'] = outputs_2D
# %%
df
# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_test = df['Label']
predictions = df['2D Predictions']
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.show()
# %%
import soundfile as sf
for i, file in enumerate(os.listdir(AUDIO_DIR)):
    audio_df = df[df['File']==file]
    full_path = os.path.join(AUDIO_DIR,file)
    y, fs = sf.read(full_path)
    y = y[:,0]
    t = np.arange(len(y))/fs
    plt.plot(t,y)
    pos_df = audio_df[audio_df['2D Predictions']==1]
    for j, row in audio_df.iterrows():
        start = row['Start']
        end = row['End']
        clip = y[start:end]
        t_clip = t[start:end]
        if (row['Label']==1) and (row['2D Predictions']==0):
            plt.plot(t_clip,clip, 'black')
        elif (row['Label']==1) and (row['2D Predictions']==1):
            plt.plot(t_clip,clip,'r')
        elif (row['Label']==0) and (row['2D Predictions']==1):
            plt.plot(t_clip,clip,'orange')
    plt.title(f'Clip #{i+1}')
    plt.show()
# %%
