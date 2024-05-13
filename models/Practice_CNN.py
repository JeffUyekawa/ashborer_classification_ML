#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
# %%
import torchaudio as ta
from torch.utils.data import Dataset,DataLoader
import numpy as numpy
import os
import subprocess
import tqdm as tqdm
import json

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'
# %%

path=r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\practice_data\public_dataset"

name_set=set()
for file in os.listdir(path):
    if file.endswith('wav'):
        name_set.add(file)
print(len(name_set))

t=os.path.join(path,list(name_set)[0])
label_path=path
fname=t[8:-4]
l=os.path.join(label_path,fname+'.json')
print(fname)
with open(l,'r') as f:
    content=json.loads(f.read())
print(content)

signal,sr=ta.load(t)
# %%

class AshBorerDataset(Dataset):

    def __init__(self,audio_path,label_path,transformation,target_sample_rate,num_samples,device):
        name_set=set()
        for file in os.listdir(audio_path):
            if file.endswith('wav'):
                name_set.add(file)
        name_set=list(name_set)
        self.datalist=name_set
        self.audio_path=audio_path
        self.label_path=label_path
        self.device=device
        self.transformation=transformation.to(device)
        self.target_sample_rate=target_sample_rate
        self.num_samples=num_samples
        
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self,idx):
        audio_file_path=os.path.join(self.audio_path,self.datalist[idx])
        label_file_path=os.path.join(self.label_path,self.datalist[idx][:-4]+'.json')
        with open(label_file_path,'r') as f:
            content=json.loads(f.read())
            f.close()
        label=content['cough_detected']
        waveform,sample_rate=ta.load(audio_file_path) #(num_channels,samples) -> (1,samples) makes the waveform mono
        waveform=waveform.to(self.device)
        waveform=self._resample(waveform,sample_rate)   
        waveform=self._mix_down(waveform)
        waveform=self._cut(waveform)
        waveform=self._right_pad(waveform)
        waveform=self.transformation(waveform)
        return waveform,float(label)
      
    def _resample(self,waveform,sample_rate):
        # used to handle sample rate
        resampler=ta.transforms.Resample(sample_rate,self.target_sample_rate)
        return resampler(waveform)
    
    def _mix_down(self,waveform):
        # used to handle channels
        waveform=torch.mean(waveform,dim=0,keepdim=True)
        return waveform
    
    def _cut(self,waveform):
        # cuts the waveform if it has more than certain samples
        if waveform.shape[1]>self.num_samples:
            waveform=waveform[:,:self.num_samples]
        return waveform
    
    def _right_pad(self,waveform):
        # pads the waveform if it has less than certain samples
        signal_length=waveform.shape[1]
        if signal_length<self.num_samples:
            num_padding=self.num_samples-signal_length
            last_dim_padding=(0,num_padding) # first arg for left second for right padding. Make a list of tuples for multi dim
            waveform=torch.nn.functional.pad(waveform,last_dim_padding)
        return waveform