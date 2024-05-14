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


class CNNNetwork(nn.Module):

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
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=128*5*4,out_features=128)
        self.linear2=nn.Linear(in_features=128,out_features=1)
        self.output=nn.Sigmoid()
    
    def forward(self,input_data):
        x=self.conv1(input_data)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.flatten(x)
        x=self.linear1(x)
        logits=self.linear2(x)
        output=self.output(logits)
        
        return output
      
model=CNNNetwork().cuda()



def train_single_epoch(model,dataloader,loss_fn,optimizer,device):
    for waveform,label in tqdm.tqdm(dataloader):
        waveform=waveform.to(device)
        # label=pt.from_numpy(numpy.array(label))
        label=label.to(device)
        # calculate loss and preds
        logits=model(waveform)
        loss=loss_fn(logits.float(),label.float().view(-1,1))
        # backpropogate the loss and update the gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss:{loss.item()}")
    
def train(model,dataloader,loss_fn,optimizer,device,epochs):
    for i in tqdm.tqdm(range(epochs)):
        print(f"epoch:{i+1}")
        train_single_epoch(model,dataloader,loss_fn,optimizer,device)
        print('-------------------------------------------')
    print('Finished Training')

audio_path='Path Where .wav files are stored'
label_path='Path Where json files are stored'
SAMPLE_RATE=22050
NUM_SAMPLES=22050
BATCH_SIZE=128
EPOCHS=1

melspectogram=ta.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,n_fft=1024,hop_length=512,n_mels=64)
coughvid_dataset=AshBorerDataset(audio_path,label_path,melspectogram,SAMPLE_RATE,NUM_SAMPLES,device)
train_dataloader=DataLoader(coughvid_dataset,batch_size=BATCH_SIZE,shuffle=True)

loss_fn=torch.nn.BCELoss()
optimizer=torch.optim.adam(model.parameters(),lr=0.1)

train(model,train_dataloader,loss_fn,optimizer,device,EPOCHS)


waveform,label=coughvid_dataset[0]

def predict(model,inputs,labels):
    model.eval()
    inputs=torch.unsqueeze(inputs,0)
    with torch.no_grad():
        predictions=model(inputs)
    return predictions,labels
  
prediction,label=predict(model,waveform,label)
print(prediction,label)