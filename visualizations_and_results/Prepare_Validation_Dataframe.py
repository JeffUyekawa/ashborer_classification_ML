#%%
#Import necessary packages
import numpy as np
import matplotlib.pyplot as plt 
import soundfile as sf
import sounddevice as sd
import torch
import torchaudio as ta
import time
import torch.nn as nn
import sys
from scipy.signal import butter, filtfilt
sys.path.insert(1, r"C:\Users\jeffu\Documents\Ash Borer Project\pre_processing")
from custom_dataset_class_96k import borer_data
from torch.utils.data import  DataLoader
import pandas as pd

#Define convolutional neural network
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
        self.conv5=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv6=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv7=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=2),
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
        output = self.output(logits)
       
        
        return output
def filter_labeled_data(in_path):
    df = pd.read_csv(in_path)
    filtered_df = df.drop_duplicates(subset=['File','Start'], keep = 'first')
    path = r"C:\Users\jeffu\Documents\Recordings\temp_path.csv"
    filtered_df.to_csv(path, index = False)
    return path
#Make predictions on test set
if __name__ == '__main__':
    AUDIO_PATH = r"C:\Users\jeffu\Documents\Recordings\recordings_for_labeling"
    ANNOTATION_PATH = filter_labeled_data(r"C:\Users\jeffu\Documents\Recordings\rich_labeled_data.csv")
    test_data = borer_data(ANNOTATION_PATH,AUDIO_PATH,spec=True)
    data_loader = DataLoader(test_data, batch_size = 1, num_workers = 0, shuffle = False)
    preds = []
    model = CNNNetwork()
    #model = nn.DataParallel(model)
    model.load_state_dict(torch.load(r"C:\Users\jeffu\Documents\Ash Borer Project\models\Best_96k_Label_Smoothed.pt"))
    for i, (inputs,labels) in enumerate(data_loader):
        pred = model(inputs)
        #pred = (pred >= 0.5)*1
        preds.append(pred.item())

    df = pd.read_csv(ANNOTATION_PATH)
    df = df.loc[:,['File','Start','End','Label']]
    df['Probability'] = preds
    df['Prediction'] = (df['Probability']>0.5)*1
    df.head()
    #%%
    #Save csv so predictions don't need to be made again
    df.to_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\data_for_validation.csv", index = False)

# %%
