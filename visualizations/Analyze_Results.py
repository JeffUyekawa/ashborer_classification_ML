#%%
import numpy as np
import soundfile as sf
import sounddevice as sd
import torch
import torchaudio as ta
import time
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import sys
from scipy.signal import butter, filtfilt
sys.path.insert(1, r"C:\Users\jeffu\Documents\Ash Borer Project\pre_processing")
from custom_dataset_class_96k import borer_data
from torch.utils.data import  DataLoader

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

if __name__ == '__main__':
    AUDIO_PATH = r"C:\Users\jeffu\Documents\Recordings\test_set"
    ANNOTATION_PATH = r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\test_set.csv"
    test_data = borer_data(ANNOTATION_PATH,AUDIO_PATH,spec=True)
    data_loader = DataLoader(test_data, batch_size = 1, num_workers = 0, shuffle = False)
    preds = []
    model = CNNNetwork()
    model.load_state_dict(torch.load(r"C:\Users\jeffu\Documents\Ash Borer Project\models\Best_96k_Label_Smoothed.pt"))
    for i, (inputs,labels) in enumerate(data_loader):
        pred = model(inputs)
        pred = (pred >= 0.5)*1
        preds.append(pred.item())

# %%
import pandas as pd
df = pd.read_csv(ANNOTATION_PATH)
df = df.loc[:,['File','Start','End','Label']]
df['Prediction'] = preds
df.head()

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Assuming you have a DataFrame df with columns "labels" and "predictions"
y_true = df['Label']
y_pred = df['Prediction']

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Use ConfusionMatrixDisplay to visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

plt.show()

# %%
TN = cm[0,0]
TP = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

acc = (TP + TN)/(TP + TN + FP + FN)
prec = (TP)/(TP + FP)
recall = (TP)/(TP + FN)

print(f'Model Accuracy: {acc} | Precision: {prec} | Recall: {recall}')
# %%
from IPython.display import Audio, display
grouped = df.groupby('File')

for file, group in grouped:
    path = os.path.join(AUDIO_PATH,file)
    y, fs = ta.load(path)
    if fs != 96000:
            start = int(5*fs)
            y = y[:,start:]
            resampler = ta.transforms.Resample(fs,96000)
            y = resampler(y)
            fs = 96000
    t = np.arange(y.shape[1])/fs
    display(Audio(y[0].numpy(), rate = fs))
    plt.plot(t,y[0].numpy())

    pos = group[group['Prediction']==1]
    for i, row in pos.iterrows():
        start = row['Start']
        end = row['End']
        plt.plot(t[start:end],y[0,start:end].numpy(),color='r')

    
    plt.show()
    

# %%
