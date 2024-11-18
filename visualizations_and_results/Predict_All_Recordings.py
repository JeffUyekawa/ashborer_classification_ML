#%%
#Import necessary packages
import numpy as np
import pandas as pd
import torch
import torchaudio as ta
import torch.nn as nn
import sys
from scipy.signal import butter, filtfilt
import os
sys.path.insert(1, r"C:\Users\jeffu\Documents\Ash Borer Project\pre_processing")
from custom_dataset_class_96k import borer_data


trans = ta.transforms.Spectrogram(n_fft = 128, hop_length=32, power = 1)

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

def prepare_y(y,fs):
    if fs != 96000:
        resampler = ta.transforms.Resample(fs,96000)
        y = resampler(y)
        fs = 96000
        y = y[0,int(5*fs):].reshape(1,-1)
        
    if y.shape[0] > 1:
        y = y[1,:int(30*fs)].reshape(1,-1)
    return y,fs

def classify_chunk(model,clip):
     spec = trans(clip.unsqueeze(0))
     pred = model(spec)
     pred = (pred >= 0.5)*1
     return pred

def make_predictions(y,fs,path, model):
    preds = []
    starts = []
    ends = []
    paths = []
    duration = int(y.shape[1]/fs)
    CHUNK = int(0.025*fs)
    num_chunks = int(duration * 1000/25)
    for i in range(num_chunks):
        start_idx = i*CHUNK
        end_idx = start_idx + CHUNK
        if end_idx > y.shape[1]:
            break
        clip = y[:,start_idx:end_idx]
        clip = clip/clip.max()
        pred = classify_chunk(model,clip)
        preds.append(pred)
        paths.append(path)
        starts.append(start_idx)
        ends.append(end_idx)
    preds_df = pd.DataFrame({'Path': paths, 'Start': starts, 'End': ends, 'Prediction': preds})
    return preds_df
            
def process_folder(folder, model):
    dfs  = []
    for file in os.listdir(folder):
        path = os.path.join(folder,file)
        y,fs = ta.load(path)
        y,fs  = prepare_y(y,fs)
        df= make_predictions(y,fs,path,model)
        dfs.append(df)
    df = pd.concat(dfs,axis=0)
    return df



#%%
#Make predictions on test set
if __name__ == '__main__':
    PARENT_PATH = r"C:\Users\jeffu\Documents\Recordings\all_recordings"
    model = CNNNetwork()
    model.load_state_dict(torch.load(r"C:\Users\jeffu\Documents\Ash Borer Project\models\Best_96k_Label_Smoothed.pt"))
    dfs = []
    for folder in os.listdir(PARENT_PATH):
        full_path = os.path.join(PARENT_PATH,folder)
        df = process_folder(full_path, model)
        dfs.append(df)
    final_df = pd.concat(dfs, axis = 0)




# %%
final_df['Prediction'] = final_df['Prediction'].values
# %%
df = final_df.copy()
# %%
df['Prediction'] = df['Prediction'].apply(lambda x: x.item())
# %%
df.to_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\visualizations\all_recording_predictions", index = False)
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\visualizations\all_recording_predictions")
# %%
import torchaudio as ta
grouped = df.groupby('Path')
paths = []
counts = []

for path, group in grouped:
    paths.append(path)
    counts.append(group['Prediction'].sum())
#%%
df_counts = pd.DataFrame({'Path':paths, 'Count':counts})    
#%%
df = df_counts.copy()
#%%
df['Date_Time_String'] = df['Path'].apply(lambda x: x.split('\\')[-1][:-4])
pattern = r'^\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}$'
# Keep only rows that match the pattern
df= df[df['Date_Time_String'].str.match(pattern)]
# %%
from datetime import datetime
df['Date_Time_Object'] = df['Date_Time_String'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d_%H_%M_%S'))
# %%
df['Date'] = df['Date_Time_Object'].apply(lambda x: x.date())
df['Hour'] = df['Date_Time_Object'].apply(lambda x: x.hour)
# %%
df.drop(columns=['Date_Time_String','Date_Time_Object'],inplace = True)
# %%
df.to_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\visualizations\all_recording_counts", index = False)
#%%
df = pd.read_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\visualizations\all_recording_counts")
df.Count.hist(bins = 100)
plt.title('Distribution of Model Positive Counts')
# %%
q1 = df.Count.quantile(0.25)
q3 = df.Count.quantile(0.75)
iqr = q3-q1
factor = 12
high = q3 + factor*iqr
low = q1 - factor*iqr

outliers = (df['Count'] > high) | (df['Count']< low)

out_df = df[outliers]
out_df = out_df.sort_values(by='Count', ascending = False)
out_df.reset_index(drop = True, inplace = True)
out_df.Count.hist()
plt.title('Outlier counts')
plt.show()
# %%
from IPython.display import display, Audio
for i, row in out_df.iterrows():
    path = row['Path']
    exts = path.split('\\')
    file = exts[-1]
    y, fs = ta.load(path)
    if fs != 96000:
        resampler = ta.transforms.Resample(96000,fs)
        fs = 96000
        y = resampler(y)
        y = y[0,:].reshape(1,-1)
    if y.shape[0] > 1:
        y = y[1,:].reshape(1,-1)

    t = np.arange(y.shape[1])/fs
    plt.plot(t,y[0].numpy())
    plt.title(f"{file}: {row['Count']}")
    plt.show()
    display(Audio(y[0].numpy(), rate = fs))
    

# %%
from IPython.display import display, Audio
clean_df = df[~outliers]
clean_df['Count'].hist()
plt.title('Dropped Outlier Counts')
plt.show()
#%%
test_set = clean_df[(clean_df['Count']>3)&(clean_df['Count']<15)]
test = test_set.sample(10, random_state = 13)
preds_df = pd.read_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\visualizations\all_recording_predictions")
for i, row in test.iterrows():
    path = row['Path']
    exts = path.split('\\')
    file = exts[-1]
    y, fs = ta.load(path)
    if fs != 96000:
        resampler = ta.transforms.Resample(fs,96000)
        y = resampler(y)
        fs = 96000
        y = y[0,int(5*fs):].reshape(1,-1)
        
    if y.shape[0] > 1:
        y = y[1,:].reshape(1,-1)

    t = np.arange(y.shape[1])/fs
    plt.plot(t,y[0].numpy())
    plt.title(f"{file}: {row['Count']}")
    preds = preds_df[(preds_df['Path']==path)&(preds_df['Prediction'])==1]
    for j, entry in preds.iterrows():
        start = entry['Start']
        end = entry['End']
        plt.plot(t[start:end],y[0,start:end].numpy(),'r')
    plt.show()
    display(Audio(y[0].numpy(), rate = fs))

# %%
clean_df = df[~outliers]
clean_df['Count'].mean()
clean_df['Count'].hist()

from scipy.stats import poisson
mu = clean_df['Count'].mean()
prob = poisson.cdf(4,mu)
print(prob)
# %%

# %%
