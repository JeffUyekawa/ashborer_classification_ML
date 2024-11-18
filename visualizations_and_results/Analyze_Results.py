#%%
#Import necessary packages
import numpy as np
import pandas as pd
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
#%%
#Make predictions on test set
if __name__ == '__main__':
    AUDIO_PATH = r"C:\Users\jeffu\Documents\Recordings\test_set"
    ANNOTATION_PATH = filter_labeled_data(r"C:\Users\jeffu\Documents\Recordings\test_set_labels.csv")
    test_data = borer_data(ANNOTATION_PATH,AUDIO_PATH,spec=True)
    data_loader = DataLoader(test_data, batch_size = 1, num_workers = 0, shuffle = False)
    preds = []
    model = CNNNetwork()
    #model = nn.DataParallel(model)
    model.load_state_dict(torch.load(r"C:\Users\jeffu\Documents\Ash Borer Project\models\Best_96k_Label_Smoothed.pt",map_location = 'cpu'))
    for i, (inputs,labels) in enumerate(data_loader):
        pred = model(inputs)
        #pred = (pred >= 0.5)*1
        preds.append(pred.item())


# %%
#Add predictions to annotation file
import pandas as pd
df = pd.read_csv(ANNOTATION_PATH)
df = df.loc[:,['File','Start','End','Label']]
df['Probability'] = preds
df['Prediction'] = ((df['Probability']>0.5))*1
df.head()
#%%
#Save csv so predictions don't need to be made again
df.to_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\test_set_preds96k.csv", index = False)
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\test_set_preds96k.csv")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#%%
# Define true and predictions
y_true = df['Label']
y_pred = df['Prediction']

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Use ConfusionMatrixDisplay to visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Test Set Predictions Confusion Matrix')
plt.show()

# %%
from sklearn.metrics import f1_score
#Calculate accuracy, precision, and recall
TN = cm[0,0]
TP = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

acc = (TP + TN)/(TP + TN + FP + FN)
prec = (TP)/(TP + FP)
recall = (TP)/(TP + FN)
f1 = f1_score(y_true,y_pred)
print(f'Model Accuracy: {acc} | Precision: {prec} | Recall: {recall} | F1: {f1}')
# %%
import os
#AUDIO_PATH = r"C:\Users\jeffu\Documents\Recordings\test_set"
df = pd.read_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\test_set_preds96k.csv")
#Visualize model predictions
#%%
from IPython.display import Audio, display, clear_output
grouped = df.groupby('File')

for i, (file, group) in enumerate(grouped):
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
    plt.title(file)

    pos = group[(group['Prediction']==1) | (group['Label']==1)]
    colors = []
    for i, row in pos.iterrows():
        start = row['Start']
        end = row['End']
        
        if (row['Label'] == 1) & (row['Prediction']==0):
            if 'black' not in colors:
                plt.plot(t[start:end],y[0,start:end].numpy(),color='black', label ='False Negative')
                colors.append('black')
            else:
                plt.plot(t[start:end],y[0,start:end].numpy(),color='black')

        elif (row['Label'] == 0)& (row['Prediction'] == 1):
            if 'purple' not in colors:
                plt.plot(t[start:end],y[0,start:end].numpy(),color='purple', label = 'False Positive')
                colors.append('purple')
            else:
                plt.plot(t[start:end],y[0,start:end].numpy(),color='purple')
        else:
            if 'red' not in colors:
                plt.plot(t[start:end],y[0,start:end].numpy(),color='r', label = 'True Positive')
                colors.append('red')
            else:
                plt.plot(t[start:end],y[0,start:end].numpy(),color='r')

    plt.legend()
    plt.show()
    

# %%
#Verify false positives and False Negatives
import time
from IPython.display import Audio, display, clear_output
grouped = df.groupby('File')
for i, (file, group) in enumerate(grouped):
    path = os.path.join(AUDIO_PATH,file)
    y, fs = ta.load(path)
    if fs != 96000:
            start = int(5*fs)
            y = y[0,start:].reshape(1,-1)
            resampler = ta.transforms.Resample(fs,96000)
            y = resampler(y)
            fs = 96000
    t = np.arange(y.shape[1])/fs
    
    pos = group[(group['Prediction']==1)&(group['Label']==0)]
    for i, row in pos.iterrows():
        clear_output()
        fig, ax = plt.subplots(2,2)
        start = row['Start']
        end = row['End']
        trans = ta.transforms.Spectrogram(n_fft = 128, hop_length = 32, power = 1)
        spec = trans(y[:,start:end])
        play_start = max(0,int(start - 48000))
        play_end = min(int(end + 49000), y.shape[1])
        ax[0,0].plot(t,y[0].numpy())
        ax[0,0].plot(t[start:end],y[0,start:end].numpy(),color='r')
        ax[0,1].plot(t[start:end],y[0,start:end].numpy(),color='r')
        ax[1,1].imshow(spec[0])
        ax[1,0].plot(t[play_start:play_end], y[0,play_start:play_end].numpy())
        ax[1,0].plot(t[start:end],y[0,start:end].numpy(),color = 'r')
        plt.show()
        

        display(Audio(y[0,play_start:play_end].numpy(), rate = fs))
        time.sleep(0.25)
        check = int(input('0: no event, 1:event'))
        if check == 1:
            df.loc[df.index==i,'Label']=1
            

    
    

# %%
df.to_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\test_set_preds96k_adjusted.csv", index = False)
# %%
# Test if model picks up false negatives if they're centered
# 3 instances 
model = CNNNetwork()
model.load_state_dict(torch.load(r"C:\Users\jeffu\Documents\Ash Borer Project\models\Best_96k_Label_Smoothed.pt"))
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
    
    pos = group[(group['Prediction']==0)&(group['Label']==1)]
    for i, row in pos.iterrows():
        fig, ax = plt.subplots(2,2)
        start = row['Start']
        end = row['End']
        if y.shape[0] > 1:
            y = y[0].reshape(1,-1)
        clip = y[:,start:end]
        clip = clip/clip.max()
        print(clip.shape)
        mid = start + torch.argmax(clip).item()
        buff = int(0.025*fs//2)
        start = mid - buff
        end = mid + buff
        clip = y[:,start:end]
        clip = clip/clip.max()
        print(clip.shape)
        clip = clip.unsqueeze(0)
        

        trans = ta.transforms.Spectrogram(n_fft = 128, hop_length = 32, power = 1)
        spec = trans(clip)
        pred = model(spec)
        print(pred.item())
        play_start = max(0,int(start - 48000))
        play_end = min(int(end + 48000), y.shape[1])
        ax[0,0].plot(t,y[0].numpy())
        ax[0,0].plot(t[start:end],y[0,start:end].numpy(),color='r')
        ax[0,1].plot(t[start:end],y[0,start:end].numpy(),color='r')
        ax[1,1].imshow(spec.squeeze())
        ax[1,0].plot(t[play_start:play_end], y[0,play_start:play_end].numpy())
        ax[1,0].plot(t[start:end],y[0,start:end].numpy(),color = 'r')
        plt.show()
        time.sleep(0.25)
# %%
import matplotlib.pyplot as plt
false_pos = df[(df['Label'] == 0) & (df['Prediction'] == 1)]
true_pos = df[(df['Label'] == 1) & (df['Prediction'] == 1)]

false_pos['Probability'].hist(label = 'false positive', bins = len(false_pos))
true_pos['Probability'].hist(label = 'true positive', alpha = 0.5, bins = len(true_pos))
plt.legend()
plt.show()
# %%

test = df[df['File'] == "2024-05-20_10_59_37.wav"]

false_pos = test[(test['Label']==0) & (test['Prediction']==1)]

y, fs = ta.load(path)
y = y[:,int(5*fs):]
resampler = ta.transforms.Resample(fs,96000)
y = resampler(y)
fs = 96000

t = np.arange(y.shape[1])/fs

plt.plot(t,y[0].numpy())

# %%
y, fs = ta.load(r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\test.wav")
y = y/y.max()
plt.plot(y[0].numpy())
# %%
Audio(y[0].numpy(), rate = fs)
# %%
import pandas as pd
df = pd.read_csv(r"C:\Users\jeffu\Documents\Recordings\new_training_data.csv")

# %%
df.head()

# %%
examples = df[df['Label']==1].sample(20)

# %%
import matplotlib.pyplot as plt
path = r"C:\Users\jeffu\Documents\Recordings\new_training"
trans = ta.transforms.Spectrogram(n_fft = 128, hop_length=32, power =1)
for i, row in examples.iterrows():
    start = row['Start']
    end = row['End']
    file = row['File']
    roll = row['Roll Amount']
    full_path = os.path.join(path, file)
    y, fs = ta.load(full_path)
    t = np.arange(y.shape[1])/fs
    if y.shape[0] > 1:
        y = y[1,:].reshape(1,-1)
    t_clip = t[start:end]
    clip = y[:,start:end]
    clip = np.roll(clip,roll)
    clip = torch.from_numpy(clip.astype('f'))
    spec = trans(clip)
    plt.imshow(spec[0])
    plt.show()

# %%
bad_files = ['2024-06-24_17_35_48.wav']
test = df[df['File'].isin(bad_files)]
test = test[test['Label']==1]
test
# %%
import pandas as pd
df_train = pd.read_csv(r"C:\Users\jeffu\Documents\Recordings\new_training_data.csv")
df_test = pd.read_csv(r"C:\Users\jeffu\Documents\Recordings\new_test_data.csv")

df_train = df_train.drop_duplicates()
df_test = df_test.drop_duplicates()
# %%
df_train.shape
# %%
df_test.shape
# %%
df_train.to_csv(r"C:\Users\jeffu\Documents\Recordings\imbalanced_training_data.csv", index = False)
df_test.to_csv(r"C:\Users\jeffu\Documents\Recordings\imbalanced_test_data.csv", index = False)

# %%
df = pd.read_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\test_set_preds96k_adjusted.csv")
false_pos = df[(df['Prediction']==1) & (df['Label']==0) & (df['File']=='2024-06-24_15_20_39.wav')]
sample = false_pos.sample(n=false_pos.shape[0])
sample.head()

# %%
trans = ta.transforms.Spectrogram(n_fft = 128, hop_length = 32, power = 1)
for i, row in sample.iterrows():
    start = row['Start']
    end = row['End']
    buff_start = max(start - 96000,0)
    buff_end = min(end + 96000, int(30*96000))
    file = row['File']
    full_path = os.path.join(AUDIO_PATH,file)
    y,fs = ta.load(full_path)
    if fs != 96000:
        resampler = ta.transforms.Resample(fs,96000)
        y = resampler(y)
        y = y[0].reshape(1,-1)
        fs = 96000
    t = np.arange(y.shape[1])/fs
    clip = y[:,start:end]
    t_clip = t[start:end]
    audio = y[0,buff_start:buff_end].numpy()
    display(Audio(audio, rate = fs))
    plt.plot(t[buff_start:buff_end], y[0,buff_start:buff_end].numpy())
    plt.plot(t_clip,clip[0].numpy())
    plt.show()
    spec = trans(clip)
    plt.imshow(spec[0])
    plt.show()
    
# %%
true_pos = df[df['Label']==1]
sample = true_pos.sample(10, random_state=13)
for i, row in sample.iterrows():
    start = row['Start']
    end = row['End']
    buff_start = max(start - 96000,0)
    buff_end = min(end + 96000, int(30*96000))
    file = row['File']
    full_path = os.path.join(AUDIO_PATH,file)
    y,fs = ta.load(full_path)
    if fs != 96000:
        resampler = ta.transforms.Resample(fs,96000)
        y = resampler(y)
        y = y[0].reshape(1,-1)
        fs = 96000
    t = np.arange(y.shape[1])/fs
    clip = y[:,start:end]
    t_clip = t[start:end]
    audio = y[0,buff_start:buff_end].numpy()
    display(Audio(audio, rate = fs))
    plt.plot(t[buff_start:buff_end], y[0,buff_start:buff_end].numpy())
    plt.plot(t_clip,clip[0].numpy(),'r')
    plt.show()
    spec = trans(clip)
    plt.imshow(spec[0])
    plt.show()

# %%
df = pd.read_csv(r"C:\Users\jeffu\Documents\Recordings\multiclass_test_data.csv")
# %%
df.shape
# %%
df.head()

# %%
df.File.unique().shape
# %%
