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
from IPython.display import Audio, display
import librosa
import torchaudio.functional as F
import torchaudio.transforms as T

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'

def print_stats(waveform, sample_rate=None, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {waveform.max().item():6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()
import time
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  
  waveform = waveform.numpy()
  
  
  num_channels, num_frames = waveform.shape
 
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  

 
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=False)

def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

def inspect_file(path):
  print("-" * 10)
  print("Source:", path)
  print("-" * 10)
  print(f" - File size: {os.path.getsize(path)} bytes")
  print(f" - {ta.info(path)}")

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)

def plot_mel_fbank(fbank, title=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Filter bank')
  axs.imshow(fbank, aspect='auto')
  axs.set_ylabel('frequency bin')
  axs.set_xlabel('mel bin')
  plt.show(block=False)

def get_spectrogram(
    n_fft = 400,
    win_len = None,
    hop_len = None,
    power = 2.0,
    sample= None
):
  waveform = sample 
  spectrogram = T.Spectrogram(
      n_fft=n_fft,
      win_length=win_len,
      hop_length=hop_len,
      center=True,
      pad_mode="reflect",
      power=power,
  )
  return spectrogram(waveform)
def get_melspectrogram(
    sample= None
):
  waveform = sample 
  spectrogram = T.MelSpectrogram(
      n_fft=1024,
      hop_length=512,
      n_mels=64,
      center=True,
      pad_mode="reflect"
  )
  return spectrogram(waveform)



path=r"C:\Users\jeffu\Documents\Recordings"
name_set=set()
for file in os.listdir(path):
    if file.endswith('wav'):
        name_set.add(file)
print(len(name_set))

t=os.path.join(path,list(name_set)[0])

#%%
folder_path = "C:\\Users\\jeffu\\Documents\\Recordings"
clusterlist = ['2024-05-16_10_53_13.wav',
 '2024-05-17_08_58_36.wav',
 '2024-05-17_09_04_39.wav',
 '2024-05-17_09_16_46.wav',
 '2024-05-17_10_23_28.wav',
 '2024-05-17_10_29_31.wav',
 '2024-05-17_11_12_02.wav',
 '2024-05-17_12_12_42.wav',
 '2024-05-17_12_36_58.wav',
 '2024-05-17_13_07_19.wav',
 '2024-05-17_13_13_23.wav',
 '2024-05-17_13_25_31.wav',
 '2024-05-17_13_49_47.wav',
 '2024-05-17_14_14_03.wav',
 '2024-05-17_14_20_10.wav',
 '2024-05-17_14_32_20.wav',
 '2024-05-17_14_56_36.wav',
 '2024-05-17_15_02_40.wav',
 '2024-05-17_15_08_44.wav',
 '2024-05-17_15_14_48.wav',
 '2024-05-17_15_20_52.wav',
 '2024-05-17_15_33_01.wav',
 '2024-05-17_15_39_04.wav',
 '2024-05-17_16_03_00.wav',
 '2024-05-17_16_21_12.wav',
 '2024-05-17_16_33_20.wav',
 '2024-05-17_16_45_30.wav',
 '2024-05-17_16_51_34.wav',
 '2024-05-17_16_57_38.wav',
 '2024-05-17_17_03_42.wav',
 '2024-05-17_18_09_28.wav',
 '2024-05-18_08_21_12.wav',
 '2024-05-18_10_03_38.wav',
 '2024-05-18_10_09_42.wav',
 '2024-05-18_11_40_21.wav',
 '2024-05-18_11_46_26.wav',
 '2024-05-18_11_52_29.wav',
 '2024-05-18_11_58_34.wav',
 '2024-05-18_12_04_37.wav',
 '2024-05-18_12_10_41.wav',
 '2024-05-18_12_16_46.wav',
 '2024-05-18_12_34_57.wav',
 '2024-05-18_12_47_06.wav',
 '2024-05-18_12_53_10.wav',
 '2024-05-18_13_17_26.wav',
 '2024-05-18_13_41_42.wav',
 '2024-05-18_15_48_30.wav',
 '2024-05-18_15_54_35.wav',
 '2024-05-18_16_18_51.wav',
 '2024-05-18_16_54_58.wav',
 '2024-05-18_17_13_13.wav',
 '2024-05-18_17_31_10.wav',
 '2024-05-19_08_30_13.wav',
 '2024-05-19_08_48_03.wav',
 '2024-05-19_09_12_19.wav',
 '2024-05-19_09_18_23.wav',
 '2024-05-19_09_48_22.wav',
 '2024-05-19_10_18_23.wav',
 '2024-05-19_10_42_42.wav',
 '2024-05-19_10_54_49.wav',
 '2024-05-19_11_13_02.wav',
 '2024-05-19_11_19_06.wav',
 '2024-05-19_11_49_31.wav',
 '2024-05-19_12_01_39.wav',
 '2024-05-19_12_19_51.wav',
 '2024-05-19_12_25_55.wav',
 '2024-05-19_12_31_59.wav',
 '2024-05-19_12_50_11.wav',
 '2024-05-19_13_08_24.wav',
 '2024-05-19_13_38_44.wav',
 '2024-05-19_13_44_48.wav',
 '2024-05-19_13_56_36.wav',
 '2024-05-19_14_51_19.wav',
 '2024-05-19_15_45_42.wav',
 '2024-05-19_15_57_51.wav',
 '2024-05-19_16_10_01.wav',
 '2024-05-19_16_21_48.wav',
 '2024-05-19_17_04_12.wav',
 '2024-05-19_17_22_24.wav',
 '2024-05-19_17_34_32.wav',
 '2024-05-19_17_40_36.wav',
 '2024-05-20_10_05_19.wav']
full_paths = [os.path.join(folder_path, file) for file in clusterlist]
for item in full_paths:
  signal,sr=ta.load(item)
  signal = signal[0,:].reshape((1,signal.shape[1]))
  y_min = float(signal.min())
  y_max = float(signal.max())
  cushion = 0.1*(max([np.abs(y_min),np.abs(y_max)]))
  y_lim = (y_min - cushion, y_max + cushion)
  plot_waveform(signal,sr, title = item, ylim=y_lim)
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




#%%
#CIFAR 10 Practice
import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer


transform = transforms.Compose( # composing several transforms together
    [transforms.ToTensor(), # to tensor object
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean = 0.5, std = 0.5

# set batch_size
batch_size = 4

# set number of workers
num_workers = 0

# load train data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

# put 10 classes into a set
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
  ''' function to show image '''
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

# get random training images with iter function
dataiter = iter(trainloader)
images, labels = next(dataiter)

# call function on our images
imshow(torchvision.utils.make_grid(images))

# print the class of the image
print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))
# %%

class Net(nn.Module):
  def __init__(self):

      super(Net, self).__init__()
  # 3 input image channel, 6 output channels, 
  # 5x5 square convolution kernel
      self.conv1 = nn.Conv2d(3, 6, 5)
  # Max pooling over a (2, 2) window
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(6, 16, 5) 
      self.fc1 = nn.Linear(16 * 5 * 5, 120)# 5x5 from image dimension
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 10)

  def forward(self, x):

      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 16 * 5 * 5)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x

net = Net()
print(net)
# %%

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# %%
import time
start = time.time()

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

end = time.time()
print(f'Training Complete: {end-start} seconds')


# %%
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%s' % classes[labels[j]] for j in range(4)))
# %%


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# %%
