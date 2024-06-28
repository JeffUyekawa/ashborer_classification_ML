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

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy().reshape(-1,1)

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
    n_fft = 2048,
    power = 2.0,
    sample= None
):
  waveform = sample 
  win_len = n_fft//2,
  hop_len = n_fft,
  n_fft = 256  # Window size
  hop_length = n_fft // 2  # Hop length
  win_length = n_fft  # Window length
  window_fn = torch.hann_window  # Window function

  # Create the Spectrogram transform
  spectrogram = T.Spectrogram(
      n_fft=n_fft,
      hop_length=hop_length,
      win_length=win_length,
      window_fn=window_fn,
      power=2.0  # Power to which the magnitude spectrogram is scaled (1.0 for amplitude, 2.0 for power)
  )
  
  spec = spectrogram(waveform)
  spec_db = 10 * torch.log10(spec + 1e-10)
  return spec_db
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
  spec = spectrogram(waveform)
  return spectrogram(waveform)

#%%
import os
path=r"C:\Users\jeffu\Documents\Recordings\05_20_2024"

#from extractAudioEvents import extract_audio_events
bad_recordings = ['2024-05-16_15_49_06.wav',
 '2024-05-17_02_42_21.wav',
 '2024-05-17_05_28_52.wav',
 '2024-05-17_12_43_02.wav',
 '2024-05-17_18_33_25.wav',
 '2024-05-18_01_29_23.wav',
 '2024-05-18_04_57_28.wav',
 '2024-05-19_02_59_33.wav',
 '2024-05-19_13_32_40.wav',
 '2024-05-19_17_28_28.wav']

#%%
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\pre_processing")

from LabelEvents import label_audio_events

audio_dict = {}


mel_list = []
label_list = []
for j, file in enumerate(os.listdir(path)):
   
   if file not in bad_recordings:
    full_path = os.path.join(path,file)
    signal,sr = ta.load(full_path)
    signal = signal[0,:]
    for i in np.arange(int(signal.shape[0]/sr)):
        start = i*sr
        end = (i+1)*sr
        clip = signal[start:end]
        gram = get_spectrogram(sample = clip)
        plt.imshow(gram)
        mel_list.append(gram)
        if label_audio_events(clip,sr) ==0:
           label_list.append(0)
        else:
           label_list.append(1)

   

#%%
import pickle
'''audio_dict['Spectrogram'] = mel_list
audio_dict['Label'] = label_list

with open('audio_data_ver1.pickle', 'wb') as handle:
    pickle.dump(audio_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
with open('audio_data_ver1.pickle', 'rb') as handle:
    b = pickle.load(handle)
# %%

df = pd.DataFrame(b)

#%%
from time import time

from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

# see https://pytorch.org/audio/stable/transforms.html
transform=transforms.ToTensor()
path = r"C:\Users\jeffu\Documents\Recordings\05_20_2024_Images"

# Load the dataset
print(f"Loading images from dataset path")
dataset = datasets.ImageFolder(path, transform=transform)

# train / test split
val_ratio = 0.2
val_size = int(val_ratio * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
print(f"{train_size} images for training, {val_size} images for validation")
#%%
from torchvision import utils
def image_display_spectrogram(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

"""
Display all the spectrogram of sounds within a batch
@param batches: Batch of data from a dataloader 
"""
def batches_display(batches):
    dataiter = iter(batches)
    images, _ = next(dataiter)
    # create grid of images
    img_grid = utils.make_grid(images)
    # show images
    image_display_spectrogram(img_grid, one_channel=False)

batch_size = 16
NUM_WORKERS = 0
# Load training dataset into batches
train_batches = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=NUM_WORKERS)
# Load validation dataset into batches
val_batches = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size*2,
                                         num_workers=NUM_WORKERS)

# display 32 (batch_size*2) sample from the first validation batch
batches_display(val_batches)
#%%
class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=2),
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
        self.linear1=nn.Linear(in_features=49152,out_features=128)
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
      


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='AshBorercheckpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def train_model(mod,num_epochs):
    start =time()
    print('Starting Training \n ----------------------------------------')
    model = mod
    avg_train_loss = []
    avg_test_loss = []
    avg_train_acc = []
    avg_test_acc = []
    
    early_stopping = EarlyStopping(patience=1, verbose=True)
    for epoch in range(num_epochs):
        if epoch % 10 == 9 or epoch == num_epochs-1 or epoch == 0:
                print(f'Epoch {epoch+1} \n---------------------')
        model.train()
        train_loss=[]
        train_acc=[]
        for j,(inputs, labels) in enumerate(train_loader):
            y_pred = model(inputs).reshape(-1,1).float()
            guess = (y_pred>0.5)*1
            labels = labels.reshape(-1,1).float()
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 9 or epoch == num_epochs-1 or epoch == 0:
                if (j+1) % 10 == 0:
                    print(f'Step {j+1}| Loss = {loss.item():.3f}')
            with torch.no_grad():
                train_loss.append(log_loss(y_true=labels.cpu(), y_pred = y_pred.cpu(),labels=[0,1]))
                train_acc.append(accuracy_score(y_true=labels.cpu(),y_pred=guess.cpu()))
        avg_train_loss.append(np.average(train_loss))
        avg_train_acc.append(np.average(train_acc))
    #if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_acc = []
            for i, (inputs,labels) in enumerate(val_loader):
                y_pred2 = model(inputs).reshape(-1,1).float()
                guess_2 = (y_pred2>=0.5)*1
                labels = labels.reshape(-1,1).float()
                test_loss.append(log_loss(y_pred=y_pred2.cpu(),y_true=labels.cpu(),labels=[0,1]))
                test_acc.append(accuracy_score(y_true=labels.cpu(),y_pred=guess_2.cpu()))
                
            avg_test_loss.append(log_loss(y_pred=y_pred2.cpu(),y_true=labels.cpu(),labels=[0,1]))
            avg_test_acc.append(accuracy_score(y_true=labels.cpu(),y_pred=guess_2.cpu()))
        
        valid_loss = test_loss[-1]
        early_stopping(valid_loss,model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
    model.load_state_dict(torch.load('AshBorercheckpoint.pt'))
    end = time()
    print(f'Training Complete, {epoch} epochs: Time Elapsed: {(end-start)//60} minutes, {(end-start)%60} seconds')
    return model, avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc
       


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)



overallloss=[]
overallacc=[]
model = CNNNetwork()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr =.00001, weight_decay = 0.0)



train_loader = DataLoader(dataset=train_dataset,
                        batch_size=128,
                        shuffle=True,
                        num_workers=0,
                        )
val_loader = DataLoader(dataset = val_dataset,
                        batch_size=128,
                        shuffle = True,
                        num_workers = 0)
#Can change if using gpu for parallel computing)
#Training Loop (Very, Very slow, so only do 100 epochs until using HPC)



model, train_loss, test_loss, train_r2, test_r2 = train_model(model,1000)



# %%
with torch.no_grad():
    fig,axs = plt.subplots(2,2)

    axs[0][0].plot(train_loss, label= 'Training Loss')
    axs[0][0].legend()
    axs[0][1].plot(test_loss, 'r', label = 'Test Loss'  )
    axs[0][1].legend()


    axs[1][0].plot(train_r2, label = 'Train Acc')
    axs[1][0].legend()
    axs[1][1].plot(test_r2, 'r', label = 'Test Acc')
    axs[1][1].legend()
'''
    fig.savefig('1Layer.png')
    overallloss.append(np.mean(overallloss))
    overallacc.append(np.mean(overallacc))'''
# %%
