#%%
import torch 
from torch.utils.data import Dataset
import pandas as pd
import torchaudio as ta
import os
import numpy as np 
import torch.nn.functional as F

class borer_data(Dataset):
    def __init__ (self, annotations_file, audio_dir, device = 'cpu', spec = False):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.spec = spec
       
        
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self,index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = ta.load(audio_sample_path)
        if sr != 96000:
            resampler = ta.transforms.Resample(sr,96000)
            start = int(5*sr)
            signal = signal[0,start:].reshape(1,-1)
            signal = resampler(signal)
            sr = 96000
        if signal.shape[0] > 1:
            signal = signal[1,:].reshape(1,-1)
        
        signal = signal/torch.max(torch.abs(signal))
        start = self._get_start_time(index)
        end = self._get_end_time(index)
        roll = self._get_roll(index)
        noise = self._get_noise(index)
        if noise < 0:
            signal = self._get_noisy(signal,noise)
        signal = signal[:,start:end]
        signal = np.roll(signal,roll)
        signal = torch.from_numpy(signal.astype('f'))
        if self.spec:
            trans1= self._get_spec_()
            signal = trans1(signal)
        return signal, label
    def _get_audio_sample_path(self,index):
        path = os.path.join(self.audio_dir,self.annotations.iloc[index,0])
        return path
    def _get_audio_sample_label(self,index):
        return self.annotations.iloc[index,5]
    def _get_spec_(self):
        return ta.transforms.Spectrogram(n_fft=128,hop_length=32, power = 1)
    def _get_start_time(self,index):
        return self.annotations.iloc[index,1]
    def _get_end_time(self,index):
        return self.annotations.iloc[index,2]
    def _get_roll(self,index):
        return self.annotations.iloc[index,3]
    def _get_noise(self,index):
        return self.annotations.iloc[index,4]
    def _get_noisy(self, y, snr):
        noise = np.random.normal(0, 1, y.shape)
        noise = torch.from_numpy(noise.astype('f'))
        desired_snr_db = snr
        audio_power = torch.mean(y**2)
        noise_power = torch.mean(noise**2)
        scaling_factor = torch.sqrt(audio_power / (10**(desired_snr_db / 10) * noise_power))
        noise = noise * scaling_factor
        noisy_y = y + noise
        return noisy_y
    
    
