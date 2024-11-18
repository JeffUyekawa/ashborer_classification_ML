#%%
import torch 
from torch.utils.data import Dataset
import pandas as pd
import torchaudio as ta
import os
import numpy as np 
import torch.nn.functional as F



class borer_data(Dataset):
    def __init__ (self, annotations_file, audio_dir, device = 'cpu', mel = False):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.mel = mel
        
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self,index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = ta.load(audio_sample_path)
        if sr != 96000:
            resampler = ta.transforms.Resample(sr,96000)
            start = int(4*sr)
            signal = signal[0,start:].reshape(1,-1)
            signal = resampler(signal)
            sr = 96000
        if signal.shape[0] > 1:
            signal = signal[1,:].reshape(1,-1)
        
        start = self._get_start_time(index)
        end = self._get_end_time(index)
        roll = self._get_roll(index)
        signal = signal[:,start:end]
        signal = np.roll(signal,roll)
        signal = torch.from_numpy(signal.astype('f'))
        signal = signal/signal.max()
        trans1, trans2 = self._get_spec_()
        if self.mel:
            signal = trans2(signal)
        else:
            signal = trans1(signal)
        return signal, label
            
    def _get_audio_sample_path(self,index):
        path = os.path.join(self.audio_dir,self.annotations.iloc[index,0])
        return path
    def _get_audio_sample_label(self,index):
        return self.annotations.iloc[index,4]
    def _get_spec_(self):
        return ta.transforms.Spectrogram(n_fft=1024,hop_length=1024//4, power = 1), ta.transforms.MelSpectrogram(96000, 2048, hop_length = 2048//2, n_mels = 32)
    def _get_start_time(self,index):
        return self.annotations.iloc[index,1]
    def _get_end_time(self,index):
        return self.annotations.iloc[index,2]
    def _get_roll(self,index):
        return self.annotations.iloc[index,3]
    

    