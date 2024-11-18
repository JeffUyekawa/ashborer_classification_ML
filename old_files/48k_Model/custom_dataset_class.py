#%%
import torch 
from torch.utils.data import Dataset
import pandas as pd
import torchaudio as ta
import os
from scipy.signal import butter, filtfilt
import numpy as np 



class borer_data(Dataset):
    def __init__ (self, annotations_file, audio_dir, device = 'cpu', mel_spec = False):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.mel_spec = mel_spec
        
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self,index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = ta.load(audio_sample_path)
        start = self._get_start_time(index)
        end = self._get_end_time(index)
        roll = self._get_roll(index)
        if signal.shape[0] > 1:
            signal = signal[0,:].reshape(1,-1)
        if sr != 48000:
            resampler = ta.transforms.Resample(sr,48000)
            signal = resampler(signal)
            sr = 48000
        signal = signal[:,start:end]
        signal = self._bandpass_filter(data=signal,fs=sr)
        signal = np.roll(signal,roll)
        signal = torch.from_numpy(signal.astype('f'))
        signal = signal.to(self.device)
        if self.mel_spec:
            trans = self._get_mel_spec_().to(self.device)
            signal = trans(signal)
        return signal, label
    def _get_audio_sample_path(self,index):
        path = os.path.join(self.audio_dir,self.annotations.iloc[index,0])
        return path
    def _get_audio_sample_label(self,index):
        return self.annotations.iloc[index,5]
    def _bandpass_filter(self,data,fs, lowcut=1000, highcut=12000, order=5, pad = 0):
        nyquist = 0.5 * fs
        data = data/data.max()
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b,a,data, padlen=pad)
        return y
    def _get_mel_spec_(self):
        return ta.transforms.MelSpectrogram(sample_rate=48000,n_fft=64,hop_length=16,f_min = 1000, f_max = 12000, n_mels = 16)
    def _get_start_time(self,index):
        return self.annotations.iloc[index,1]
    def _get_end_time(self,index):
        return self.annotations.iloc[index,2]
    def _get_roll(self,index):
        return self.annotations.iloc[index,3]
    
