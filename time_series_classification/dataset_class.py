#%%
import torch 
from torch.utils.data import Dataset
import pandas as pd
import torchaudio as ta
import os
import numpy as np 
import torch.nn.functional as F
import math

class timeseries_data(Dataset):
    def __init__ (self, X, y, adjust = False):
        self.X = X
        self.y = y
        self.adjust = adjust
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self,index):
        signal = self.X[index,:,:]
        signal = torch.from_numpy(signal.astype('f'))
        label = self.y[index].astype('long')
        if self.adjust:
            label -= 1
        n_fft = self._get_nearest_power(signal.shape[1])
        hop_length = n_fft//4
        trans = ta.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power = 1)
        spec = trans(signal)
        return spec, label
    def _get_nearest_power(self, n):
        result = n // 5
        power_of_2 = 2 ** round(math.log2(result))
        return power_of_2

    
    
    
