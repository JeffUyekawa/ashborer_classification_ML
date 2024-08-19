#%%
import pyaudio
import numpy as np
import soundfile as sf
import sounddevice as sd
import torch
import torchaudio as ta
from scipy.signal import butter, filtfilt
import time
import torch.nn as nn


FORMAT = None
CHANNELS = 1
RATE = 48000
CHUNK = int(RATE * 0.05) 

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
        output=self.output(logits)
        
        return output



def bandpass_filter(data,fs, lowcut=1000, highcut=12000, order=5, pad = 0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b,a,data, padlen=pad)
    return y

def chunk_to_spec(audio_chunk):
    transform = ta.transforms.MelSpectrogram(sample_rate=48000,n_fft=64,hop_length=16,f_min = 1000, f_max = 12000,n_mels=16)
    y = transform(audio_chunk)
    return y

def classify_chunk(model,audio_chunk):
    spec = chunk_to_spec(audio_chunk)
    with torch.no_grad():
        output = model(spec)
        pred =  (output>=0.5)*1
        del output, spec 
    return pred.item()

def simulate_real_time_classification(model,wav_file, duration = 30):
    audio_data, sr = ta.load(wav_file)
    y,fs = sf.read(wav_file)
    y = y/np.max(np.abs(y))
    sd.play(y,fs)
    if audio_data.shape[0] > 1:
        audio_data = audio_data[0,:].reshape(1,-1)
    
    if sr != 48000:
        resampler = ta.transforms.Resample(sr,48000)
        audio_data = resampler(audio_data)
        sr = 48000

    
    num_chunks = int(duration * 1000/50)
    times_list = []
    '''
    audio_data = audio_data/audio_data.max()
    
    audio_data = bandpass_filter(data=audio_data,fs=sr)
    
    audio_data = torch.from_numpy(audio_data.astype('f'))'''
    
    try:
        for i in range(num_chunks):
            start_idx = i*CHUNK
            end_idx = start_idx + CHUNK
            if end_idx > audio_data.shape[1]:
                break
            audio_chunk = audio_data[:,start_idx:end_idx]
            audio_chunk = audio_chunk/audio_chunk.max()
            audio_chunk = bandpass_filter(audio_chunk,sr)
            audio_chunk = torch.from_numpy(audio_chunk.astype('f'))
            audio_chunk = audio_chunk.unsqueeze(0)
            pred = classify_chunk(model,audio_chunk)
            
            if pred== 1:
                times_list.append((start_idx,end_idx))
                print(f'Event Detected at {int(start_idx/sr)} seconds')
            
            time.sleep(0.05)
    except KeyboardInterrupt:
        print('Simulation Interrupted')
    finally:
        print('Simulation Complete')
    return times_list

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    #model = torch.load(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\models\saved_2D_model.pth")
    model = CNNNetwork()
    model.load_state_dict(torch.load(r"C:\Users\jeffu\Documents\Ash Borer Project\models\Current_Best_2D.pt"))
    
    #Clip recorded at 48k
    #wav_file = r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\validation_recordings - Copy\Clip 1.wav"
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\05_20_2024\2024-05-16_16_55_29.wav"
    #Clips recorded at 96k
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\06_27_2024_R1\2024-06-24_15_02_24.wav"
    wav_file = r"C:\Users\jeffu\Documents\Recordings\06_27_2024_R1\2024-06-24_15_13_04.wav"

    #96k Forestry
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\06_28_2024_F1\2024-06-27_14_31_12.wav"


    times = simulate_real_time_classification(model,wav_file, duration=30)
    y, fs = ta.load(wav_file)
    
    if fs != 48000:
        resampler = ta.transforms.Resample(fs,48000)
        y = resampler(y)
        fs = 48000
    y = y.numpy()
    y = y[0,:]
    t = np.arange(len(y))/fs
    plt.plot(t,y)
    for i, (start,end) in enumerate(times):
        t_clip = t[start:end]
        clip = y[start:end]
        if i== 0:
            plt.plot(t_clip,clip,'r', label='Event Detected')
            
        else:
           plt.plot(t_clip,clip,'r')
    plt.legend()
    plt.show() 





# %%
import os
path = r"C:\Users\jeffu\Documents\Recordings\06_28_2024_F1"
detected = 0
false_positives = []
for file in os.listdir(path):
    wav_file = os.path.join(path,file)
    times = simulate_real_time_classification(model,wav_file,duration=30)
    detections = len(times)
    if detections > 0:
        false_positives.append(wav_file)
    print(f'{detections} events detected')
    detected = detected + detections

print(f'Total detections: {detected}')

# %%
len(os.listdir(path))
# %%
false_positives = ['C:\\Users\\jeffu\\Documents\\Recordings\\06_28_2024_F1\\2024-06-27_14_06_37.wav',
 'C:\\Users\\jeffu\\Documents\\Recordings\\06_28_2024_F1\\2024-06-27_16_53_29.wav',
 'C:\\Users\\jeffu\\Documents\\Recordings\\06_28_2024_F1\\2024-06-28_08_24_21.wav',
 'C:\\Users\\jeffu\\Documents\\Recordings\\06_28_2024_F1\\2024-06-28_09_03_31.wav',
 'C:\\Users\\jeffu\\Documents\\Recordings\\06_28_2024_F1\\2024-06-28_09_48_33.wav',
 'C:\\Users\\jeffu\\Documents\\Recordings\\06_28_2024_F1\\2024-06-28_12_55_29.wav',
 'C:\\Users\\jeffu\\Documents\\Recordings\\06_28_2024_F1\\2024-06-28_13_32_57.wav',
 'C:\\Users\\jeffu\\Documents\\Recordings\\06_28_2024_F1\\2024-06-28_13_39_09.wav']
for i, file in enumerate(false_positives):
    y, fs = sf.read(file)
    y = y/np.max(np.abs(y))
    sd.play(y,fs)
    t = np.arange(len(y))/fs
    plt.plot(t,y)
    plt.show()
    time.sleep(30)
# %%
print(np.array(false_positives)[[0,5, 16, 19,22, 27,29,30]])
# %%
