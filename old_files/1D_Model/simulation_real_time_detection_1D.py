#%%
import pyaudio
import numpy as np
import soundfile as sf
import sounddevice as sd
import torch
import torchaudio as ta
import time
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt


FORMAT = None
CHANNELS = 1
RATE = 96000
CHUNK = int(RATE * 0.025) 

class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv5=nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
       
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=19456,out_features=128)
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
    
def bandpass_filter(data,fs, lowcut=8000, highcut=30000, order=5, pad = 0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b,a,data, padlen=pad)
    return y


def classify_chunk(model,audio_chunk, fs=96000):
    spec = bandpass_filter(audio_chunk,fs)
    spec = torch.from_numpy(spec.astype('f'))
    with torch.no_grad():
        output = model(spec)        
        pred =  (output>=0.5)*1
        del output, spec 
    return pred.item()

def simulate_real_time_classification(model,wav_file, duration = 30, live = True):
    audio_data, sr = ta.load(wav_file)
    y,fs = sf.read(wav_file)
    y = y/np.max(np.abs(y))
    if live:
        sd.play(y,fs)
    if audio_data.shape[0] > 1:
        audio_data = audio_data[1,:].reshape(1,-1)
    if sr != 96000:
        resampler = ta.transforms.Resample(sr,96000)
        start = int(5*sr)
        audio_data = audio_data[:,start:]
        audio_data = resampler(audio_data)
        duration = 25
        sr = 96000
    num_chunks = int(duration * 1000/25)
    times_list = []
    if live:
        try:
            for i in range(num_chunks):
                start_idx = i*CHUNK
                end_idx = start_idx + CHUNK
                if end_idx > audio_data.shape[1]:
                    break
                audio_chunk = audio_data[:,start_idx:end_idx]
                audio_chunk = audio_chunk/audio_chunk.max()
            
                audio_chunk = audio_chunk.unsqueeze(0)
                pred = classify_chunk(model,audio_chunk)
                
                if pred== 1:
                    times_list.append((start_idx,end_idx))
                    print(f'Event Detected at {int(start_idx/sr)} seconds')
                
                time.sleep(0.025)
        except KeyboardInterrupt:
            print('Simulation Interrupted')
        finally:
            print('Simulation Complete')
    else:
        try:
            for i in range(num_chunks):
                start_idx = i*CHUNK
                end_idx = start_idx + CHUNK
                if end_idx > audio_data.shape[1]:
                    break
                audio_chunk = audio_data[:,start_idx:end_idx]
                audio_chunk = audio_chunk/audio_chunk.max()
            
                audio_chunk = audio_chunk.unsqueeze(0)
                pred = classify_chunk(model,audio_chunk)
                
                if pred== 1:
                    times_list.append((start_idx,end_idx))
        finally:
            pass
        
    return times_list

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    #model = torch.load(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\models\saved_1d_model.pth")
    model = CNNNetwork()
    #model = nn.DataParallel(model)
    #model.load_state_dict(torch.load(r"C:\Users\jeffu\Documents\Ash Borer Project\models\Best_96k_Label_Smoothed.pt"))
    model.load_state_dict(torch.load(r"C:\Users\jeffu\Downloads\1DAshBorercheckpoint (2).pt", map_location='cpu'))
    #Clip recorded at 48k
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\07_25_2024_LAB\2024-07-25_09_51_20.wav"
    #wav_file = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\Datasets\validation_recordings - Copy\Clip 2.wav"
    #wav_file = r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\validation_recordings - Copy\Clip 1.wav"

    #Clips recorded at 96k
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\06_27_2024_R1\2024-06-24_15_02_24.wav"
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\06_27_2024_R1\2024-06-24_15_13_04.wav"
    #wav_file = r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\recordings_for_test\2024-05-20_11_53_58.wav"
    #wav_file = r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\recordings_for_test\2024-05-21_16_25_40.wav"

    #96k Forestry
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\06_28_2024_F1\2024-06-27_14_31_12.wav"
    #wav_file = r'C:\\Users\\jeffu\\Documents\\Recordings\\06_28_2024_F1\\2024-06-28_09_48_33.wav'

    #96k Lab Recordings
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\07_26_2024_LAB\2024-07-26_09_49_11.wav"
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\07_26_2024_LAB\2024-07-26_09_27_55.wav"
    #wav_file = r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\test.wav"

    #Lab recording with knocking
    wav_file = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train_96k\2024-07-26_09_27_55.wav"
    #noisy lab recording
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train_96k\2024-07-25_10_05_28.wav"
    times = simulate_real_time_classification(model,wav_file, duration=30, live = False)
    y, fs = ta.load(wav_file)
    
    if fs != 96000:
        resampler = ta.transforms.Resample(fs,96000)
        start = int(5*fs)
        y = y[:,start:]
        y = resampler(y)
        fs = 96000
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





#%%
import pandas as pd
import os
path = r"C:\Users\jeffu\Documents\Recordings\06_28_2024_F1"
detected = 0
false_positives = []
for file in os.listdir(path):
    wav_file = os.path.join(path,file)
    times = simulate_real_time_classification(model,wav_file,duration=30, live = False)
    detections = len(times)
    if detections > 0:
        false_positives.append(wav_file)
    print(f'{detections} events detected')
    detected = detected + detections

print(f'Total detections: {detected} in {len(false_positives)} out of {len(os.listdir(path))}')
data = {'File': false_positives}
df = pd.DataFrame(data)
df.to_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\false_positives.csv",index = False)
# %%
from IPython.display import Audio, display
for file in false_positives:
    wav_file = os.path.join(path,file)
    y, fs = ta.load(wav_file)
    audio = y.numpy().copy()
    audio = audio/audio.max()
    display(Audio(audio, rate = fs))
    t = np.arange(y.shape[1])/fs
    plt.plot(t,y[0].numpy())
    times = simulate_real_time_classification(model,wav_file,duration = 30, live=False)
    for i, (start,end) in enumerate(times):
        t_clip = t[start:end]
        clip = y[0,start:end].numpy()
        if i== 0:
            plt.plot(t_clip,clip,'r', label='Event Detected')
            
        else:
           plt.plot(t_clip,clip,'r')
    plt.legend()
    plt.show()

# %%
from IPython.display import Audio, display
wav_file = r"C:\Users\jeffu\Documents\Recordings\test_set\2024-06-24_19_16_14.wav"
y, fs = ta.load(wav_file)
audio = y/y.max()
t = np.arange(y.shape[1])/fs
plt.plot(t,y[0].numpy())
times = simulate_real_time_classification(model,wav_file,duration = 30, live=False)
for i, (start,end) in enumerate(times):
    t_clip = t[start:end]
    clip = y[0,start:end].numpy()
    if i== 0:
        plt.plot(t_clip,clip,'r', label='Event Detected')
        
    else:
        plt.plot(t_clip,clip,'r')
plt.legend()
plt.show()

display(Audio(audio[0].numpy(), rate = fs))

# %%
start = int(21.7*fs)
end = int(21.725*fs)
clip = y[:,start:end]
t_clip = t[start:end]
plt.plot(t_clip,clip[0].numpy())
#%%
clip = clip/clip.max()
trans = ta.transforms.Spectrogram(n_fft = 128, hop_length = 32, power = 1)
spec = trans(clip)
plt.imshow(spec[0])

# %%
clip = clip/clip.max()
filt_clip = bandpass_filter(clip,fs, lowcut = 10000)
filt_clip = torch.from_numpy(filt_clip.astype('f'))
spec = trans(filt_clip)
plt.imshow(spec[0])

# %%
start = int(2.625*fs)
end = int(2.65*fs)
clip = y[:,start:end]
t_clip = t[start:end]
plt.plot(t_clip,clip[0].numpy())
#%%
trans = ta.transforms.Spectrogram(n_fft = 128, hop_length = 32, power = 1)
spec = trans(clip)
plt.imshow(spec[0])

# %%
clip = clip/clip.max()
filt_clip = bandpass_filter(clip,fs, lowcut = 10000)
filt_clip = torch.from_numpy(filt_clip.astype('f'))
spec = trans(filt_clip)
plt.imshow(spec[0])

# %%
plt.plot(t,y[0].numpy())
# %%
y, fs = ta.load(r"C:\Users\jeffu\Documents\Recordings\06_27_2024_R1\2024-06-24_15_26_07.wav")
# %%
thresh = y.mean() + 5*y.std()
t = np.arange(y.shape[1])/fs
plt.plot(t,y[0].numpy())
plt.hlines(thresh.item(),0,30,'r', label = 'Threshold')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Example Audio Clip with Labelling Threshold')
plt.legend()
# %%
