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
       
        
        return logits
    
def bandpass_filter(data,fs, lowcut=1000, highcut=15000, order=5, pad = 0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b,a,data, padlen=pad)
    return y



def chunk_to_spec(audio_chunk):
    transform = ta.transforms.Spectrogram(n_fft=128,hop_length=32,power=1)
    transform2 = ta.transforms.AmplitudeToDB(stype="magnitude")
    y = transform(audio_chunk)
    #y=transform2(y)
    return y

def classify_chunk(model,audio_chunk):
    sig = nn.Sigmoid()
    spec = chunk_to_spec(audio_chunk)
    with torch.no_grad():
        output = model(spec)
        output = sig(output)
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
    
    if sr != 96000:
        resampler = ta.transforms.Resample(sr,96000)
        audio_data = resampler(audio_data)
        sr = 96000
    num_chunks = int(duration * 1000/25)
    times_list = []
    audio_data = audio_data/audio_data.max()
    try:
        for i in range(num_chunks):
            start_idx = i*CHUNK
            end_idx = start_idx + CHUNK
            if end_idx > audio_data.shape[1]:
                break
            audio_chunk = audio_data[:,start_idx:end_idx]
            #audio_chunk = audio_chunk/audio_chunk.max()
           
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
    return times_list

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    #model = torch.load(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\models\saved_2D_model.pth")
    model = CNNNetwork()
    model = nn.DataParallel(model)
    #model.load_state_dict(torch.load(r"C:\Users\jeffu\Downloads\Best_96k_Label_Smoothed.pt"))
    model.load_state_dict(torch.load(r"C:\Users\jeffu\Downloads\check_gpu.pt", map_location = torch.device('cpu')))
    #Clip recorded at 48k
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\07_25_2024_LAB\2024-07-25_09_51_20.wav"
    #wav_file = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\Datasets\validation_recordings - Copy\Clip 2.wav"
    #wav_file = r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\validation_recordings - Copy\Clip 2.wav"

    #Clips recorded at 96k
    wav_file = r"C:\Users\jeffu\Documents\Recordings\06_27_2024_R1\2024-06-24_15_02_24.wav"
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\06_27_2024_R1\2024-06-24_15_13_04.wav"
    #wav_file = r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\recordings_for_test\2024-05-20_11_53_58.wav"
    #wav_file = r"C:\Users\jeffu\Documents\Ash Borer Project\Datasets\recordings_for_test\2024-05-21_16_25_40.wav"

    #96k Forestry
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\06_28_2024_F1\2024-06-27_14_31_12.wav"

    #96k Lab Recordings
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\07_26_2024_LAB\2024-07-26_09_49_11.wav"
    #wav_file = r"C:\Users\jeffu\Documents\Recordings\07_26_2024_LAB\2024-07-26_09_27_55.wav"

    times = simulate_real_time_classification(model,wav_file, duration=30)
    y, fs = ta.load(wav_file)
    
    if fs != 96000:
        resampler = ta.transforms.Resample(fs,96000)
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
for start, end in times:
    start = start+1000
    end = end+1000
    trans = ta.transforms.Spectrogram(n_fft=128, hop_length=32, power = 1)
    audio = torch.from_numpy(y.astype('f'))
    spec = trans(audio[start:end])
    fig, ax = plt.subplots(2,1)
    ax[0].plot(t[start:end],y[start:end])
    ax[1].imshow(spec)

    plt.show()
# %%
y = bandpass_filter(y,fs)
plt.plot(t,y/np.max(np.abs(y)))
# %%
