#%%
import pyaudio
import numpy as np
import soundfile as sf
import sounddevice as sd
import torch
import torchaudio as ta
from scipy.signal import butter, filtfilt
import time



FORMAT = None
CHANNELS = 1
RATE = 48000
CHUNK = int(RATE * 0.05) 

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

def classify_chunk(audio_chunk):
    spec = chunk_to_spec(audio_chunk)
    with torch.no_grad():
        output = model1(spec)
        pred =  (output>=0.5)*1
    return pred.item()

def simulate_real_time_classification(wav_file, duration = 30):
    audio_data, sr = ta.load(wav_file)
    y,fs = sf.read(wav_file)
    sd.play(y,fs)
    if audio_data.shape[0] > 1:
        audio_data = audio_data[0,:].reshape(1,-1)
    
    num_chunks = int(duration * 1000/50)
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
            pred = classify_chunk(audio_chunk)
            
            if pred== 1:
                print(f'Event Detected at {int(start_idx/sr)} seconds')
            time.sleep(0.05)
    except KeyboardInterrupt:
        print('Simulation Interrupted')
    finally:
        print('Simulation Complete')



if __name__ == "__main__":
    
    model = torch.load(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\models\saved_2D_model.pt")
    wav_file = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\Datasets\validation_recordings - Copy\Clip 3.wav"
    simulate_real_time_classification(wav_file, duration=30)



# %%
