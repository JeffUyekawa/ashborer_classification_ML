#%%
'''
Likely need to install:
pyaudio, numpy, torch, torchaudio, scipy?

Make sure to first figure out the correct recording device index

Update saved model path before run 
'''
import pyaudio
import numpy as np
import torch
import torchaudio as ta
from scipy.signal import butter, filtfilt
import torch.nn as nn
from gpiozero import LED, Button
import time
import os
import datetime

GPIO Pin Numers (BCM not Board Pins nums) 
relayChannel1 = 26 #Board pin 37, BCM pin 26
relayChannel2 = 20 #Board pin 38, BCM pin 20
relayChannel3 = 21 #Board pin 40, BCM pin 21
codecLEDGreen = 23
codecLEDRed   = 24
codecSwitch   = 27



greenLED = LED(codecLEDGreen)
redLED   = LED(codecLEDRed)
relayCh1 = LED(relayChannel1)
relayCh1.on() #Energize to turn off voltage to signal conditioner
codecButton = Button(codecSwitch)

primaryPath = "/home/dasl/repos/boring_recorder/"



# PyAudio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = int(RATE * 0.05)  # 50ms chunk
INPUT_DEVICE_INDEX = 1 #Need to figure out what the index is on rPi

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
        self.linear1=nn.Linear(in_features=1024,out_features=128)
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

# Blink the LED for a short duration (e.g., 100ms on, 100ms off)
def flash_green_led():
    greenLED.blink(on_time=0.1, off_time=0.1, n=1, background=True)

def bandpass_filter(data, fs, lowcut=1000, highcut=12000, order=5, pad=0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, padlen=pad)
    return y

def chunk_to_spec(audio_chunk):
    transform = ta.transforms.MelSpectrogram(sample_rate=RATE, n_fft=64, hop_length=16, f_min=1000, f_max=12000, n_mels=16)
    y = transform(audio_chunk)
    return y

def classify_chunk(audio_chunk):
    spec = chunk_to_spec(audio_chunk)
    spec = spec.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(spec)
        pred = (output >= 0.5) * 1
    return pred.item()

def record_and_classify(duration_seconds=30):
    relayCh1.off() #Denergize to turn on voltage to signal conditioner
    print('Waiting 5 seconds to begin.')
    time.sleep(5) #give voltage in signal condition time to settle
    print('Beginning detection')
    p = pyaudio.PyAudio()

    # Open a stream for input
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,

                    )
    redLED.on()

    try:
        num_chunks = int(duration_seconds * 1000 / 50)  # Total number of 50ms chunks
        for i in range(num_chunks):
            if haltFlag:
                break

            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32)

            audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))  # Normalize
            audio_chunk = bandpass_filter(audio_chunk, RATE)
            audio_chunk = torch.from_numpy(audio_chunk.astype('f'))
            audio_chunk = audio_chunk.unsqueeze(0)

            pred = classify_chunk(audio_chunk)

            if pred == 1:
                flash_green_LED()

    except KeyboardInterrupt:
        print('Recording Interrupted')
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        relayCh1.on() #Denergize to turn on voltage to signal conditioner
        redLED.off()
        print('Recording Complete')


def setdeviceparameters(fileNum=1):
    setupPath = "Pi-Codec/"
    setupFilename1 = "Codec_Zero_AUXIN_record_and_HP_playback.state"
    setupFilename2 = "Codec_Zero_OnboardMIC_record_and_SPK_playback.state"
    setupFilename3 = "Codec_Zero_Playback_only.state"
    setupFilename4 = "Codec_Zero_StereoMIC_record_and_HP_playback.state"
    if fileNum==1:
        setupFilename = setupFilename1
    elif fileNum==2:
        setupFilename = setupFilename2
    elif fileNum==3:
        setupFilename = setupFilename3
    elif fileNum==4:
        setupFilename = setupFilename4
    cwdir=os.getcwd()
    os.system("sudo alsactl restore -f "+primaryPath+setupPath+setupFilename)
    #os.system("amixer")
    #os.system("amixer sset Master unmute")#use Master when running for debugging. Use PCM when executing as shell script
    #os.system("amixer sset Master 50%")

if __name__ == "__main__":
    # Load your trained model
    model = CNNNetwork()
    model_path = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\models\Current_Best_2D.pt"
    model.load_state_dict(torch.load(model_path))  # Update this with the path to your model
    model.eval()
    recordingDuration = 45
    setdeviceparameters(4)
    record_and_classify(duration_seconds=recordingDuration)



# %%
