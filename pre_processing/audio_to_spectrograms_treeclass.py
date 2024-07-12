#%%
import numpy as np
import torchaudio as ta
import matplotlib.pyplot as plt
from PIL import Image


SPECTROGRAM_DPI = 90 # image quality of spectrograms
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_HOP_LENGHT = 1024

class audio():
    def __init__(self, waveform, sample_rate, hop_lenght = DEFAULT_HOP_LENGHT, samples_rate = DEFAULT_SAMPLE_RATE):
        self.hop_lenght = hop_lenght
        self.samples_rate = samples_rate
        self.waveform, self.sample_rate = waveform, sample_rate

    def plot_spectrogram(self) -> None:
        waveform = self.waveform.numpy()
        _, axes = plt.subplots(1, 1)
        axes.specgram(waveform, Fs=self.sample_rate)
        plt.axis('off')
        #plt.show(block=False)
    
    def write_disk_spectrogram(self, path, dpi=SPECTROGRAM_DPI) -> None:
        self.plot_spectrogram()
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close()


#%%
import os
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\pre_processing")

from LabelEvents import label_audio_events

path1=r"C:\Users\jeffu\Documents\Recordings\06_27_2024_R1"
path2 = r"C:\Users\jeffu\Documents\Recordings\06_28_2024_F1"


#%%

output = r"C:\Users\jeffu\Documents\Recordings\Tree_Classifier_Images"

for j, file in enumerate(os.listdir(path1)): 
    full_path = os.path.join(path1,file)
    signal,sr = ta.load(full_path)
    signal = signal[0,:]
    
    for i in np.arange(int(signal.shape[0]/sr)):
        
        start = i*sr
        end = (i+1)*sr
        clip = signal[start:end]
        sound = audio(clip,sr)
        out_file = file[:-4] +'_'+ str(i)+'.png'
        out_path = output + "\\Infested"
        output_path = os.path.join(out_path,out_file)
        sound.write_disk_spectrogram(output_path, dpi=SPECTROGRAM_DPI)

# %%
for j, file in enumerate(os.listdir(path2)): 
    full_path = os.path.join(path2,file)
    signal,sr = ta.load(full_path)
    signal = signal[0,:]
    
    for i in np.arange(int(signal.shape[0]/sr)):
        
        start = i*sr
        end = (i+1)*sr
        clip = signal[start:end]
        sound = audio(clip,sr)
        out_file = file[:-4] +'_'+ str(i)+'.png'
        out_path = output + "\\Non_infested"
        output_path = os.path.join(out_path,out_file)
        sound.write_disk_spectrogram(output_path, dpi=SPECTROGRAM_DPI)

# %%
