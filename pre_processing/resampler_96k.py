#%%
import torchaudio as ta
import numpy as np
import soundfile as sf

path ='C:\\Users\\jeffu\\Documents\\Recordings\\06_28_2024_F1\\2024-06-28_13_39_09.wav'
y, fs = ta.load(path)
resampler = ta.transforms.Resample(fs,48000)
y = resampler(y)
y = y/y.max()
y = y[0,:].numpy()

out_path = r"C:\Users\jeffu\Documents\Recordings\06_28_2024_F1\resampled_clip_3.wav"
sf.write(out_path,y,48000)
# %%
