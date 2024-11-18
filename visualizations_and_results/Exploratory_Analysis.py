#%%
import numpy as np
import pandas as pd
import torch
import torchaudio as ta
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from scipy.signal import butter, filtfilt
import torch.nn.functional as F

def sobel_filter(image):
    sobel_kernel = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]])

    # Reshape the kernel to match the expected format for convolution (out_channels, in_channels, H, W)
    sobel_kernel = sobel_kernel.view(1, 1, 3, 3)
    # Add batch and channel dimensions to the image (needed for 2D convolution)
    image = image.unsqueeze(0).unsqueeze(0)
    # Pad the image to apply the filter (since Sobel is 3x3, we pad to maintain the size)
    image_padded = F.pad(image, (1, 1, 1, 1), mode='constant', value=0)
    # Apply the Sobel filter using 2D convolution
    filtered_image = F.conv2d(image_padded, sobel_kernel)
    # Remove the extra batch and channel dimensions for display
    filtered_image = filtered_image.squeeze()
    return filtered_image

def get_noisy(y, snr):
    noise = np.random.normal(0, 1, y.shape)
    noise = torch.from_numpy(noise.astype('f'))
    desired_snr_db = snr
    audio_power = torch.mean(y**2)
    noise_power = torch.mean(noise**2)
    scaling_factor = torch.sqrt(audio_power / (10**(desired_snr_db / 10) * noise_power))
    noise = noise * scaling_factor
    noisy_y = y + noise
    return noisy_y

#path = r"C:\Users\jeffu\Documents\Recordings\all_recordings\08_28_2024_LAB_LargeLog\2024-08-28_09_34_23.wav"
#path = r"C:\Users\jeffu\Documents\Recordings\all_recordings\08_28_2024_LAB_LargeLog\2024-08-28_10_46_17.wav"
#path = r"C:\Users\jeffu\Documents\Recordings\all_recordings\08_28_2024_LAB_LargeLog\TASCAM_1582.wav"
#path = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train_96k\2024-06-24_15_24_46.wav"
#path = r"C:\Users\jeffu\Documents\Recordings\all_recordings\06_27_2024_R1\2024-06-24_18_49_27.wav"
#path = r"C:\Users\jeffu\Downloads\TASCAM_1585.wav"
path = r"C:\Users\jeffu\Documents\Recordings\all_recordings\06_27_2024_R1\2024-06-24_15_20_39.wav"
y, fs = ta.load(path)
#y = get_noisy(y,5)

t = np.arange(len(y[0]))/fs
plt.plot(t,y[0].numpy())
plt.show()
#%%
start = int(21*fs)
end = int(23*fs)
clip = y[:,start:end]
plt.plot(t[start:end],clip[0].numpy())
#%%
trans = ta.transforms.MelSpectrogram(sample_rate = fs, n_fft = 2048, hop_length =2048//2, n_mels = 32)
spec = trans(clip)
plt.imshow(spec[0], aspect = 'auto')
#%%
start = int(22.1*fs)
end = int(22.125*fs)
clip = y[:,start:end]
clip = clip/clip.max()

plt.plot(t[start:end],clip[0,:].numpy())
plt.show()
#%%
n_fft =128
hop_length = n_fft//4
trans = ta.transforms.Spectrogram(n_fft = n_fft, hop_length=hop_length, power = 1)
trans2 = ta.transforms.AmplitudeToDB()

spec = trans(clip)
#spec = trans2(spec)
plt.imshow(spec[0], aspect = 'auto')
#%%
sobel_spec = sobel_filter(spec[0])
plt.imshow(sobel_spec, aspect = 'auto')
# %%
y,fs = ta.load(r"C:\Users\jeffu\Documents\Recordings\all_recordings\08_28_2024_LAB_LargeLog\2024-08-28_09_34_23.wav")
plt.plot(y[0].numpy())
# %%
import os 
ann_path = r"C:\Users\jeffu\Documents\Recordings\one_second_training_labels.csv"
audio_path = r"C:\Users\jeffu\Documents\Recordings\one_second_training"

df = pd.read_csv(ann_path)
df = df.drop_duplicates(subset = ['File','Start'], keep = 'first')

pos_df = df[df.Label.eq(1)].sample(5)

for i, row in pos_df.iterrows():
    file = row['File']
    full_path = os.path.join(audio_path, file)
    y, fs = ta.load(full_path)
    if fs != 96000:
        resampler = ta.transforms.Resample(fs,96000)
        y = resampler(y)
        fs = 96000
        y = y[0,int(4*fs):].reshape(1,-1)
    if y.shape[0] > 1:
        y = y[1,:].reshape(1,-1)
    start = row['Start']
    end = row['End']
    clip = y[:,start:end]
    t = np.arange(clip.shape[1])/fs
    fig, ax = plt.subplots(3,1)
    display(Audio(clip[0].numpy(), rate = fs))
    ax[0].plot(t,clip[0].numpy())
    
    
    trans = ta.transforms.Spectrogram(n_fft = 1024, hop_length = 1024//4, power =1)
    spec = trans(clip)
    ax[1].imshow(spec[0],aspect = 'auto')
    

    spec = sobel_filter(spec[0])
    ax[2].imshow(spec, aspect = 'auto')
    
# %%
