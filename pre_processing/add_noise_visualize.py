#%%
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

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
      

# Function to get the spectrogram using torchaudio
def get_spec(clip, n_fft):
    hop_length = int(n_fft / 4)
    spectrogram_transform = T.Spectrogram( n_fft=n_fft, hop_length=hop_length, power = 1)
    amplitude_to_db = T.AmplitudeToDB(stype='magnitude')
    spectrogram = spectrogram_transform(clip)
    DB = amplitude_to_db(spectrogram)
    return spectrogram  

# Function to add noise to the signal with a specific SNR
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
def bandpass_filter(data,fs, lowcut=1000, highcut=20000, order=5, pad = 0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b,a,data, padlen=pad)
    return y

# Function to prepare the clips and compute the spectrograms
def prepare_clips(t, y, snr, fs):
    peak_idx = torch.argmax(y).item()
    buffer = int(0.025 * fs / 2)
    peak_start = max(0, peak_idx - buffer)
    peak_end = min(len(y), peak_idx + buffer)
    noisy_y = get_noisy(y, snr)
    #test no-noise clip
    #peak_start = int(12.5*fs)
    #peak_end = int(12.525*fs)
    #test smaller peak
    #peak_start = int(9.4*fs)
    #peak_end = int(9.425*fs)
    t_clip = t[peak_start:peak_end]
    y_clip = y[peak_start:peak_end]
    noisy_clip = noisy_y[peak_start:peak_end]
    return noisy_y, t_clip, y_clip, noisy_clip, fs

# Function to add the spectrogram to the plot
def spec_to_axis(spec, i):
    ax[i][2].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]])
    ax[i][2].set_aspect('auto')

if __name__ == "__main__":
    # Load the audio file using torchaudio
    wav_file = r"C:\Users\jeffu\Documents\Recordings\07_25_2024_LAB\2024-07-25_10_05_28.wav"
    model = CNNNetwork()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(r"C:\Users\jeffu\Downloads\2DAshBorercheckpoint (13).pt", map_location = torch.device('cpu')))
    y, fs = torchaudio.load(wav_file)
    y = y / torch.max(torch.abs(y))  # Normalize
    t = torch.arange(len(y[0])) / fs
    thresh = np.mean(y[0,:].numpy()) + 5*np.std(y[0,:].numpy())

    noise_levels = np.arange(0, -22, -2)
    fig, ax = plt.subplots(len(noise_levels), 3, figsize=(14, 24), sharex='col')

    for i, item in enumerate(noise_levels):
        noisy_y, t_clip, y_clip, noisy_clip, fs = prepare_clips(t, y[0], item, fs)
        noisy_spec = get_spec(noisy_clip, 128)
        pred = model(noisy_spec.unsqueeze(0).unsqueeze(0))
        sig = nn.Sigmoid()
        pred = sig(pred)
        print(pred.item())
        pred = (pred.item() >= 0.5)*1
        if noisy_clip.max()<= thresh:
            pred = 0
        
        noisy_spec = noisy_spec.squeeze().numpy()
        if i == 0:
            spec = get_spec(y_clip, 128)
            pred = model(spec.unsqueeze(0).unsqueeze(0))
            sig = nn.Sigmoid()
            pred = sig(pred)
            print(pred.item())
            pred = (pred.item()>=0.5)*1
            if y_clip.max().item() <= thresh:
                pred = 0
            ax[0][0].plot(t.numpy(), y[0].numpy())
            ax[0][0].plot(t_clip.numpy(), y_clip.numpy(), 'r')
            ax[0][1].plot(t_clip.numpy(), y_clip.numpy())
            if pred == 1:
                ax[0][1].plot(t_clip.numpy(), y_clip.numpy(),'r')
            spec_to_axis(spec, 0)
            ax[0][0].text(-0.1, 0.5, 'Original', va='center', ha='right', transform=ax[i][0].transAxes, fontsize=12)
        else:
            ax[i][0].plot(t.numpy(), noisy_y.numpy())
            ax[i][0].plot(t_clip.numpy(), noisy_clip.numpy(), 'r')
            ax[i][1].plot(t_clip.numpy(), noisy_clip.numpy())
            if pred == 1:
                ax[i][1].plot(t_clip.numpy(),noisy_clip.numpy(),'r')
            spec_to_axis(noisy_spec, i)
            ax[i][0].text(-0.1, 0.5, f'SNR: {item} dB', va='center', ha='right', transform=ax[i][0].transAxes, fontsize=12)

    fig.tight_layout()
    plt.show()



# %%
idx = torch.argmax(y).item()
print(idx)
buffer = int(0.025 * fs / 2)
start = max(0, idx - buffer)
end = min(y.shape[1], idx + buffer)
edge = int(1200-0.001*fs)
clip =torch.from_numpy(np.roll(y[0,start:end].numpy(),-600).astype('f'))
plt.plot(t.numpy(),y[0].numpy())
plt.show()
plt.plot(t[start:end].numpy(),clip.numpy())
# %%
import time
from IPython.display import clear_output

num_rolls = 241-13
idx = torch.argmax(clip)
right_bumper = (2300-idx)//10
for roll in range(num_rolls):
    if roll >= right_bumper:
        roll = roll+13
    rolled = np.roll(clip.numpy(),roll*10)
    plt.plot(t[start:end].numpy(),rolled)
    plt.show()
    time.sleep(0.1)
    clear_output()
    

# %%
import torchaudio as ta
from IPython.display import Audio, display
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
y, fs = ta.load(r"C:\Users\jeffu\Documents\Recordings\07_26_2024_LAB\2024-07-26_09_35_09.wav")
y = y/y.max()
print(y.max())
noisy_y = get_noisy(y,-0.5)
print(noisy_y.max())
t = np.arange(y.shape[1])/fs

fig, ax = plt.subplots(2,1)
ax[0].plot(t,y[0].numpy())
ax[1].plot(t,noisy_y[0].numpy())
plt.show()
display(Audio(y[0].numpy(), rate=fs))
display(Audio(noisy_y.numpy(),rate=fs))
# %%
mid = y.argmin().item()
buff = int(0.025/2*fs)
start = max(0,mid-buff)
end = min(mid + buff, int(fs*30))
clip = y[:,start:end]
trans = ta.transforms.Spectrogram(n_fft = 128, hop_length=32, power = 1)
spec = trans(clip)
plt.plot(t[start:end],clip[0].numpy())
plt.show()
plt.imshow(spec[0])

# %%
spec.max()

# %%
