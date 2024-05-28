#%%
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import stft
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

def extract_audio_events(file_name, make_events_wav_files_logic=False):
    file_path, name_ext = os.path.split(file_name)
    name, ext = os.path.splitext(name_ext)

    y, fs = sf.read(file_name)
    f_nyquist = fs / 2
    num_channels = y.shape[1]
    num_samps = y.shape[0]
    t = np.arange(num_samps) / fs

    t_wind = 0.05
    n_wind = int(np.ceil(t_wind * fs))
    ovlp_frac = 0.75
    ovlp_num = int(np.ceil(n_wind * ovlp_frac))

    window = np.kaiser(n_wind, 5)  # Use numpy to create Kaiser window
    f, T, s = stft(y.T, fs, window=window, nperseg=n_wind, noverlap=ovlp_num, return_onesided=True)
    s = np.transpose(s, (1, 2, 0))
    num_freqs = len(f)
    num_stft_times = len(T)
    psd_factor = 1 / fs * 1 / np.sum(window**2)
    psd_db = 10 * np.log10(psd_factor * np.abs(s)**2)

    median_noise_pow_db = np.median(10 * np.log10(np.abs(s)**2), axis=1)
    real_stft_std = np.std(np.real(s), axis=1)
    imag_stft_std = np.std(np.imag(s), axis=1)

    std_fact_thresh = 10
    threshold = 10 * np.log10(psd_factor * np.abs(std_fact_thresh * real_stft_std + 1j * std_fact_thresh * imag_stft_std)**2)
    threshold_mats = np.repeat(threshold[:, np.newaxis, :], s.shape[1], axis=1)

    select_mat = psd_db >= threshold_mats

    select_mat[f >= f_nyquist, :] = False

    psd_db_thresholded = np.copy(psd_db)
    psd_db_thresholded[~select_mat] = np.nan

    # Replicate F along the second dimension `num_stft_times` times
    F_replicated_1 = np.repeat(f[:, np.newaxis], num_stft_times, axis=1)

    # Replicate F along the third dimension `num_channels` times
    FMat = np.repeat(F_replicated_1[:, :, np.newaxis], num_channels, axis=2)
    T_transposed = T[:, np.newaxis]
    # Replicate T along the first dimension `num_freqs` times
    T_replicated_1 = np.repeat(T_transposed[np.newaxis, :, :], num_freqs, axis=0)

    # Replicate T along the third dimension `num_channels` times
    TMat = np.repeat(T_replicated_1[:, :, np.newaxis], num_channels, axis=2)

    t_select = TMat[select_mat]
    f_select = FMat[select_mat]
    channel_select = np.where(select_mat)[1]
    t_select_unique, ic = np.unique(t_select, return_inverse=True)

    if len(t_select_unique) > 1:
        t_dist = pdist(t_select_unique.reshape(-1,1))
        t_link = linkage(t_dist, method='average')
        cutoff = 5
        t_clust = fcluster(t_link, cutoff, criterion='distance')
        num_clusters = np.max(t_clust)

        t_clust = t_clust[ic]
    elif len(t_select_unique) == 1:
        num_clusters = 1
        t_clust = np.array([1])
    else:
        num_clusters = 0
        t_clust = np.array([1])

    t_clusters = np.zeros((num_clusters, 2))
    f_peak = np.zeros(num_clusters)
    channel_trigger = [None] * num_clusters

    for i in range(num_clusters):
        t1 = np.min(t_select[t_clust == i + 1])
        t2 = np.max(t_select[t_clust == i + 1])
        t_clust_curr = t_select[t_clust == i + 1]
        f_clust_curr = f_select[t_clust == i + 1]

        f_peak[i] = np.max(f_clust_curr)
        channel_trigger[i] = np.unique(channel_select[t_clust == i + 1])

        t_clusters[i, 0] = t1
        t_clusters[i, 1] = t2

    t0 = np.zeros(num_clusters)
    y_out = [None] * num_clusters
    t_out = [None] * num_clusters

    time_pad = 2

    for i in range(num_clusters):
        t0[i] = t_clusters[i, 0]
        select = (t > t_clusters[i, 0] - time_pad) & (t <= t_clusters[i, 1] + time_pad)
        t_out[i] = t[select]
        y_out[i] = y[select, :]

    ind_sort = np.argsort(t0)
    t0 = t0[ind_sort]
    y_out = [y_out[i] for i in ind_sort]
    channel_trigger = [channel_trigger[i] for i in ind_sort]
    f_peak = f_peak[ind_sort]
    t_out = [t_out[i] for i in ind_sort]

    if make_events_wav_files_logic:
        folder_name = file_path + "_Events"
        os.makedirs(folder_name, exist_ok=True)
        for i in range(num_clusters):
            audio_out_file_name = f"{name}_Event_{i+1}_ChannelTrig_{channel_trigger[i][0]}{ext}"
            sf.write(os.path.join(folder_name, audio_out_file_name), y_out[i] / np.max(np.abs(y_out[i])), fs)

    return y_out, t_out, t0, f_peak, channel_trigger


def play_events(y_out, fs):
    for i, event in enumerate(y_out):
        print(f"Playing event #: {i + 1}")
        event = event / np.max(np.abs(event))
        sd.play(event, fs)
        sd.wait()
        print("done.")


# Uncomment the following lines to extract audio events and play them
file_name = r"C:\Users\jeffu\Documents\Recordings\05_20_2024\2024-05-17_04_23_22.wav"
y_out, t_out, t0, f_peak, channel_trigger = extract_audio_events(file_name)
play_events(y_out, 96000)


