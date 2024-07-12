#%%
from sklearn.cluster import KMeans
import librosa
import numpy as np
import os

def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.concatenate((mfccs.mean(axis=1), chroma.mean(axis=1), zcr.mean(axis=1)))

folder_path = r"C:\Users\jeffu\Documents\Recordings\05_20_2024"
audio_data, file_names = [], []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.wav') :
        file_path = os.path.join(folder_path, file_name)
        y, sr = librosa.load(file_path, sr=None)
        features = extract_features(y, sr)
        audio_data.append(features)
        file_names.append(file_name)

audio_data = np.array(audio_data)

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(audio_data)
labels = kmeans.labels_
#%%
audio_clusters = {}
# You can then listen to a few samples from each cluster to understand and label them
for cluster in range(2):
    print(f"Cluster {cluster}")
    clusterlist = []
    for i, file_name in enumerate(file_names):
        if labels[i] == cluster:
            print(file_name)
            clusterlist.append(file_name)
    audio_clusters[str(cluster)] = clusterlist 

            # Listen to the audio file to determine the nature of the cluster
            # e.g., librosa.output.write_wav('output.wav', y, sr)

# %%
clusterlist
# %%
