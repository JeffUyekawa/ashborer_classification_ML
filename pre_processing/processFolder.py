#%%
import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
from extractAudioEvents import extract_audio_events

def process_folder(folder_path):
    folder_contents = os.listdir(folder_path)
    num_files = len(folder_contents)
    f = None

    try:
        f = tqdm(total=num_files, desc="Processing Data...")
        #print('Check 1')

        for file_name in folder_contents:
            _, ext = os.path.splitext(file_name)
            full_file_path = os.path.join(folder_path, file_name)
            print(f'file path: {full_file_path}')
            if ext.lower() == '.wav': #and int(file_name[-8:-4]) > 1168:
                f.update(1)
                print("Processing file {} of {}".format(f.n, len(folder_contents)))
                try:
                    y_out, t_out, t0, f_peak, channel_trigger = extract_audio_events(full_file_path, True)
                except Exception as e:
                    print("Error processing file:", file_name)
                    print(e)
                

    finally:
        if f:
            f.close()

# Example folder path
folder_path = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\wav_files"

process_folder(folder_path)

# %%
