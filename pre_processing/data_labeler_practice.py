#%%
import os
import librosa
import soundfile as sf
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import csv
from pydub import AudioSegment
from pydub.playback import play
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Directory containing the .wav files
audio_dir = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\wav_files_Events"

# Output CSV file to save labels
output_csv = r"C:\Users\jeffu\Downloads\label_practice.csv"

# Function to display spectrogram and play audio
def display_spectrogram_and_play_audio(file_path, canvas):
    y, sr = librosa.load(file_path)
    '''
    # Display the spectrogram
    plt.figure(figsize=(4, 2))
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    
    # Clear the previous plot
    canvas.figure.clf()
    
    # Draw the new plot
    figure = plt.gcf()
    canvas.figure = figure
    canvas.draw()'''
    
    # Play the audio
    audio = AudioSegment.from_wav(file_path)
    play(audio)

# Function to save labels to a CSV file
def save_labels():
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['file_path', 'label'])
        for file_path, var in checkboxes.items():
            label = 'yes' if var.get() else 'no'
            writer.writerow([file_path, label])
    messagebox.showinfo("Saved", "Labels have been saved to " + output_csv)

# Create the main window
root = tk.Tk()
root.title("Audio Labeling Tool")

# Create a frame for the audio files and labels
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Create a dictionary to hold the checkbox variables
checkboxes = {}

# Add audio files and checkboxes to the frame
row = 0
for file in os.listdir(audio_dir):
    if file.endswith('.wav'):
        file_path = os.path.join(audio_dir, file)

        # Create a canvas for spectrograms
        fig, ax = plt.subplots(figsize=(4, 2))
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().grid(row=row, column=3, padx=5, pady=5)

        # Add play button
        play_button = ttk.Button(frame, text="Play", command=lambda f=file_path, c=canvas: display_spectrogram_and_play_audio(f, c))
        play_button.grid(row=row, column=0, padx=5, pady=5)

        # Add label
        ttk.Label(frame, text=file).grid(row=row, column=1, padx=5, pady=5)

        # Add checkbox
        var = tk.IntVar()
        checkbox = ttk.Checkbutton(frame, text="Event", variable=var)
        checkbox.grid(row=row, column=2, padx=5, pady=5)
        checkboxes[file_path] = var

        row += 1

# Add save button
save_button = ttk.Button(root, text="Save Labels", command=save_labels)
save_button.grid(row=1, column=0, pady=10)

# Run the application
root.mainloop()


# %%
