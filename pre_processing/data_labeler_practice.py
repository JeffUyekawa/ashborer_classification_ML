#%%
import os
import librosa
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import csv
from pygame import mixer
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Initialize pygame mixer
mixer.init()

# Directory containing the .wav files
audio_dir = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\wav_files_Events"

# Output CSV file to save labels
output_csv = r"C:\Users\jeffu\Downloads\label_practice.csv"

# Function to display spectrogram and play audio
def display_spectrogram_and_play_audio(file_path, canvas):
    y, sr = librosa.load(file_path)
    
    # Display the spectrogram
    plt.figure(figsize=(4, 2))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
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
    canvas.draw()
    
    # Play the audio
    mixer.music.load(file_path)
    mixer.music.play()

# Function to stop audio and update checkbox state
def stop_audio_and_update_checkbox_state(var, file_path):
    mixer.music.stop()
    var.set(1)  # Set the checkbox to checked

# Function to save labels to a CSV file
def save_labels():
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['file_name', 'label'])
        for file_path, var in checkboxes.items():
            label = 1 if var.get() else 0
            file_name = os.path.basename(file_path)
            writer.writerow([file_name, label])
    messagebox.showinfo("Saved", "Labels have been saved to " + output_csv)

# Create the main window
root = tk.Tk()
root.title("Audio Labeling Tool")

# Create a canvas for scrollable frame
canvas = tk.Canvas(root)
scroll_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scroll_x = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
frame = ttk.Frame(canvas)

# Create a window inside the canvas to hold the frame
window = canvas.create_window((0, 0), window=frame, anchor="nw")

# Configure the scrollbars
canvas.configure(yscrollcommand=scroll_y.set)
canvas.configure(xscrollcommand=scroll_x.set)

# Pack the scrollbars and canvas
scroll_y.pack(side="right", fill="y")
scroll_x.pack(side="bottom", fill="x")
canvas.pack(side="left", fill="both", expand=True)

# Create a dictionary to hold the checkbox variables
checkboxes = {}

# Add audio files and checkboxes to the frame
row = 0
for file in os.listdir(audio_dir):
    if file.endswith('.wav'):
        file_path = os.path.join(audio_dir, file)

        # Create a canvas for spectrograms
        fig, ax = plt.subplots(figsize=(4, 2))
        spectrogram_canvas = FigureCanvasTkAgg(fig, master=frame)
        spectrogram_canvas.get_tk_widget().grid(row=row, column=3, padx=5, pady=5)

        # Add play button
        play_button = ttk.Button(frame, text="Play", command=lambda f=file_path, c=spectrogram_canvas: display_spectrogram_and_play_audio(f, c))
        play_button.grid(row=row, column=0, padx=5, pady=5)

        # Add label
        ttk.Label(frame, text=file).grid(row=row, column=1, padx=5, pady=5)

        # Add checkbox
        var = tk.IntVar()
        checkbox = ttk.Checkbutton(frame, text="Event", variable=var, command=lambda v=var, f=file_path: stop_audio_and_update_checkbox_state(v, f))
        checkbox.grid(row=row, column=2, padx=5, pady=5)
        checkboxes[file_path] = var

        row += 1

# Add save button
save_button = ttk.Button(root, text="Save Labels", command=save_labels)
save_button.pack(pady=10)

# Update scroll region after adding all widgets
frame.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))

# Run the application
root.mainloop()

# Quit mixer on application close
mixer.quit()




# %%
