# Summary
This repository contains code used for the detection and classification of wood-boring insect audio files.
This readme will be updated as the project advances. For now, see below an explanation of the important files in each folder for the current workflow. 

## Datasets
This folder contains two folders of .wav files that were taken from a tree in Riordan Mansion. The files are a randomly chosen subset of 30 second clips that were retrieved on 05/20/2024.
In addition, this folder contains two .csv files that are used to label 50ms clips of the audio for model training and cross-validation.

## Models
Current working models. 1D CNN for classifying waveforms as timeseries. 2D CNN for classifying spectrograms

## Pre-processing
Many files here are works in progress that may not end up being used. Currently, the ones that are most useful are:
* * **subset_recording**: This file is used to walk through the recordings in a folder to classify 50ms chunks of each recording. Currently, this is done by a basic threshold, then a user-verification of chewing events. Each true or false positive from the thresholding is then turned into multiple synthesized recordings by rolling the array.
* * **custom_dataset_class**: This custom class is used to load recordings in Pytorch for training models on either timeseries or spectrograms.
  * **label_events**: A work in progress. Detects chewing events with a power-based threshold on spectrogram. Can potentially be used for more effective auto-labeling of data.

## Visualizations
Early visualizations of chewing events.

## Weather
A script that can line up weather data with each recording using the date-time provided from the RPi recording device. 
