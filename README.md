# Summary
This repository contains code used for the detection and classification of wood-boring insect audio files.
This readme will be updated as the project advances. For now, see below an explanation of the important files in each folder for the current workflow. 

## Datasets
All necessary audio files and data sets will be uploaded separately to the shared google drive. 

## Detection scripts
Scripts used to implement detection on the device. One important script for workflow is:
1.**simulation_real_time_detection_96k**: This script can be used to test current models in psuedo-realtime on chosen audio clips. 

## New Ideas and Exploration
Scripts used for exploring new ideas and techniques. Current ideas that have not yet been fleshed out are using isolation forest for anomaly detection and using gramian angular difference fields for image classification. 

## Old Files
Early model designs that have been shelved for now. Models include 1D CNN, Deep Cluster, Multiclass model, a 2d model trained on 2 second windows, and a model trained on 48k clips using Mel Spectrograms. 

## Models
This folder contains the current best performing model used in the demonstration. The model is a simple 2D CNN trained on 96k audio, 25ms in length. The necessary files are:
1. **2D_CNN**: The training script
2. **CNN_Model**: The model architecture
3. **early_stopping**: A class used to implement early stopping to avoid overfitting

## Pre-processing
Files used to preprocess the data. 
1. **subset_recording**: This file is used to take a folder of audio data, segmet clips into 25ms windows, set a threshold, then manually label chewing events. Labeled data is used to create synthesized data by either adding synthetic noise, or implementing a temporal shift to positive instances. One can optionally upsample the minority class to create an evenly split training or test set. 
2. **custom_dataset_class_96k**: Custom class used to load data as spectrograms in the training script. 
3. **data labeler**: Data labeling with more user friendly features (stop and save, go back to previous)
4. **resampler_96k**: Can be used to resample audio when necessary
5. **train_test_split_dataset**: If one chooses to label one large batch of audio clips, this script can be used to split the labeled dataframe into train and test dataframes
6. **add_noise_visualize**: A script written to test adding synthetic noise to the data.

## Time Series Classification
These scripts are currently in development. We are looking into whether our method can be used more generally for time series, or if current state of the art time series classification models show promise on our dataset. Details to come. 

## Visualizations and Results
Early visualizations of chewing events and scripts to analyze model results. The important scripts for the workflow are:
1. **Analyze Results**: With a pre-trained model, this script can be used to analyze the model performance with a confusion matrix and plotting audio clips with false/true positive and false/true negative predictions. 

## Weather
A script that can line up weather data with each recording using the date-time provided from the RPi recording device. So far, this functionality has not been used in model development. 
