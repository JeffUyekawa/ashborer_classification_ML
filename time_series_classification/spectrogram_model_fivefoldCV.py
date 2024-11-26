# %%
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torchaudio as ta
import matplotlib.pyplot as plt
import math
from General_Timeseries_Model import CNNNetwork
from train_model import train_model
from torch.utils.data import DataLoader
from dataset_class import timeseries_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from time import time 
from sklearn.model_selection import KFold
import os

PARENT_PATH = "/home/jru34/Ashborer/time_series_classification"

def plot_training_curves(train_loss, test_loss, train_acc, test_acc):
    fig,axs = plt.subplots(2,2)

    axs[0][0].plot(train_loss, label= 'Training Loss')
    axs[0][0].legend()
    axs[0][1].plot(test_loss, 'r', label = 'Test Loss'  )
    axs[0][1].legend()


    axs[1][0].plot(train_acc, label = 'Train Acc')
    axs[1][0].legend()
    axs[1][1].plot(test_acc, 'r', label = 'Test Acc')
    axs[1][1].legend()
    plt.show()

def train_spectrogram_model(X_train, X_test, y_train, y_test):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    train_dataset = timeseries_data(X_train, y_train, adjust = False)
    val_dataset = timeseries_data(X_test,y_test, adjust = False)

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=16,
                            shuffle=True,
                            num_workers=0,
                            )
    val_loader = DataLoader(dataset = val_dataset,
                            batch_size=1,
                            shuffle = False,
                            num_workers =0)
    model = CNNNetwork(num_channels =X_train.shape[1] , num_classes=2, first_input = next(iter(train_loader))[0])
    if torch.cuda.device_count()> 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr =.00001, weight_decay = 0.0)


    start = time()
    model, train_loss, test_loss, train_acc, test_acc = train_model(model,1000, 0.0,device, train_loader, val_loader, loss_fn, optimizer, "ts_checkpoint.pt")
    end = time()
    train_time = end-start
    start = time()
    with torch.no_grad():
        predictions = []
        for inputs, _ in val_loader:
            pred = model(inputs)
            guess = torch.argmax(pred, axis=1)
            predictions.append(guess.item())
    plot_training_curves(train_loss, test_loss, train_acc, test_acc)
    end = time()
    pred_time = end-start   
    y_test = y_test.astype('long')
    
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"Spectrogram Accuracy: {acc:.2f}\n \
        Training time: {train_time:.2f} seconds\n \
            Prediction time: {pred_time:.2f} seconds")
    return acc, prec, rec, f1, train_time, pred_time 

def evaluate_models(X,y):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    train_times = []
    pred_times = []
    folds = []
    

    CV = KFold(n_splits = 5, shuffle = True, random_state = 13)
    for i , (train_idx, test_idx) in enumerate(CV.split(X)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        acc, prec, rec, f1, train_time, pred_time = train_spectrogram_model(X_train, X_test, y_train, y_test)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        train_times.append(train_time)
        pred_times.append(pred_time)
        folds.append(i+1)
    folds.append("Average")
    for metric in [accuracies, precisions, recalls, f1s, train_times, pred_times]:
        metric.append(np.average(metric))
    

       
    results = {"Fold": folds, "Accuracy": accuracies, "Precision": precisions, "Recall": recalls, "F1": f1s, "Train Time": train_times, "Prediction Time": pred_times}
    df = pd.DataFrame(results)
    return df

# %%
if __name__ == "__main__":
    data = np.load('/scratch/jru34/minimal_train_test_arrays.npz')
    X = data['X_test']
    y = data['y_test']
    del data
    df = evaluate_models(X,y)
    save_path = os.path.join(PARENT_PATH,'spectrogram_results.csv')
    df.to_csv(save_path, index = False)
