#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import  DataLoader
import torchaudio as ta
import os
import sys
from time import time
from sklearn.metrics import log_loss, accuracy_score
import torch.multiprocessing as mp
import warnings
#warnings.filterwarnings("ignore")
sys.path.insert(1, "/home/jru34/Ashborer/Models")
from custom_dataset_class_96k import borer_data
from CNN_Model import CNNNetwork
from early_stopping import EarlyStopping
#%%
def train_model(mod,num_epochs,eps,device, train_loader, val_loader):
    start =time()
    print('Starting Training \n ----------------------------------------')
    
    model = mod
    avg_train_loss = []
    avg_test_loss = []
    avg_train_acc = []
    avg_test_acc = []
    
    early_stopping = EarlyStopping(patience=3, verbose=True)
    sig = nn.Sigmoid()
    for epoch in range(num_epochs):
        scaler = torch.cuda.amp.GradScaler()
        if epoch % 10 == 9 or epoch == num_epochs-1 or epoch == 0:
                print(f'Epoch {epoch+1} \n---------------------')
        model.train()
        train_loss=[]
        train_acc=[]
        for j,(inputs, labels) in enumerate(train_loader):
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            smoothed_labels = labels.clamp(eps/2, 1-eps/2).reshape(-1,1)
            with torch.cuda.amp.autocast():
                y_pred = model(inputs)
                loss = loss_fn(y_pred, smoothed_labels)
            y_pred = sig(y_pred).reshape(-1,1).float()
            guess = (y_pred>0.5)*1
            labels = labels.reshape(-1,1).float()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if epoch % 10 == 9 or epoch == num_epochs-1 or epoch == 0:
                if (j+1) % 10 == 0:
                    print(f'Step {j+1}| Loss = {loss.item():.3f}')
            with torch.no_grad():
                train_loss.append(loss.item())
                train_acc.append(accuracy_score(y_true=labels.cpu(),y_pred=guess.cpu()))
        avg_train_loss.append(np.average(train_loss))
        avg_train_acc.append(np.average(train_acc))
    #if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_acc = []
            for i, (inputs,labels) in enumerate(val_loader):
                y_pred2 = model(inputs)
                y_pred2 = sig(y_pred2).reshape(-1,1).float()
                guess_2 = (y_pred2>=0.5)*1
                labels = labels.reshape(-1,1).float()
                test_loss.append(log_loss(y_pred=y_pred2.cpu(),y_true=labels.cpu(),labels=[0,1]))
                test_acc.append(accuracy_score(y_true=labels.cpu(),y_pred=guess_2.cpu()))
                
            avg_test_loss.append(np.average(test_loss))
            avg_test_acc.append(np.average(test_acc))
        
        valid_loss = np.average(test_loss)
        early_stopping(valid_loss,model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
    model.load_state_dict(torch.load('/home/jru34/Ashborer/Checkpoints/2DAshBorercheckpoint.pt'))
    end = time()
    print(f'Training Complete, {epoch} epochs: Time Elapsed: {(end-start)//60} minutes, {(end-start)%60} seconds')
    return model, avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc


if __name__ == '__main__':
    mp.set_start_method('spawn')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    TRAIN_ANNOTATION = "/home/jru34/Ashborer/Datasets/training_recordings_96k.csv"
    VAL_ANNOTATION = "/home/jru34/Ashborer/Datasets/val_recordings_96k.csv"

    TRAIN_AUDIO = "/home/jru34/Ashborer/Audio_Files/recordings_for_train_labnoise"
    VAL_AUDIO = "/home/jru34/Ashborer/Audio_Files/recordings_for_test_labnoise"

    train_dataset = borer_data(TRAIN_ANNOTATION,TRAIN_AUDIO,spec=True, device=device)
    val_dataset = borer_data(VAL_ANNOTATION, VAL_AUDIO, spec = True, device=device)

    
    model = CNNNetwork()
    if torch.cuda.device_count()> 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr =.00003, weight_decay = 0.0)

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=256,
                            shuffle=True,
                            num_workers=4,
                            )
    val_loader = DataLoader(dataset = val_dataset,
                            batch_size=256,
                            shuffle = True,
                            num_workers =4)

    model, train_loss, test_loss, train_acc, test_acc = train_model(model,1000, 0.1,device, train_loader, val_loader)
    with torch.no_grad():
        fig,axs = plt.subplots(2,2)

        axs[0][0].plot(train_loss, label= 'Training Loss')
        axs[0][0].legend()
        axs[0][1].plot(test_loss, 'r', label = 'Test Loss'  )
        axs[0][1].legend()


        axs[1][0].plot(train_acc, label = 'Train Acc')
        axs[1][0].legend()
        axs[1][1].plot(test_acc, 'r', label = 'Test Acc')
        axs[1][1].legend()

        fig.savefig( "/home/jru34/Ashborer/outputs/training_curves.png")
    