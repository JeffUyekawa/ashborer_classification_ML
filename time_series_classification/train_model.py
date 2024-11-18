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

#sys.path.insert(1, "/home/jru34/Ashborer/Models")
sys.path.insert(1, r"C:\Users\jeffu\Documents\Ash Borer Project\models")
from early_stopping import EarlyStopping
#%%
def train_model(mod,num_epochs,eps,device, train_loader, val_loader, loss_fn, optimizer, checkpoint_path, verbose = False):
    start =time()
    print('Starting Training \n ----------------------------------------')
    
    model = mod
    avg_train_loss = []
    avg_test_loss = []
    avg_train_acc = []
    avg_test_acc = []
    
    early_stopping = EarlyStopping(patience=3, verbose=verbose, path = checkpoint_path)
    sig = nn.Sigmoid()
    for epoch in range(num_epochs):
        
        if verbose:
            if epoch % 10 == 9 or epoch == num_epochs-1 or epoch == 0:
                    print(f'Epoch {epoch+1} \n---------------------')
        model.train()
        train_loss=[]
        train_acc=[]
        for j,(inputs, labels) in enumerate(train_loader):
            
            inputs, labels = inputs.to(device), labels.to(device).long()
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
           
            guess = torch.argmax(y_pred, dim = 1)
            labels = labels.reshape(-1,1).float()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 10 == 9 or epoch == num_epochs-1 or epoch == 0:
                if verbose:
                    if (j+1) % 10 == 0:
                        print(f'Step {j+1}| Loss = {loss.item():.3f}')
            with torch.no_grad():
                train_loss.append(loss.item())
                train_acc.append(accuracy_score(y_true=labels,y_pred=guess))
        avg_train_loss.append(np.average(train_loss))
        avg_train_acc.append(np.average(train_acc))
    #if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_acc = []
            for i, (inputs,labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device).long()
                y_pred2 = model(inputs)
                loss2 = loss_fn(y_pred2, labels)
                guess_2 = torch.argmax(y_pred2, axis = 1)
                guess_2 = guess_2.numpy().astype('long')
                labels = labels.reshape(-1,1).float()
                
                test_loss.append(loss2.item())
                test_acc.append(accuracy_score(y_true=labels,y_pred=guess_2))
                
            avg_test_loss.append(np.average(test_loss))
            avg_test_acc.append(np.average(test_acc))
        
        valid_loss = np.average(test_loss)
        early_stopping(valid_loss,model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
    model.load_state_dict(torch.load(checkpoint_path))
    end = time()
    print(f'Training Complete, {epoch} epochs: Time Elapsed: {(end-start)//60} minutes, {(end-start)%60} seconds')
    return model, avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc

# %%
