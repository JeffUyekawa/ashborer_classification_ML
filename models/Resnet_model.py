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
import tqdm as tqdm
import sys
from time import time
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(1, "/home/jru34/Ashborer/Models")
from custom_dataset_class import borer_data


if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'

TRAIN_ANNOTATION = "/home/jru34/Ashborer/Datasets/training_data_info.csv"
VAL_ANNOTATION = "/home/jru34/Ashborer/Datasets/val_data_info.csv"

TRAIN_AUDIO = "/home/jru34/Ashborer/Audio_Files/recordings_for_train"
VAL_AUDIO = "/home/jru34/Ashborer/Audio_Files/recordings_for_test"

train_dataset = borer_data(TRAIN_ANNOTATION,TRAIN_AUDIO,mel_spec=True)
val_dataset = borer_data(VAL_ANNOTATION, VAL_AUDIO, mel_spec = True)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
original_weights = model.conv1.weight.data
new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
new_conv1.weight.data = original_weights.mean(dim=1, keepdim=True)
model.conv1 = new_conv1
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

class ResNetWithSigmoid(nn.Module):
    def __init__(self, base_model):
        super(ResNetWithSigmoid, self).__init__()
        # Copy the ResNet model
        self.resnet = base_model
        
        # Define a sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass input through ResNet model
        x = self.resnet(x)
        
        # Apply sigmoid activation
        x = self.sigmoid(x)
        
        return x

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='/home/jru34/Ashborer/Checkpoints/ResnetAshBorercheckpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def train_model(mod,num_epochs):
    start =time()
    print('Starting Training \n ----------------------------------------')
    model = mod
    avg_train_loss = []
    avg_test_loss = []
    avg_train_acc = []
    avg_test_acc = []
    
    early_stopping = EarlyStopping(patience=5, verbose=True)
    for epoch in range(num_epochs):
        if epoch % 10 == 9 or epoch == num_epochs-1 or epoch == 0:
                print(f'Epoch {epoch+1} \n---------------------')
        model.train()
        train_loss=[]
        train_acc=[]
        for j,(inputs, labels) in enumerate(train_loader):
            y_pred = model(inputs).reshape(-1,1).float()
            guess = (y_pred>0.5)*1
            labels = labels.reshape(-1,1).float()
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 9 or epoch == num_epochs-1 or epoch == 0:
                if (j+1) % 10 == 0:
                    print(f'Step {j+1}| Loss = {loss.item():.3f}')
            with torch.no_grad():
                train_loss.append(log_loss(y_true=labels.cpu(), y_pred = y_pred.cpu(),labels=[0,1]))
                train_acc.append(accuracy_score(y_true=labels.cpu(),y_pred=guess.cpu()))
        avg_train_loss.append(np.average(train_loss))
        avg_train_acc.append(np.average(train_acc))
    #if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_acc = []
            for i, (inputs,labels) in enumerate(val_loader):
                y_pred2 = model(inputs).reshape(-1,1).float()
                guess_2 = (y_pred2>=0.5)*1
                labels = labels.reshape(-1,1).float()
                test_loss.append(log_loss(y_pred=y_pred2.cpu(),y_true=labels.cpu(),labels=[0,1]))
                test_acc.append(accuracy_score(y_true=labels.cpu(),y_pred=guess_2.cpu()))
                
            avg_test_loss.append(log_loss(y_pred=y_pred2.cpu(),y_true=labels.cpu(),labels=[0,1]))
            avg_test_acc.append(accuracy_score(y_true=labels.cpu(),y_pred=guess_2.cpu()))
        
        valid_loss = np.average(test_loss)
        early_stopping(valid_loss,model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
    model.load_state_dict(torch.load('/home/jru34/Ashborer/Checkpoints/ResnetAshBorercheckpoint.pt'))
    end = time()
    print(f'Training Complete, {epoch} epochs: Time Elapsed: {(end-start)//60} minutes, {(end-start)%60} seconds')
    return model, avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc
       


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)



overallloss=[]
overallacc=[]
loss_fn = nn.BCELoss()
model = ResNetWithSigmoid(model)
optimizer = torch.optim.Adam(model.parameters(),lr =.0003, weight_decay = 0.0)



train_loader = DataLoader(dataset=train_dataset,
                        batch_size=128,
                        shuffle=True,
                        num_workers=0,
                        )
val_loader = DataLoader(dataset = val_dataset,
                        batch_size=128,
                        shuffle = True,
                        num_workers = 0)
#Can change if using gpu for parallel computing)
#Training Loop (Very, Very slow, so only do 100 epochs until using HPC)



model, train_loss, test_loss, train_r2, test_r2 = train_model(model,1000)



# %%
with torch.no_grad():
    fig,axs = plt.subplots(2,2)

    axs[0][0].plot(train_loss, label= 'Training Loss')
    axs[0][0].legend()
    axs[0][1].plot(test_loss, 'r', label = 'Test Loss'  )
    axs[0][1].legend()


    axs[1][0].plot(train_r2, label = 'Train Acc')
    axs[1][0].legend()
    axs[1][1].plot(test_r2, 'r', label = 'Test Acc')
    axs[1][1].legend()

fig.savefig( "/home/jru34/Ashborer/outputs/resnet_training_curves.png")