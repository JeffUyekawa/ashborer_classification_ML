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
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


sys.path.insert(1, r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\pre_processing")
from custom_dataset_class import borer_data

TRAIN_ANNOTATION = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\Datasets\training_data_info.csv"
VAL_ANNOTATION = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\Datasets\val_data_info.csv"

TRAIN_AUDIO = r"C:\Users\jeffu\Documents\Recordings\recordings_for_train"
VAL_AUDIO = r"C:\Users\jeffu\Documents\Recordings\recordings_for_test"

train_dataset = borer_data(TRAIN_ANNOTATION,TRAIN_AUDIO,mel_spec=True)
val_dataset = borer_data(VAL_ANNOTATION, VAL_AUDIO, mel_spec = True)

class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=3072,out_features=128)
        self.linear2=nn.Linear(in_features=128,out_features=1)
        self.output=nn.Sigmoid()
    
    def forward(self,input_data):
        x=self.conv1(input_data)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.flatten(x)
        x=self.linear1(x)
        logits=self.linear2(x)
        output=self.output(logits)
        
        return output
#%%
class LitCNN(L.LightningModule):
    def __init__(self, CNN):
        super().__init__()
        self.CNN = CNN
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z = self.CNN(x)
        y = y.view(-1,1).float()
        loss = F.binary_cross_entropy(z,y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        z = self.CNN(x)
        y = y.view(-1,1).float()
        val_loss = F.binary_cross_entropy(z, y)
        self.log("val_loss", val_loss,on_step = True, on_epoch=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.CNN(x)
        y = y.view(-1,1).float()
        test_loss = F.binary_cross_entropy(z, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
train_loader = DataLoader(dataset=train_dataset,
                        batch_size=128,
                        shuffle=True,
                        num_workers=8,
                        persistent_workers=True
                        )
val_loader = DataLoader(dataset = val_dataset,
                        batch_size=128,
                        shuffle = False,
                        num_workers = 8,
                        persistent_workers=True)

#%%

# model
model = LitCNN(CNNNetwork())
root_dir = r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Ash Borer Project\models"

# train model
trainer = L.Trainer(default_root_dir=root_dir, callbacks=[EarlyStopping(monitor = "val_loss", mode="min")], profiler='simple')
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)



#%%
# initialize the Trainer
trainer = L.Trainer()

# test the model
trainer.test(model, dataloaders=DataLoader(test_set))