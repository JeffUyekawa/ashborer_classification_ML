# %%
from aeon.datasets import load_classification
import numpy as np
import pandas as pd
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.hybrid import HIVECOTEV2
import torch 
import torch.nn as nn
import torchaudio as ta
import matplotlib.pyplot as plt
import math
from General_Timeseries_Model import CNNNetwork
from train_model import train_model
from torch.utils.data import DataLoader
from dataset_class import timeseries_data
from sklearn.metrics import accuracy_score
from time import time 

def load_and_split(dataset):
    X_train,y_train, meta = load_classification(dataset, return_metadata=True, split = "train")
    X_test, y_test = load_classification(dataset,split = "test")
    print('Data Loaded')
    return X_train, X_test, y_train, y_test, meta

def train_benchmark(classifier, X_train, X_test, y_train, y_test):
    clf = classifier
    start = time()
    print('Starting Training')
    clf.fit(X_train,y_train)
    print('Training Complete')
    end= time()
    train_time = end-start
    start = time()
    print('Making Prediction')
    y_pred = clf.predict(X_test)
    end = time()
    pred_time = end-start
    print('Calculating Accuracy')
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc: .2f}\n \
        Training time: {train_time:.2f} seconds\n \
        Prediction Time: {pred_time:.2f} seconds")
    return acc, train_time, pred_time

def get_nearest_power(n):
    result = n // 5
    power_of_2 = 2 ** round(math.log2(result))
    return power_of_2

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

def train_spectrogram_model(X_train, X_test, y_train, y_test, meta, adjust = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    train_dataset = timeseries_data(X_train, y_train, adjust)
    val_dataset = timeseries_data(X_test,y_test, adjust)

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0,
                            )
    val_loader = DataLoader(dataset = val_dataset,
                            batch_size=1,
                            shuffle = False,
                            num_workers =0)
    model = CNNNetwork(num_channels =X_train.shape[1] , num_classes=len(meta['class_values']), first_input = next(iter(train_loader))[0])
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
    if adjust:
        y_test = y_test-1
    acc = accuracy_score(y_test, predictions)
    print(f'True: {y_test}')
    print(f'Pred: {predictions}')
    print(f"Spectrogram Accuracy: {acc:.2f}\n \
        Training time: {train_time:.2f} seconds\n \
            Prediction time: {pred_time:.2f} seconds")
    return acc, train_time, pred_time 

def evaluate_models():
    datasets = ["ACSF1",
    "Adiac",
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MelbournePedestrian",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "PLAID",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga"]
    
    classifiers = {"DTW":KNeighborsTimeSeriesClassifier(distance = 'dtw'),"Rocket": RocketClassifier(), "HEC":HIVECOTEV2()}
    problems = []
    models = []
    accuracies = []
    train_times = []
    pred_times = []
    

    for i, data in enumerate(datasets):
        if i == 2:
            break
        X_train, X_test, y_train, y_test, meta = load_and_split(data)
        if meta["equallength"] == "False":
            continue
        for model_name in classifiers:
            clf = classifiers[model_name]
            acc, train_time, pred_time = train_benchmark(clf, X_train, X_test, y_train, y_test)
            accuracies.append(acc)
            train_times.append(train_time)
            pred_times.append(pred_time)
            models.append(model_name)
            problems.append(data)
        if meta['class_values'][0]=='0':
            adjust = False
        else:
            adjust = True
        acc, train_time, pred_time = train_spectrogram_model(X_train, X_test, y_train, y_test, meta, adjust = adjust)
        accuracies.append(acc)
        train_times.append(train_time)
        pred_times.append(pred_time)
        models.append("Spectrogram")
        problems.append(data)
    results = {"Model": models, "Problem": problems, "Accuracy": accuracies, "Train Time": train_times, "Prediction Time": pred_times}
    df = pd.DataFrame(results)
    return df

# %%
if __name__ == "__main__":
    df = evaluate_models()
    df.to_csv('initial_results.csv', index = False)

#%%
X_train, X_test, y_train, y_test, meta = load_and_split('ACSF1')

train_spectrogram_model(X_train, X_test, y_train, y_test, meta, False)
# %%
