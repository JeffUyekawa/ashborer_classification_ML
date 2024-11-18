# %%
import numpy as np
import pandas as pd
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.hybrid import HIVECOTEV2
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from time import time 
import pickle 
import os


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
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"Accuracy: {acc: .2f}\n \
          Precision: {prec: .2f}\n \
          Recall: {rec: .2f}\n \
        F1: {f1:.2f}\n \
        Training time: {train_time:.2f} seconds\n \
        Prediction Time: {pred_time:.2f} seconds")
    print('Saving Model')
    #parent_path = r"C:\Users\jeffu\Documents\Ash Borer Project\time_series_classification"
    #file_name = 'rocket_classifier.pkl'
    #file_path = os.path.join(parent_path, file_name)
    #with open(file_path, 'wb') as f:
        #pickle.dump(clf, f)
    return acc, f1, prec, rec,  train_time, pred_time





def evaluate_models(X_train, y_train, X_test, y_test):

    
    #data = np.load(r"C:\Users\jeffu\Documents\Ash Borer Project\time_series_classification\filtered_train_test_arrays.npz")
    #X_train = data['X_train'][:100]
    #X_test = data['X_test'][:100]
    #y_train = data['y_train'][:100]
    #y_test = data['y_test'][:100]
    

    #classifiers = {"DTW":KNeighborsTimeSeriesClassifier(distance = 'dtw'),"Rocket": RocketClassifier(num_kernels=200), "HEC":HIVECOTEV2()}
    
    kernels = []
    accuracies = []
    train_times = []
    pred_times = []
    f1s = []
    precs = []
    recs = []
    models = []
    rockets = []

    num_kernels = [100, 500, 5000, 10000]
    transforms = ["rocket", "minirocket", "multirocket"]
    estimators = {"RidgeCV":None, "Logistic Regression":LogisticRegression(max_iter=500)}
    for trans in transforms:
        for estimator in estimators:
            for n in num_kernels:
                est = estimators[estimator]
                clf = RocketClassifier(num_kernels=n, rocket_transform=trans,   random_state = 13, estimator= est)
                print(f'{trans} - {estimator} - {n} kernels \n---------------------------------')
                acc, f1, prec, rec, train_time, pred_time = train_benchmark(clf, X_train, X_test, y_train, y_test)
                accuracies.append(acc)
                train_times.append(train_time)
                pred_times.append(pred_time)
                kernels.append(n)
                f1s.append(f1)
                precs. append(prec)
                recs.append(rec)
                rockets.append(trans)
                models.append(estimator)
                del clf
        
       
    results = {"Model": rockets, "Estimator": models,"Kernels": kernels, "Accuracy": accuracies, "F1": f1s, "Precision": precs, "Recall": recs, "Train Time": train_times, "Prediction Time": pred_times}
    df = pd.DataFrame(results)
    return df

# %%
data = np.load(r"C:\Users\jeffu\Documents\Ash Borer Project\time_series_classification\filtered_train_test_arrays.npz")
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
# %%
df = evaluate_models(X_train, y_train, X_test, y_test)

# %%
df.to_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\time_series_classification\rocket_results.csv", index = False)
# %%
clf = RocketClassifier(num_kernels = 5000, estimator = LogisticRegression(max_iter=2000))
acc, f1, prec, rec, train_time, pred_time = train_benchmark(clf, X_train, X_test, y_train, y_test)
# %%
