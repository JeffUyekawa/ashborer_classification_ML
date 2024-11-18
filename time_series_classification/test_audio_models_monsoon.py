# %%
import numpy as np
import pandas as pd
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.hybrid import HIVECOTEV2

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from time import time 
import os


PARENT_PATH = "/home/jru34/Ashborer/time_series_classification"
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
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc: .2f}\n \
        Training time: {train_time:.2f} seconds\n \
        Prediction Time: {pred_time:.2f} seconds")
    '''
    Eventually add some code to save model paramters here
    '''
    
    
    return acc, prec, rec, f1, train_time, pred_time





def evaluate_models():
    data = np.load('/scratch/jru34/filtered_train_test_arrays.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    classifiers = {"DTW":KNeighborsTimeSeriesClassifier(distance = 'dtw'),"Rocket": RocketClassifier(), "HEC":HIVECOTEV2()}
    
    models = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    train_times = []
    pred_times = []

    for model_name in classifiers:
        clf = classifiers[model_name]
        acc, prec, rec, f1, train_time, pred_time = train_benchmark(clf, X_train, X_test, y_train, y_test)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        train_times.append(train_time)
        pred_times.append(pred_time)
        models.append(model_name)
       
    results = {"Model": models, "Accuracy": accuracies, "Train Time": train_times, "Prediction Time": pred_times}
    df = pd.DataFrame(results)
    return df

# %%
if __name__ == "__main__":
    df = evaluate_models()
    save_path = os.path.join(PARENT_PATH,'ts_model_results.csv')
    df.to_csv(save_path, index = False)
    

# %%
