# %%
import numpy as np
import pandas as pd
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier, ProximityForest
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.feature_based import FreshPRINCEClassifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.classification.dictionary_based import WEASEL_V2
from aeon.classification.interval_based import RSTSF
from aeon.classification.shapelet_based import RDSTClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import argparse

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





def evaluate_models(X,y, model_name):
    
    classifiers = {"DTW":KNeighborsTimeSeriesClassifier(distance = 'dtw'),
                   "Rocket": RocketClassifier(num_kernels=2000, estimator= LogisticRegression(), random_state=13), 
                   "HEC":HIVECOTEV2(), "InceptionTime": InceptionTimeClassifier(), "RDST": RDSTClassifier(max_shapelets=1000), 
                   "Weasel": WEASEL_V2(min_window=8, norm_options=[False], use_first_differences= [False], word_lengths=[7,8,16]), 
                   "RSTSF": RSTSF(), 
                   "FreshPRINCE": FreshPRINCEClassifier(base_estimator=None, default_fc_parameters= 'minimal', n_estimators= 100, pca_solver= 'covariance_eigh', random_state= 13),
                   "PF": ProximityForest()}
    
    models = []
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
        clf = classifiers[model_name]
        acc, prec, rec, f1, train_time, pred_time = train_benchmark(clf, X_train, X_test, y_train, y_test)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        train_times.append(train_time)
        pred_times.append(pred_time)
        models.append(model_name)
        folds.append(i+1)
    folds.append("Average")
    models.append(model_name)
    for metric in [accuracies, precisions, recalls, f1s, train_times, pred_times]:
        metric.append(np.average(metric))
    

       
    results = {"Model": models,"Fold": folds, "Accuracy": accuracies, "Precision": precisions, "Recall": recalls, "F1": f1s, "Train Time": train_times, "Prediction Time": pred_times}
    df = pd.DataFrame(results)
    return df



# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a time series classification model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate.")
    args = parser.parse_args()
    data = np.load('/scratch/jru34/minimal_train_test_arrays.npz')
    X = data['X_train']
    y = data['y_train']
    del data
    model_name = args.model_name
    df = evaluate_models(X,y, model_name)
    save_path = os.path.join(PARENT_PATH,f'{model_name}_results.csv')
    df.to_csv(save_path, index = False)
    
