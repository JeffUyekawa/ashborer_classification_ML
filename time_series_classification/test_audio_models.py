# %%
import numpy as np
import pandas as pd
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier, ProximityForest
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.feature_based import FreshPRINCEClassifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.classification.dictionary_based import WEASEL
from aeon.classification.interval_based import RSTSF
from aeon.classification.shapelet_based import RDSTClassifier
import matplotlib.pyplot as plt
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
    #print('Saving Model')
    #parent_path = r"C:\Users\jeffu\Documents\Ash Borer Project\time_series_classification"
    #file_name = 'rocket_classifier.pkl'
    #file_path = os.path.join(parent_path, file_name)
    #with open(file_path, 'wb') as f:
        #pickle.dump(clf, f)
    return acc, prec, rec, f1,  train_time, pred_time





def evaluate_models(X_train, y_train, X_test, y_test):
 
    classifiers = {"DTW":KNeighborsTimeSeriesClassifier(distance = 'dtw'),"Rocket": RocketClassifier(num_kernels=500, random_state=13), "HEC":HIVECOTEV2(), "InceptionTime": InceptionTimeClassifier(), "RDST": RDSTClassifier(), "Weasel": WEASEL(), "RSTSF": RSTSF(), "FreshPRINCE": FreshPRINCEClassifier(), "PF": ProximityForest()}
    
    models = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    train_times = []
    pred_times = []

    for model_name in classifiers:
        print(f'{model_name}\n--------------------------------------')
        clf = classifiers[model_name]
        acc, prec, rec, f1, train_time, pred_time = train_benchmark(clf, X_train, X_test, y_train, y_test)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        train_times.append(train_time)
        pred_times.append(pred_time)
        models.append(model_name)
       
    results = {"Model": models, "Accuracy": accuracies, "Precision": precisions, "Recall": recalls, "F1 Score":f1s, "Train Time": train_times, "Prediction Time": pred_times}
    df = pd.DataFrame(results)
    return df

def sample_x_y(X,y, n, m, seed):
    indices_0 = np.where(y == 0)[0]
    indices_1 = np.where(y == 1)[0]

    # Randomly sample n negative an m positive instances
    np.random.seed(seed)  # For reproducibility
    sample_indices_0 = np.random.choice(indices_0, size=n, replace=False)
    sample_indices_1 = np.random.choice(indices_1, size=m, replace=False)

    # Get the corresponding samples from X and y
    X_sampled = np.concatenate([X[sample_indices_0], X[sample_indices_1]])
    y_sampled = np.concatenate([y[sample_indices_0], y[sample_indices_1]])   
    return X_sampled, y_sampled
# %%
data = np.load(r"C:\Users\jeffu\Documents\Ash Borer Project\time_series_classification\minimal_train_test_arrays.npz")
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

'''X_train, y_train = sample_x_y(X_train, y_train, 10, 10, 13)
X_test, y_test = sample_x_y(X_test, y_test, 5, 5, 13)'''
# %%
df = evaluate_models(X_train, y_train, X_test, y_test)

# %%
df.to_csv(r"C:\Users\jeffu\Documents\Ash Borer Project\time_series_classification\early_all_timeseries_results.csv", index = False)
# %%
