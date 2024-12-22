# %%
import numpy as np
import pandas as pd
from aeon.classification.distance_based import ProximityForest
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.feature_based import FreshPRINCEClassifier
from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.classification.dictionary_based import WEASEL_V2
from aeon.classification.interval_based import RSTSF
from aeon.classification.shapelet_based import RDSTClassifier
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import argparse

PARENT_PATH = "/home/jru34/Ashborer/time_series_classification"


def optimize_params(X, y, model_name):
    
    classifiers = {"Rocket": {"model": RocketClassifier(), "params":{
                            'num_kernels': [10, 100, 500,1000,2000,5000],
                            'estimator': [None, LogisticRegression(max_iter = 1000), RandomForestClassifier()]
                            }},
                   "InceptionTime": {"model":InceptionTimeClassifier(),"params":{
                            'n_classifiers': [1,3,5],
                            'depth': [2,4,6],
                            'n_filters': [8, 16, 32],
                            'batch_size': [8, 16, 32],
                            'n_epochs': [2, 5, 10, 20]
                   }}, 
                   "RDST": {"model": RDSTClassifier(), "params": {
                            'max_shapelets': [10, 100, 500, 1000, 10000],
                            'estimator': [None, LogisticRegression(max_iter = 1000)]
                   }}, 
                   "Weasel":{"model": WEASEL_V2(), "params": {
                            'min_window': [2,4, 8, 16]
                   }}, 
                   "RSTSF": {"model": RSTSF(), "params": {
                            'n_estimators': [50, 100, 200, 500],
                            'n_intervals': [10, 50, 100, 200]
                   }}, 
                   "FreshPRINCE": {"model":FreshPRINCEClassifier(), "params":{
                       'default_fc_parameters': ['minimal', 'efficient', 'comprehensive'],
                       'base_estimator': [None, LogisticRegression()],
                       'pca_solver': ['auto', 'full', 'covariance_eigh', 'arpack', 'randomized'],
                       'n_estimators': [50, 100, 200, 500],
                       'random_state': [13]
                   }}, 
                   "PF": {"model":ProximityForest(), "params":{
                       'n_trees': [10, 50, 100],
                       'n_splitters': [1,3,5],
                       'max_depth': [3,5,10,20]
                   }}}

    if model_name not in classifiers:
        raise ValueError(f"Model {model_name} is not supported for Bayesian Optimization.")

    clf = classifiers[model_name]['model']
    params = classifiers[model_name]['params']

    #Hyperparameter Gridsearch
    opt = GridSearchCV(
        estimator = clf,
        param_grid= params,
        scoring = 'accuracy',
        cv = 5
    )
    opt.fit(X,y)
    df = pd.DataFrame(opt.cv_results_)
    df = df.sort_values(by='rank_test_score')
    return df.head(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a time series classification model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate.")
    args = parser.parse_args()

    data = np.load('/scratch/jru34/minimal_train_test_arrays.npz')
    X = data['X_test']
    y = data['y_test']
    del data

    model_name = args.model_name
    df = optimize_params(X, y, model_name)
    save_path = os.path.join(PARENT_PATH, f'{model_name}_hyperparameters.csv')
    df.to_csv(save_path, index=False)
