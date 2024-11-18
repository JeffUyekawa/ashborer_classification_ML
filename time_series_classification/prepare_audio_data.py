#%%
import numpy as np
from time import time 
import sys
from torch.utils.data import  DataLoader
import pandas as pd

sys.path.insert(1, r"C:\Users\jeffu\Documents\Ash Borer Project\pre_processing")
from custom_dataset_class_96k import borer_data


def prepare_data(loader):
    for i, (input, label) in enumerate(loader):
        input, label = input.numpy(), label.numpy()
        if i == 0:
            ts = input
            lab = label
        else:
            ts = np.concatenate((ts,input))
            lab = np.concatenate((lab,label))
        
    return ts, lab
def filter_labeled_data(in_path, out_path):
    df = pd.read_csv(in_path)
    filtered_df = df.drop_duplicates(subset=['File','Start'], keep = 'first')
    path = out_path
    filtered_df.to_csv(path, index = False)
    return path
#%%
if __name__ == '__main__':
    train_temp = r"C:\Users\jeffu\Documents\Recordings\temp_path_train.csv"
    test_temp = r"C:\Users\jeffu\Documents\Recordings\temp_path_test.csv"
    TRAIN_ANNOTATION = filter_labeled_data(r"C:\Users\jeffu\Documents\Recordings\new_training_data.csv", train_temp)
    VAL_ANNOTATION = filter_labeled_data(r"C:\Users\jeffu\Documents\Recordings\test_set_labels.csv", test_temp)
    TRAIN_AUDIO = r"C:\Users\jeffu\Documents\Recordings\new_training"
    VAL_AUDIO = r"C:\Users\jeffu\Documents\Recordings\test_set"

    train_dataset = borer_data(TRAIN_ANNOTATION,TRAIN_AUDIO,spec=False)
    val_dataset = borer_data(VAL_ANNOTATION, VAL_AUDIO, spec = False)


    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = False)
    val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = False)
    print('Preparing training data')
    start = time()
    X_train, y_train = prepare_data(train_loader)
    end = time()
    print(f'Training data prepared in {end-start:.2f} seconds')
    print('Preparing test data')
    start = time()
    X_test, y_test = prepare_data(val_loader)
    end = time()
    print(f'Test data prepared in {end-start:.2f} seconds')
# %%
np.savez(r"C:\Users\jeffu\Documents\Ash Borer Project\time_series_classification\filtered_train_test_arrays.npz", X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)
# %%