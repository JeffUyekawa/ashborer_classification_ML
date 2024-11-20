#%%
#Import necessary packages
import numpy as np
import pandas as pd
from aeon.classification.convolution_based import RocketClassifier, Arsenal
import time

def filter_labeled_data(in_path):
    df = pd.read_csv(in_path)
    filtered_df = df.drop_duplicates(subset=['File','Start'], keep = 'first')
    path = r"C:\Users\jeffu\Documents\Recordings\temp_path.csv"
    filtered_df.to_csv(path, index = False)
    return path
#%%
#Make predictions on test set
data = np.load(r"C:\Users\jeffu\Documents\Ash Borer Project\time_series_classification\filtered_train_test_arrays.npz")
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
clf = Arsenal(num_kernels=100, random_state=13)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = np.load(r"C:\Users\jeffu\Documents\Ash Borer Project\time_series_classification\cross_val_arrays.npz")
X_cv = data['X_test']
y_cv = data['y_test']
y_pred = clf.predict(X_cv)

#%%
# Compute the confusion matrix
cm = confusion_matrix(y_cv, y_pred)

# Use ConfusionMatrixDisplay to visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Test Set Predictions Confusion Matrix')
plt.show()

# %%
from sklearn.metrics import f1_score
#Calculate accuracy, precision, and recall
TN = cm[0,0]
TP = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

acc = (TP + TN)/(TP + TN + FP + FN)
prec = (TP)/(TP + FP)
recall = (TP)/(TP + FN)
f1 = f1_score(y_cv,y_pred)
print(f'Model Accuracy: {acc} | Precision: {prec} | Recall: {recall} | F1: {f1}')
#%%
#Need to figure out a way to graph these next
from IPython.display import Audio, display, clear_output
import os
import torchaudio as ta
AUDIO_PATH = r"C:\Users\jeffu\Documents\Recordings\new_test"
AUDIO_LABELS =filter_labeled_data(r"C:\Users\jeffu\Documents\Recordings\new_test_data.csv")
df = pd.read_csv(AUDIO_LABELS)
df['Prediction'] = y_pred
grouped = df.groupby('File')

for i, (file, group) in enumerate(grouped):
    path = os.path.join(AUDIO_PATH,file)
    y, fs = ta.load(path)
    if fs != 96000:
            start = int(5*fs)
            y = y[:,start:]
            resampler = ta.transforms.Resample(fs,96000)
            y = resampler(y)
            fs = 96000
    t = np.arange(y.shape[1])/fs
    display(Audio(y[0].numpy(), rate = fs))
    plt.plot(t,y[0].numpy())
    plt.title(file)

    pos = group[(group['Prediction']==1) | (group['Label']==1)]
    colors = []
    for i, row in pos.iterrows():
        start = row['Start']
        end = row['End']
        
        if (row['Label'] == 1) & (row['Prediction']==0):
            if 'black' not in colors:
                plt.plot(t[start:end],y[0,start:end].numpy(),color='black', label ='False Negative')
                colors.append('black')
            else:
                plt.plot(t[start:end],y[0,start:end].numpy(),color='black')

        elif (row['Label'] == 0)& (row['Prediction'] == 1):
            if 'purple' not in colors:
                plt.plot(t[start:end],y[0,start:end].numpy(),color='purple', label = 'False Positive')
                colors.append('purple')
            else:
                plt.plot(t[start:end],y[0,start:end].numpy(),color='purple')
        else:
            if 'red' not in colors:
                plt.plot(t[start:end],y[0,start:end].numpy(),color='r', label = 'True Positive')
                colors.append('red')
            else:
                plt.plot(t[start:end],y[0,start:end].numpy(),color='r')

    plt.legend()
    plt.show()
    


# %%
