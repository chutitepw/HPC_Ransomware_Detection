import os
import fnmatch
import pandas as pd
import numpy as np
import imblearn
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

def findfile(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

#### Load training file ####
def load_file(file_path, window_size):
    sequenceX, sequenceY = [], []
    for files in file_path:
        #df = pd.read_csv(files,nrows=100)
        df = pd.read_csv(files)
        #### Prep data sequence for each file ####:
        for i in range(0, len(df)-window_size):
            df_slice = df.iloc[i:i+window_size]
            sequenceX.append(df_slice)
            if df_slice.iloc[-1,-1] == 'ransom':
                sequenceY.append(1)
            else:
                sequenceY.append(0)
    print("Finsish loading file length: ",len(sequenceX))
    sequenceX, sequenceY = np.array(sequenceX), np.array(sequenceY)
    sequenceX = sequenceX[:,:, 0:sequenceX.shape[2]-1]
    
    return sequenceX, sequenceY

#### Balance classes ####
def class_rebalance(sequenceX, sequenceY, window_size):
    sequenceX = sequenceX.reshape(sequenceX.shape[0],-1)
    rus = RandomUnderSampler(random_state=42, replacement=True)
    x_rus, y_rus = rus.fit_resample(sequenceX,sequenceY)
    x_rus = x_rus.reshape(x_rus.shape[0],window_size,x_rus.shape[1]//window_size)
    print("Classes rebalance: ",x_rus.shape,y_rus.shape)
    val, count = np.unique(y_rus, return_counts=True)
    # of classes
    for v, c in zip(val,count):
        print(f"{v}: {c}")

    return x_rus, y_rus

def plot(history):

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()