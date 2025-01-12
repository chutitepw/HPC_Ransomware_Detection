import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

import numpy as np
from utils import findfile, load_file, class_rebalance, plot

#### size of windows ####
TIME_STEPS=50

#### Find Files for Training/Testing ####
train_path = findfile('*.csv', '../data/train/')
test_path = findfile('*.csv', '../data/test/')

trainX, trainy = load_file(train_path, TIME_STEPS)
testX, testy = load_file(test_path, TIME_STEPS)

#### Classes rebalance if necessary ####
#trainX, trainy = class_rebalance(trainX, trainy, TIME_STEPS)

#### X Input [# of sliding windows for training,window size,# of events]
#### Y Input [data label for each sliding window can be 0 or 1] 
trainX = np.array(trainX).astype('float')
trainy = np_utils.to_categorical(trainy, 2)
testX = np.array(testX).astype('float')
testy = np_utils.to_categorical(testy, 2)

print("Training Data Shape: ", trainX.shape,trainy.shape)
print("Testing Data Shape: ", testX.shape,testy.shape)

model = Sequential()
model.add(LSTM(128, input_shape=(trainX.shape[1],trainX.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

history = model.fit(trainX, trainy, epochs=100, batch_size=128,callbacks=[callback],validation_data=(testX, testy))

plot(history)