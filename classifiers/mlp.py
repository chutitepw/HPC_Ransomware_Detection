# keras imports for the dataset and building our neural network
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
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

#### X Input [# of sliding windows for training,window size x # of events]
#### Y Input [# of sliding windows for training, data label for each sliding window can be 0 or 1] 
trainX = np.array(trainX).astype('int')
trainX = trainX.reshape(trainX.shape[0],-1)
trainy = np_utils.to_categorical(trainy, 2)

testX = np.array(testX).astype('int')
testX = testX.reshape(testX.shape[0],-1)
testy = np_utils.to_categorical(testy, 2)

print("Training Data Shape: ", trainX.shape,trainy.shape)
print("Testing Data Shape: ", testX.shape,testy.shape)

model = Sequential()
# hidden layer
model.add(Dense(100, input_shape=(trainX.shape[1],), activation='relu'))
# output layer
model.add(Dense(2, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

es = EarlyStopping(monitor='loss', verbose=1, patience=10)

# training the model
history = model.fit(trainX, trainy, batch_size=128, epochs=1000,callbacks=[es],validation_data=(testX, testy))

plot(history)