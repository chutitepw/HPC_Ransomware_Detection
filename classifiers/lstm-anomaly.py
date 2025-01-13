import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

tf.compat.v1.random.set_random_seed(1)

import numpy as np
from utils import findfile, load_file_anomaly, create_sequences, threshold_calculation, plot, plot_anomaly

TIME_STEPS=50

#### Find Files for Training/Testing ####
train_path = findfile('benign-1.csv', '../data/train')
test_path = findfile('benign-*.csv', '../data/test')
test_path2 = findfile('ransom-*.csv', '../data/test')

X_train, y_train, scaler = load_file_anomaly(train_path, TIME_STEPS, 1)
X_test, y_test, scaler = load_file_anomaly(test_path, TIME_STEPS, 0, scaler)
X_test2, y_test2, scaler = load_file_anomaly(test_path2, TIME_STEPS, 0, scaler)
print("Training Data Shape: ", X_train.shape,y_train.shape)
print("Testing Benign Data Shape: ", X_test.shape,y_test.shape)
print("Testing Malicious Data Shape: ", X_test2.shape,y_test2.shape)

model = Sequential()

model.add(LSTM(128, activation='relu', return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(128, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(X_train.shape[2]))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mae')
model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2, shuffle=False, callbacks=[callback])

plot(history)

#### Find threshold ####
X_train_pred = model.predict(X_train, verbose=0)
train_mae_loss = np.abs(X_train_pred - y_train)

threshold = threshold_calculation(train_mae_loss)

#### Test Model ####
X_test_pred = model.predict(X_test, verbose=0)
test_mae_loss = np.abs(X_test_pred-y_test)

plot_anomaly(test_mae_loss, X_test_pred.shape[1], threshold)

X_test_pred2 = model.predict(X_test2, verbose=0)
test_mae_loss2 = np.abs(X_test_pred2-y_test2)

plot_anomaly(test_mae_loss2, X_test_pred2.shape[1], threshold)
