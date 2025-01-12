import xgboost as xgb
from sklearn.metrics import precision_score,recall_score, accuracy_score, f1_score

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
#### Y Input [# of sliding windows label for training] --> data label for each sliding window can be 0 or 1
trainX = trainX.reshape(trainX.shape[0],-1)
testX = testX.reshape(testX.shape[0],-1)

print("Training Data Shape: ", trainX.shape,trainy.shape)
print("Testing Data Shape: ", testX.shape,testy.shape)

# Train XGBoost Model
dtrain_reg = xgb.DMatrix(trainX, trainy, enable_categorical=True)
dtest_reg = xgb.DMatrix(testX, testy, enable_categorical=True)
# #Specifying the parameter
params = {
            'objective':'multi:softmax',
            'max_depth': 10,
            'learning_rate': 0.1,
            'max_leaves':32,
            'num_class': 2
}
model = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=100,
    verbose_eval=10,
    # Activate early stopping
    #early_stopping_rounds=10
)

#### Model Test ####
y_pred=model.predict(dtest_reg)
accuracy=accuracy_score(testy, y_pred)
recall = recall_score(testy, y_pred)
precision = precision_score(testy, y_pred)
f1 = f1_score(testy, y_pred)

print("Model Accuracy: ", accuracy," Precision: ", precision, " Recall: ", recall, " F1: ", f1)