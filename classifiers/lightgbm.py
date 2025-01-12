import lightgbm as lgb
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

#### Train LightGBM Model ####
d_train=lgb.Dataset(trainX, label=trainy)

#### Specifying the parameter ####
params={}
params['learning_rate']=0.1
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='multiclass' #Multi-class target feature
params['metric']='multi_logloss' #metric for multi-class
params['max_depth']=10
params['num_leaves']=32
params['num_class']=2 #no.of unique values in the target class not inclusive of the end value
#params['is_unbalance']='true'
params['verbose']=1
clf=lgb.train(params,d_train,100)

#### Model Test ####
y_pred=clf.predict(testX)

y_pred = [np.argmax(line) for line in y_pred]
accuracy=accuracy_score(testy, y_pred)
recall = recall_score(testy, y_pred)
precision = precision_score(testy, y_pred)
f1 = f1_score(testy, y_pred)

print("Model Accuracy: ", accuracy," Precision: ", precision, " Recall: ", recall, " F1: ", f1)