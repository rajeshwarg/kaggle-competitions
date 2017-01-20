from __future__ import division
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from keras.models import load_model

def calculateAccuracy(train_data, predictors, target_col, num_fold, alg_name):
    kf = KFold(n_splits=num_fold)
    predictions = []
    alg = joblib.load('model/' + alg_name + '.pkl') 
    for train, test in kf.split(train_data):
        train_predictors = (train_data[predictors].iloc[train, :])
        train_target = train_data[target_col].iloc[train]
        #alg.fit(train_predictors, train_target)
        test_predictions = alg.predict(train_data[predictors].iloc[test, :])
        predictions.append(test_predictions)
    predictions = np.concatenate(predictions, axis=0)
    predictions[predictions > .5] = 1
    predictions[predictions <= .5] = 0
    predictions = predictions.astype(int)
    match = (predictions == train_data[target_col])
    match_df = pd.DataFrame({'match':match})
    match_count = len(match_df[match_df['match'] == 1])
    accuracy = (match_count/predictions.size)
    print "Accuracy of the model for " + alg_name + " is: ", accuracy
    train_data["Predicted"] = predictions
    train_data = train_data.drop('Name', 1)
    train_data = train_data.drop('Ticket', 1)
    train_data = train_data.drop('Cabin', 1)
    train_data = train_data.drop('PassengerId', 1)
    train_data = train_data.drop('SibSp', 1)
    train_data = train_data.drop('Parch', 1)
    print(train_data.loc[train_data["Predicted"] != train_data["Survived"]])

def calculateAccuracyNeuralNetwork(train_data, predictors, target_col, num_fold, alg_name):
    kf = KFold(n_splits=num_fold)
    predictions = []
    model = load_model('model/' + alg_name + '.h5')
    for train, test in kf.split(train_data):
        train_predictors = (train_data[predictors].iloc[train, :])
        train_target = train_data[target_col].iloc[train]
        test_predictions = model.predict_classes(train_data[predictors].iloc[test, :].as_matrix(), batch_size=32)
        predictions.append(test_predictions)
    predictions = np.concatenate(predictions, axis=0)
    predictions[predictions > .5] = 1
    predictions[predictions <= .5] = 0
    predictions = predictions.astype(int)
    match = (predictions == train_data[target_col])
    match_df = pd.DataFrame({'match':match})
    match_count = len(match_df[match_df['match'] == 1])
    accuracy = (match_count/predictions.size)
    print "Accuracy of the model for " + alg_name + " is: ", accuracy
    train_data["Predicted"] = predictions
    train_data = train_data.drop('Name', 1)
    train_data = train_data.drop('Ticket', 1)
    train_data = train_data.drop('Cabin', 1)
    train_data = train_data.drop('PassengerId', 1)
    train_data = train_data.drop('SibSp', 1)
    train_data = train_data.drop('Parch', 1)
    print(train_data.loc[train_data["Predicted"] != train_data["Survived"]])    
