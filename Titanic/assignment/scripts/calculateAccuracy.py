from __future__ import division
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.externals import joblib

def calculateAccuracy(train_data, test_data, predictors, target_col, num_fold, alg_name):
    kf = KFold(n_splits=num_fold)
    predictions = []
    alg = joblib.load(alg_name + '.pkl') 
    for train, test in kf.split(train_data):
        train_predictors = (train_data[predictors].iloc[train, :])
        train_target = train_data[target_col].iloc[train]
        alg.fit(train_predictors, train_target)
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
    print "Accuracy of the model is: ", accuracy    
