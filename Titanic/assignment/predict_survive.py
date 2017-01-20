from __future__ import division
from scripts.read_data import *
from scripts.prepareFeatures import *
from scripts.selectFeatures import *
from scripts.algorithms import *
from scripts.calculateAccuracy import *
from scripts.saveResults import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

train_data, test_data = readData()
train_data, test_data = prepareFeatures(train_data, test_data)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "NameL", "FSize"]

linearRegression(train_data[predictors], train_data["Survived"])
logisticRegression(train_data[predictors], train_data["Survived"])
gradientboostingClassifier(train_data[predictors], train_data["Survived"])
svmClassifier(train_data[predictors], train_data["Survived"])
randomforestClassifier(train_data[predictors], train_data["Survived"])
neuralNetwork(train_data[predictors], train_data["Survived"])

calculateAccuracy(train_data, predictors, "Survived", 3, 'linear_regression')
calculateAccuracy(train_data, predictors, "Survived", 3, 'logistic_regression')
calculateAccuracy(train_data, predictors, "Survived", 3, 'gradient_boosting_classifier')
calculateAccuracy(train_data, predictors, "Survived", 3, 'svm_classifier')
calculateAccuracy(train_data, predictors, "Survived", 3, 'random_forest_classifier')
#calculateAccuracyNeuralNetwork(train_data, predictors, "Survived", 3, 'neural_networks')

saveResults('linear_regression', test_data, predictors)
saveResults('logistic_regression', test_data, predictors)
saveResults('gradient_boosting_classifier', test_data, predictors)
saveResults('svm_classifier', test_data, predictors)
saveResults('random_forest_classifier', test_data, predictors)
saveResultsNeuralNetworks('neural_networks', test_data, predictors)



