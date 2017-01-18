from __future__ import division
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Process training data. fill empty age values, change sex value to 0 or 1, change embarked value to 0, 1 or 2
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())
train_data.loc[train_data["Sex"] == "male", "Sex"] = 0
train_data.loc[train_data["Sex"] == "female", "Sex"] = 1
train_data["Embarked"] = train_data["Embarked"].fillna("S")
train_data.loc[train_data["Embarked"] == "S", "Embarked"] = 0
train_data.loc[train_data["Embarked"] == "C", "Embarked"] = 1
train_data.loc[train_data["Embarked"] == "Q", "Embarked"] = 2

# Process testing data. fill empty age values, change sex value to 0 or 1, change embarked value to 0, 1 or 2, fill empty fare values.
test_data["Age"] = test_data["Age"].fillna(train_data["Age"].median())
test_data.loc[test_data["Sex"] == "male", "Sex"] = 0
test_data.loc[test_data["Sex"] == "female", "Sex"] = 1
test_data["Embarked"] = test_data["Embarked"].fillna("S")
test_data.loc[test_data["Embarked"] == "S", "Embarked"] = 0
test_data.loc[test_data["Embarked"] == "C", "Embarked"] = 1
test_data.loc[test_data["Embarked"] == "Q", "Embarked"] = 2
test_data["Fare"] = test_data["Fare"].fillna(train_data["Fare"].median())

# Select fields to be used for training the model
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize Linear Regression Algorithm
alg = LinearRegression()
kf = KFold(n_splits=3)
predictions = []

for train, test in kf.split(train_data):
    train_predictors = (train_data[predictors].iloc[train, :])
    train_target = train_data["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(train_data[predictors].iloc[test, :])
    predictions.append(test_predictions)

# Calculate the accuracy of the model
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
match = (predictions == train_data["Survived"])
match_df = pd.DataFrame({'match':match})
match_count = len(match_df[match_df['match'] == 1])
accuracy = (match_count/predictions.size)
print "Accuracy of the model is: ", accuracy

# Initialize logistic regression algorithm
alg = LogisticRegression(random_state=1)
scores = cross_val_score(alg, train_data[predictors], train_data["Survived"], cv=3)
print "Cross validation score is: ", scores

# fit the logistic regression algorithm
alg.fit(train_data[predictors], train_data["Survived"])
# predict the output for testing data
predictions = alg.predict(test_data[predictors])
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("kaggle_lr.csv", index=False)

train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"]
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"]

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Age", "Embarked"]]
]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(train_data[predictors], train_data["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(test_data[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
predictions = predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("kaggle.csv", index=False)