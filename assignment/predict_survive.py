from __future__ import division
from scripts.read_data import *
from scripts.prepareFeatures import *
from scripts.selectFeatures import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

train_data, test_data = readData()
train_data, test_data = prepareFeatures(train_data, test_data)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "NameLength", "FamilySize"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors]
    #[LogisticRegression(random_state=1), predictors]
]
kf = KFold(n_splits=3)
predictions = []
for alg, predictors in algorithms:
    for train, test in kf.split(train_data):
        train_predictors = (train_data[predictors].iloc[train, :])
        train_target = train_data["Survived"].iloc[train]
        alg.fit(train_predictors, train_target)
        test_predictions = alg.predict(train_data[predictors].iloc[test, :])
        predictions.append(test_predictions)

# Calculate the accuracy of the model
#predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
predictions = predictions.astype(int)
match = (predictions == train_data["Survived"])
match_df = pd.DataFrame({'match':match})
match_count = len(match_df[match_df['match'] == 1])
accuracy = (match_count/predictions.size)
print "Accuracy of the model is: ", accuracy


#train_X = selectFeatures(train_data, predictors, target_col="Survived", feature_count=5)





