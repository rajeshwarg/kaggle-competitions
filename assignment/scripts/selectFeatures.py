from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def selectFeatures(train_data, predictors, target_col, feature_count):
    train_X = train_data[predictors]
    train_Y = train_data["Survived"]
    train_X_new = SelectKBest(chi2, k=5).fit_transform(train_X, train_Y)
    return train_X_new
