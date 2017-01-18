
def prepareFeatures(train_data, test_data):
    
    train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())
    test_data["Age"] = test_data["Age"].fillna(train_data["Age"].median())

    train_data.loc[train_data["Sex"] == "male", "Sex"] = 0
    train_data.loc[train_data["Sex"] == "female", "Sex"] = 1
    test_data.loc[test_data["Sex"] == "male", "Sex"] = 0
    test_data.loc[test_data["Sex"] == "female", "Sex"] = 1

    train_data["Fare"] = train_data["Fare"].fillna(train_data["Fare"].median())
    test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].median())
    
    train_data["Embarked"] = train_data["Embarked"].fillna("S")
    train_data.loc[train_data["Embarked"] == "S", "Embarked"] = 0
    train_data.loc[train_data["Embarked"] == "C", "Embarked"] = 1
    train_data.loc[train_data["Embarked"] == "Q", "Embarked"] = 2
    test_data["Embarked"] = test_data["Embarked"].fillna("S")
    test_data.loc[test_data["Embarked"] == "S", "Embarked"] = 0
    test_data.loc[test_data["Embarked"] == "C", "Embarked"] = 1
    test_data.loc[test_data["Embarked"] == "Q", "Embarked"] = 2
    
    train_data["NameLength"] = train_data["Name"].apply(lambda x: len(x))
    test_data["NameLength"] = test_data["Name"].apply(lambda x: len(x))

    train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"]
    test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"]

    return train_data, test_data
