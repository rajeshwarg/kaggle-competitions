from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.externals import joblib

def linearRegression(X, Y):
    alg = LinearRegression();
    alg.fit(X, Y)
    joblib.dump(alg, 'linear_regression.pkl')
def logisticRegression(X, Y):
    alg = LogisticRegression(random_state=1)
    alg.fit(X, Y)
    joblib.dump(alg, 'logistic_regression.pkl')
def gradientboostingClassifier(X, Y):
    alg = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
    alg.fit(X, Y)
    joblib.dump(alg, 'gradient_boosting_classifier.pkl')
def svmClassifier(X, Y):
    alg = svm.SVC()
    alg.fit(X, Y)
    joblib.dump(alg, 'svm_classifier.pkl')
