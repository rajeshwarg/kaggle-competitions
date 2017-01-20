from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

def linearRegression(X, Y):
    alg = LinearRegression();
    alg.fit(X, Y)
    joblib.dump(alg, 'model/linear_regression.pkl')
def logisticRegression(X, Y):
    alg = LogisticRegression(random_state=1)
    alg.fit(X, Y)
    joblib.dump(alg, 'model/logistic_regression.pkl')
def gradientboostingClassifier(X, Y):
    alg = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
    alg.fit(X, Y)
    joblib.dump(alg, 'model/gradient_boosting_classifier.pkl')
def svmClassifier(X, Y):
    alg = svm.SVC()
    alg.fit(X, Y)
    joblib.dump(alg, 'model/svm_classifier.pkl')
def randomforestClassifier(X, Y):
    alg = RandomForestClassifier(n_estimators=10)
    alg.fit(X, Y)
    joblib.dump(alg, 'model/random_forest_classifier.pkl')
def neuralNetwork(X, Y):
    model = Sequential()
    model.add(Dense(891, input_dim=9, init='uniform', activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(9, init='uniform', activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    X_train = X.as_matrix()
    Y_train = Y.as_matrix()
    model.fit(X_train, Y_train, nb_epoch=70, batch_size=10)
    model.save('model/neural_networks.h5')
