from sklearn.externals import joblib
import pandas as pd
from keras.models import load_model
from keras.models import model_from_json
import numpy as np

def saveResults(alg_name, test_data, predictors):
    alg = joblib.load('model/' + alg_name + '.pkl')
    predictions = alg.predict(test_data[predictors])
    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv("results/" + alg_name + ".csv", index=False)
def saveResultsNeuralNetworks(alg_name, test_data, predictors):
    model = load_model('model/' + alg_name + '.h5')
    predictions = model.predict(test_data[predictors].as_matrix(), batch_size=32)
    predictions_final = [round(x) for x in predictions]
    predictions_out = np.asarray(predictions_final)
    predictions_out = predictions_out.astype(int)
    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions_out
    })
    submission.to_csv("results/" + alg_name + ".csv", index=False)
