from os import error
from flask import Flask, request
import numpy as np
import pickle
import symptoms_model
import symptoms_model_store

app = Flask(__name__)

isTrue = symptoms_model_store.symptoms_prediction()

@app.route("/", methods = ["POST"])
def hello():
    if request.method == "POST":
        data = request.json.values()
        symptoms_test = np.array(list(data))
        symptoms_test = symptoms_test.reshape((1,-1))
        try:
            if(isTrue == True):
                filename = 'trained_model.sav'
                loaded_model = pickle.load(open(filename , 'rb'))
                symptom = loaded_model.predict(symptoms_test)
                print("Saved_Model")
            else:
                symptom = symptoms_model.symptoms_prediction(symptoms_test)
                print("Unsaved_Model")
            return symptom[0] , 200
        except:
            return "Invalid Input", 400
    return 405

if __name__ == "__main__":
    app.run(debug=True)