import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
import traceback
import pandas as pd
import sys
import json
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import column
from model import data_prep

# Your API definition
app = Flask(__name__)

# get required pickle files
xgb_model = pickle.load(open('models/xgb_model.pickle','rb'))
scaler = pickle.load(open('models/std_scaler.pickle','rb'))
ohe = pickle.load(open('models/ohe.pickle','rb'))

@app.route('/welcome', methods =['GET', 'POST'])
def welcome():
    return render_template("welcome.html")

@app.route('/predict', methods=['GET', 'POST'])
def heart_prediction():
    patient_inputs = request.form.to_dict()
    data = pd.DataFrame([patient_inputs.values()], columns=patient_inputs.keys())
    print(data, file=sys.stderr)

    data = data.apply(pd.to_numeric)

    # one hot encoding
    data['cp'] = data['cp'].astype(str)
    new_data = data_prep(data, ohe, False)

    #standard scaling
    new_data_std = scaler.transform(new_data)

    print(f"{new_data_std}", file=sys.stderr)
    prediction = xgb_model.predict(new_data_std)

    return render_template("prediction.html", result = diagnosis(prediction))

def diagnosis(prediction):
    if prediction[0]:
        return "Your heart might be at risk. As a precaution, you should see your doctor immediately."
    else:
        return "You are not in the risk group. But do not forget your yearly routine doctor appointment."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)