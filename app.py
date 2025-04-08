from flask import Flask, render_template, request
import numpy as np
import joblib
import xgboost as xgb
import keras
import ydf
from tensorflow.keras.models import load_model
import pandas as pd
import os
# Load models and data
rf_model = joblib.load('models/rf_model.pkl')  # Random Forest
svm_model = joblib.load('models/svm_model.pkl')  # SVM
xgb_model = xgb.Booster()  # XGBoost
xgb_model.load_model('models/xgb_model.json')  # Load XGBoost model
ydf_model = ydf.load_model('models/ydf_model')  # YDF
ann_model = load_model("models/ann_best_model.keras")
X = joblib.load("X_columns.pkl")  # Feature columns

# Prediction function
def prediction_ml(Gender, Age, Pressure, CGPA, Satisfaction, Sleep, Dietary, Suicide, Study, Financial, History, Subject, model):
    values = np.zeros(len(X.columns))
    values[0] = Gender
    values[1] = Age
    values[2] = Pressure
    values[3] = CGPA
    values[4] = Satisfaction
    values[5] = Sleep
    values[6] = Dietary
    values[8] = Suicide
    values[9] = Study
    values[10] = Financial
    values[11] = History

    index = np.where(X.columns == Subject)[0][0]    
    values[index] = 1

    # Create a dictionary with the features and their values
    input_data = {X.columns[i]: values[i] for i in range(len(X.columns))}

    if isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(values.reshape(1, -1), feature_names=X.columns.tolist())
        return int(model.predict(dmatrix)[0] >= 0.5)  

    if isinstance(model, ydf.model.generic_model.GenericModel):
        input_df = pd.DataFrame([input_data])

        return model.predict(input_df)

    return model.predict(np.array([values]))

app = Flask(__name__)


app.config["DEBUG"] = False


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        Gender = int(request.form['gender'])
        Age = float(request.form['age'])
        Pressure = float(request.form['pressure'])
        CGPA = float(request.form['cgpa'])
        Satisfaction = float(request.form['satisfaction'])
        Sleep = int(request.form['sleep'])
        Dietary = int(request.form['dietary'])
        Suicide = int(request.form['suicide'])
        Study = float(request.form['study'])
        Financial = float(request.form['financial'])
        History = int(request.form['history'])
        Subject = request.form['subject']

        ann_pred = prediction_ml(Gender, Age, Pressure, CGPA, Satisfaction, Sleep, Dietary, Suicide, Study, Financial, History, Subject, ann_model)
        rf_pred = prediction_ml(Gender, Age, Pressure, CGPA, Satisfaction, Sleep, Dietary, Suicide, Study, Financial, History, Subject, rf_model)
        svm_pred = prediction_ml(Gender, Age, Pressure, CGPA, Satisfaction, Sleep, Dietary, Suicide, Study, Financial, History, Subject, svm_model)
        xgb_pred = prediction_ml(Gender, Age, Pressure, CGPA, Satisfaction, Sleep, Dietary, Suicide, Study, Financial, History, Subject, xgb_model)
        ydf_pred = prediction_ml(Gender, Age, Pressure, CGPA, Satisfaction, Sleep, Dietary, Suicide, Study, Financial, History, Subject, ydf_model)

        return render_template('index.html', ann_pred=ann_pred, rf_pred=rf_pred, svm_pred=svm_pred, xgb_pred=xgb_pred, ydf_pred=ydf_pred)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
