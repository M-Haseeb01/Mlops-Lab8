# app.py
import os
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

APP_ROOT = os.path.dirname(__file__)


# Load artifacts
MODEL_PATH = os.path.join(APP_ROOT, "model.pkl")

# Load only your model (ignore others for now)
model = joblib.load(MODEL_PATH)

# Define placeholder features manually (since we don't have metadata files)
feature_list = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
feature_field_map = {f: f for f in feature_list}
label_encoders = {}

# Build basic meta info for form inputs
feature_meta = [
    {"name": f, "field": f, "type": "numeric", "options": None}
    for f in feature_list
]


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", feature_meta=feature_meta)

@app.route("/predict", methods=["POST"])
def predict():
    # build input vector in same order as feature_list
    row = []
    
    print("Added a new feature")
    print("Added a second feature")
    for feat in feature_list:
        field = feature_field_map[feat]
        val = request.form.get(field)
        if val is None:
            return f"Missing value for {feat}", 400
        if feat in label_encoders:
            # safe: value should be one of label_encoders[feat].classes_
            le = label_encoders[feat]
            try:
                encoded = int(le.transform([val])[0])
            except Exception as e:
                return f"Unexpected categorical value for {feat}: {val}", 400
            row.append(encoded)
        else:
            # numeric
            try:
                row.append(float(val))
            except:
                return f"Invalid numeric value for {feat}: {val}", 400

    X = np.array(row).reshape(1, -1)
    pred = model.predict(X)[0]
    # format prediction
    try:
        pred_fmt = round(float(pred), 2)
    except:
        pred_fmt = str(pred)
    return render_template("result.html", prediction=pred_fmt)

# Optional JSON API
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    if not data:
        return {"error": "JSON payload required"}, 400
    row = []
    for feat in feature_list:
        if feat not in data:
            return {"error": f"Missing field: {feat}"}, 400
        val = data[feat]
        if feat in label_encoders:
            le = label_encoders[feat]
            try:
                encoded = int(le.transform([str(val)])[0])
            except Exception as e:
                return {"error": f"Invalid categorical value for {feat}: {val}"}, 400
            row.append(encoded)
        else:
            try:
                row.append(float(val))
            except:
                return {"error": f"Invalid numeric value for {feat}: {val}"}, 400
    X = np.array(row).reshape(1, -1)
    pred = model.predict(X)[0]
    return {"prediction": float(pred)}

if __name__ == "__main__":
    app.run(debug=True, port=5000)
