from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_pkl(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    print(f"⚠️ Warning: {filename} not found in models/.")
    return None


diabetes_lr = load_pkl("diabetes_model.pkl")
diabetes_rf = load_pkl("diabetes_rf_model.pkl")
diabetes_scaler = load_pkl("diabetes_scaler.pkl")

heart_lr = load_pkl("heart_model.pkl")
heart_rf = load_pkl("heart_rf_model.pkl")
heart_scaler = load_pkl("heart_scaler.pkl")

kidney_lr = load_pkl("kidney_model.pkl")
kidney_rf = load_pkl("kidney_rf_model.pkl")
kidney_scaler = load_pkl("kidney_scaler.pkl")


def get_prediction(lr_model, rf_model, scaler, feature_array):
    if lr_model is None or rf_model is None or scaler is None:
        return None, "Models not loaded properly."

    scaled  = scaler.transform([feature_array])
    lr_prob = float(lr_model.predict_proba(scaled)[0][1])
    rf_prob = float(rf_model.predict_proba(scaled)[0][1])
    ensemble_prob = (lr_prob + rf_prob) / 2
    prediction = int(ensemble_prob > 0.5)

    return {
        "prediction": prediction,
        "ensemble_probability": ensemble_prob,
        "lr_probability": lr_prob,
        "rf_probability": rf_prob,
    }, None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    data = request.get_json(force=True)
    try:
        features = [
            float(data["glucose"]),
            float(data["bmi"]),
            float(data["age"]),
            float(data["insulin"]),
        ]
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    result, err = get_prediction(diabetes_lr, diabetes_rf, diabetes_scaler, features)
    if err:
        return jsonify({"error": err}), 500
    return jsonify(result)


@app.route("/predict/heart", methods=["POST"])
def predict_heart():
    data = request.get_json(force=True)
    try:
        f = [0.0] * 22
        f[0] = float(data["age"])
        f[1] = float(data["resting_blood_pressure"])
        f[2] = float(data["cholestoral"])
        f[3] = float(data["max_heart_rate"])
        f[4] = float(data["oldpeak"])

        sex = data.get("sex", "")
        cp = data.get("chest_pain_type", "")
        fbs = data.get("fasting_blood_sugar", "")
        ecg = data.get("rest_ecg", "")
        exang = data.get("exercise_induced_angina", "")
        slope = data.get("slope", "")
        vessels = data.get("vessels_colored_by_flourosopy", "")
        thal = data.get("thalassemia", "")

        if sex == "Male": f[5]  = 1.0
        if cp == "Atypical angina": f[6]  = 1.0
        if cp == "Non-anginal pain": f[7]  = 1.0
        if cp == "Typical angina": f[8]  = 1.0
        if fbs == "Lower than 120 mg/ml": f[9]  = 1.0
        if ecg == "Normal": f[10] = 1.0
        if ecg == "ST-T wave abnormality": f[11] = 1.0
        if exang == "Yes": f[12] = 1.0
        if slope == "Flat": f[13] = 1.0
        if slope == "Upsloping": f[14] = 1.0
        if vessels == "One": f[15] = 1.0
        if vessels == "Three": f[16] = 1.0
        if vessels == "Two": f[17] = 1.0
        if vessels == "Zero": f[18] = 1.0
        if thal == "No": f[19] = 1.0
        if thal == "Normal": f[20] = 1.0
        if thal == "Reversable Defect": f[21] = 1.0

    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    result, err = get_prediction(heart_lr, heart_rf, heart_scaler, f)
    if err:
        return jsonify({"error": err}), 500
    return jsonify(result)


@app.route("/predict/kidney", methods=["POST"])
def predict_kidney():
    data = request.get_json(force=True)
    try:
        features = [
            float(data["age"]),
            float(data["bp"]),
            float(data["bgr"]),
            float(data["bu"]),
            float(data["sc"]),
            float(data["hemo"]),
        ]
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    result, err = get_prediction(kidney_lr, kidney_rf, kidney_scaler, features)
    if err:
        return jsonify({"error": err}), 500
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
