# VitalScan AI 

AI-powered health risk predictor for Heart Disease 🫀, Diabetes 🩸, and Kidney Disease 🫘.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

## Project Structure

```
flask_app/
├── app.py                  # Flask application (routes + ML inference)
├── requirements.txt
├── models/                 # 🤖 Pre-trained model & scaler files
│   ├── diabetes_model.pkl
│   ├── diabetes_rf_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── heart_model.pkl
│   ├── heart_rf_model.pkl
│   ├── heart_scaler.pkl
│   ├── kidney_model.pkl
│   ├── kidney_rf_model.pkl
│   └── kidney_scaler.pkl
├── templates/
│   └── index.html          # Jinja2 template (SPA shell)
└── static/
    ├── style.css           # Dark/light theme styles
    └── script.js           # SPA logic, charts, chatbot
```

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Serves the frontend SPA |
| POST | `/predict/diabetes` | 🩸 Diabetes risk prediction |
| POST | `/predict/heart` | 🫀 Heart disease risk prediction |
| POST | `/predict/kidney` | 🫘 Kidney disease risk prediction |

### Request / Response format

All prediction endpoints accept JSON and return:

```json
{
  "prediction": 1,
  "ensemble_probability": 0.74,
  "lr_probability": 0.68,
  "rf_probability": 0.80
}
```

#### Diabetes payload
```json
{ "glucose": 120, "bmi": 25.0, "age": 45, "insulin": 79 }
```

#### Heart payload
```json
{
  "age": 50, "resting_blood_pressure": 120, "cholestoral": 200,
  "max_heart_rate": 150, "oldpeak": 1.0,
  "sex": "Male",
  "chest_pain_type": "Typical angina",
  "fasting_blood_sugar": "Lower than 120 mg/ml",
  "rest_ecg": "Normal",
  "exercise_induced_angina": "No",
  "slope": "Upsloping",
  "vessels_colored_by_flourosopy": "Zero",
  "thalassemia": "Normal"
}
```

#### Kidney payload
```json
{ "age": 50, "bp": 80, "bgr": 150, "bu": 40, "sc": 1.2, "hemo": 15 }
```

## Production

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```
