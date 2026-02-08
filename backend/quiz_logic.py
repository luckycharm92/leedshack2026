import pandas as pd
import xgboost as xgb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'quiz_risk_model.json')

def get_quiz_risk_multiplier(nhs_number, data):
    if not os.path.exists(MODEL_PATH):
        return 1.0

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    # Aggregate symptoms
    has_physical_symptoms = 1 if (
        int(data.get('lumps', 0)) == 1 or 
        int(data.get('pain', 0)) == 1 or 
        int(data.get('skin_change', 0)) == 1
    ) else 0

    # Prepare input
    input_row = pd.DataFrame([{
        'density': {'A': 0, 'B': 1, 'C': 2, 'D': 3}.get(data.get('density'), 0),
        'alcohol': {'Light': 0, 'Moderate': 1, 'Heavy': 2}.get(data.get('alcohol'), 0),
        'hrt': int(data.get('hrt', 0)),
        'early_period': int(data.get('early_period', 0)),
        'late_meno': int(data.get('late_meno', 0)),
        'child_after_30': int(data.get('child_after_30', 0)),
        'hyperplasia': int(data.get('hyperplasia', 0)),
        'lcis': int(data.get('lcis', 0)),
        'benign': int(data.get('benign', 0)),
        'symptoms': has_physical_symptoms 
    }])

    # Get the raw multiplier (e.g., 1.35 or 4.2)
    multiplier = model.predict(input_row)[0]
    
    # Return rounded to 2 decimal places, minimum of 1.0
    return round(max(1.0, multiplier), 2)