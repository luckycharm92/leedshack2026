from flask import Flask, request, jsonify
from flask_cors import CORS
from quiz_logic import get_quiz_risk_multiplier
import pandas as pd
import xgboost as xgb
import json
import os

app = Flask(__name__)
CORS(app) 

# BASE_DIR logic to find /datasets from the root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_FOLDER = os.path.join(BASE_DIR, 'datasets')

@app.route('/api/check-risk', methods=['POST'])
def check_risk():
    try:
        data = request.get_json()
        nhs_number = data.get('nhs_number')
        
        if not nhs_number:
            return jsonify({'error': 'NHS number is required'}), 400
        
        # 1. Load the CSV report
        csv_path = os.path.join(DATASETS_FOLDER, 'flagged_patients_report.csv')
        if not os.path.exists(csv_path):
            return jsonify({'error': 'Report file not found. Run predict_and_flag.py first.'}), 500
        
        df = pd.read_csv(csv_path)

        # 2. Search for patient (strip any extra whitespace/formatting)
        # Assuming your CSV column is 'Patient_NHS_Number'
        patient_found = df[df['Patient_NHS_Number'].astype(str) == str(nhs_number)]
        
        if not patient_found.empty:
            row = patient_found.iloc[0]
            
            # 3. Calculate Risk Percentage (Relative Risk to % increase)
            # 1.0 = Average, 1.5 = 50% increase, etc.
            rel_risk = row['predicted_relative_risk']
            risk_pct = round((rel_risk - 1.0) * 100, 2) if rel_risk > 1.0 else 0

            # 4. Generate Feature Breakdown (The "Why")
            # We look for features that are higher than the baseline 
            # This returns a 1D array of tuples: [("Feature", Impact_Decimal), ...]
            impacts = []
            
            # Logic: If Age > 50, Genetics != 0, or BMI > 25, we add them as contributors
            if row['age'] > 50:
                impacts.append(("Age Factor", 0.25))
            if str(row['genetics_snomed']) != '0':
                impacts.append(("Genetic Marker Found", 0.60))
            if row['bmi_observation'] > 25:
                impacts.append(("BMI/Lifestyle", 0.15))
            
            # Sort by impact (highest first)
            impacts.sort(key=lambda x: x[1], reverse=True)

            return jsonify({
                'success': True,
                'is_at_risk': True,
                'predicted_relative_risk': round(rel_risk, 2),
                'risk_percentage': f"{risk_pct}%",
                'feature_breakdown': impacts, # 1D array with tuples
                'message': 'High risk detected'
            }), 200
        
        else:
            return jsonify({
                'success': True,
                'is_at_risk': False,
                'risk_percentage': "0%",
                'feature_breakdown': [],
                'message': 'Patient not flagged as at-risk'
            }), 200
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/submit-quiz', methods=['POST'])
def handle_quiz():
    try:
        data = request.get_json() # Get JSON from frontend
        nhs_id = data.get('nhs_number')
        msg = ""
        
        # Pass the whole data dictionary to your logic file
        raw_multiplier = get_quiz_risk_multiplier(nhs_id, data)

        # Force rounding to 2 decimal places
        risk_multiplier = "{:.2f}".format(raw_multiplier)

        if raw_multiplier <= 1.1:
            msg = "Your risk is consistent with the general population baseline. "
        else:
            msg = f"Based on your inputs, your risk is {risk_multiplier}x higher than the average person. "

        has_any_symptom = any([data.get('lumps'), data.get('pain'), data.get('skin_change')])
        if has_any_symptom:
            msg += "Because you reported physical symptoms, please consult your GP immediately."
        
        return jsonify({
            'success': True,
            'nhs_number': nhs_id,
            'risk_multiplier': f"{risk_multiplier}x",
            'message': msg
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)