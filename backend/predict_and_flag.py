import pandas as pd
import xgboost as xgb
import numpy as np
import os
GENERAL_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'breast_cancer_model.json')

def run_gp_screening():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'models', 'breast_cancer_model.json') # UPDATED
    dataset_path = os.path.join(BASE_DIR, 'datasets', 'leeds_gp_dataset.csv') # UPDATED
    report_path = os.path.join(BASE_DIR, 'datasets', 'flagged_patients_report.csv')

    if not os.path.exists(model_path):
        print("❌ Error: Model file not found. Run your training script first!")
        return
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}. Run your GP generator script first!")
        return

    df = pd.read_csv(dataset_path)

    relevant_features = [
    'age', 'sex', 'imd_score', 'genetics_snomed',
    'mother_history', 'sister_history', 'relative_under_50',
    'bmi_observation', 'smoking_status'
    ]

    X_live = df[relevant_features].copy()
    genetics_map = {'0': 0, '765057007': 1, '412734009': 2, '442525003': 3}
    smoking_map = {'266919005': 0, '77176002': 1}

    X_live['genetics_snomed'] = X_live['genetics_snomed'].astype(str).map(genetics_map)
    X_live['smoking_status'] = X_live['smoking_status'].astype(str).map(smoking_map)

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    df['predicted_relative_risk'] = model.predict(X_live)


    # Apply flagging logic (Threshold of 2.0x risk)
    df['screening_status'] = '✅ Routine'

    # 3. IDENTIFY LIFESTYLE/AI RISK (Lowered threshold to 1.5 for better demo visibility)

    ai_high_risk = df['predicted_relative_risk'] > 1.5
    df.loc[ai_high_risk, 'screening_status'] = '🚨 URGENT: High Predicted Risk'

    # 2. Genetic Override: Even if AI score is lower, certain codes ARE High Risk

    # 765057007=BRCA1, 412734009=BRCA2, 442525003=PALB2

    df['genetics_snomed'] = df['genetics_snomed'].astype(str)
    high_risk_codes = ['765057007', '412734009', '442525003']
    is_genetic = df['genetics_snomed'].isin(high_risk_codes)

    # Use string addition to keep BOTH flags if both are true!

    df.loc[is_genetic, 'screening_status'] += ' + 🧬 Genetic Marker'
    df['screening_status'] = df['screening_status'].str.replace('✅ Routine + ', '')

    # 3. Missing Data Flag: High risk score but no BMI
    missing_bmi = (df['predicted_relative_risk'] > 1.2) & (df['bmi_observation'].isna())
    df.loc[missing_bmi, 'screening_status'] = '⚠️ REVIEW: High Risk + Missing BMI'

    output_cols = [
        'Patient_Name', 
        'Patient_NHS_Number', 
        'patient_email', 
        'age', 
        'predicted_relative_risk', 
        'screening_status'
    ]
    # 9. Output the "Action List" for the GP
    print("\n--- LEEDS GP SURGERY: BREAST CANCER SCREENING ACTION LIST ---")

    # Sort by risk so the most urgent cases are at the top
    action_list = df[df['screening_status'] != '✅ Routine'].sort_values(by='predicted_relative_risk', ascending=False)
    print(action_list[output_cols])
    # Save the results to a new file for the GP to open in Excel
    action_list.to_csv(report_path, index=False)
    print(f"\n💾 Report saved: {len(action_list)} patients flagged for review.")

    return action_list 

if __name__ == "__main__":
    run_gp_screening()