import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, 'datasets', 'general_training_data.csv')
VAL_PATH = os.path.join(BASE_DIR, 'datasets', 'general_validation_data.csv')

def generate_general_ML_data(n_samples=12000):
    np.random.seed(42)
    
    # SNOMED CT Mappings for your features
    genetics_snomed = {'None': 0, '765057007': 1, '412734009': 2, 'PALB2': 3}
    smoking_snomed = {0: 266919005, 1: 77176002} # Non-smoker vs Smoker

    age = np.random.randint(18, 90, n_samples)
    sex = np.random.choice([0, 1], n_samples, p=[0.02, 0.98])
    imd_score = np.random.randint(1, 11, n_samples) # 1=Most Deprived, 10=Least
    
    # Generate Coded Genetics
    genetics = np.random.choice(['0', '765057007', '412734009', 'PALB2'], n_samples, p=[0.96, 0.015, 0.015, 0.01])
    
    # Family History 
    mother = np.random.choice([0, 1], n_samples, p=[0.88, 0.12])
    sister = np.random.choice([0, 1], n_samples, p=[0.92, 0.08])
    under_50 = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    # Lifestyle with missing data 
    bmi = np.random.normal(26, 5, n_samples).clip(17, 45)
    smoking = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    # Introduce 10% Missing Values 
    bmi[np.random.choice(n_samples, int(n_samples*0.1), replace=False)] = np.nan
    
    # Risk Calculation
    risk = np.ones(n_samples)
    risk *= np.where(genetics == '765057007', 10.5, 1.0) # BRCA1
    risk *= np.where((mother+sister >= 1) & (under_50 == 1), 2.0, 1.0)
    risk *= np.where(sex == 0, 0.01, 1.0)

    df = pd.DataFrame({
        'age': age, 'sex': sex, 'imd_score': imd_score,
        'genetics_snomed': genetics, 'mother_history': mother,
        'sister_history': sister, 'relative_under_50': under_50,
        'bmi_observation': np.round(bmi, 1), 'smoking_status': smoking,
        'target_relative_risk': np.round(risk, 2)
    })

    # Encode genetics for the model
    df['genetics_snomed'] = df['genetics_snomed'].replace({'PALB2': '442525003'}) # Add SNOMED for PALB2
    
    train, val = train_test_split(df, test_size=0.3, random_state=42)
    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)
    print("✅ Generated GP-coded datasets with simulated missing data.")

generate_general_ML_data()