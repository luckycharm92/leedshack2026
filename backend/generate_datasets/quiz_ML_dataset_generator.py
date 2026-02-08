import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Setting up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'datasets', 'quiz_training_data.csv')
VAL_PATH = os.path.join(BASE_DIR, 'datasets', 'quiz_validation_data.csv')

def generate_weighted_quiz_data(n_samples=12000):
    np.random.seed(42)
    
    # 1. Generate Raw Features
    density = np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.1, 0.4, 0.4, 0.1])
    alcohol = np.random.choice(['Light', 'Moderate', 'Heavy'], n_samples, p=[0.7, 0.2, 0.1])
    hrt = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    early_period = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    late_meno = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    child_after_30 = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    hyperplasia = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    lcis = np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    benign = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    symptoms = np.random.choice([0, 1], n_samples, p=[0.96, 0.04])

    # 2. APPLY RESEARCH WEIGHTS (THE GROUND TRUTH)
    # We start at 1.0 and multiply based on your researched decimals
    risk = np.ones(n_samples)
    
    # Density Multipliers (A is baseline)
    risk *= np.where(density == 'B', 1.2, 1.0)
    risk *= np.where(density == 'C', 1.5, 1.0)
    risk *= np.where(density == 'D', 3.0, 1.0) 
    
    # Alcohol Multipliers 
    risk *= np.where(alcohol == 'Moderate', 1.23, 1.0)
    risk *= np.where(alcohol == 'Heavy', 1.60, 1.0)
    
    # Clinical/Hormonal Factors
    risk *= np.where(hrt == 1, 1.35, 1.0)
    risk *= np.where(early_period == 1, 1.15, 1.0)
    risk *= np.where(late_meno == 1, 1.30, 1.0)
    risk *= np.where(child_after_30 == 1, 1.40, 1.0)
    
    # High-Risk Conditions & Symptoms
    risk *= np.where(hyperplasia == 1, 4.0, 1.0)
    risk *= np.where(lcis == 1, 8.5, 1.0) 
    risk *= np.where(benign == 1, 1.6, 1.0)
    risk *= np.where(symptoms == 1, 5.0, 1.0)

    # 3. Create DataFrame
    df = pd.DataFrame({
        'density': density, 
        'alcohol': alcohol, 
        'hrt': hrt,
        'early_period': early_period, 
        'late_meno': late_meno,
        'child_after_30': child_after_30, 
        'hyperplasia': hyperplasia,
        'lcis': lcis, 
        'benign': benign, 
        'symptoms': symptoms,
        'quiz_risk_multiplier': np.round(risk, 2)
    })

    # 4. Split and Save
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    train, val = train_test_split(df, test_size=0.2, random_state=42)
    
    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)
    
    print(f"✅ Generated {n_samples} samples.")
    print(f"📁 Training: {TRAIN_PATH}")
    print(f"📁 Validation: {VAL_PATH}")
    
    return df # Returns the full dataframe if you need it in the same script

if __name__ == "__main__":
    generate_weighted_quiz_data()