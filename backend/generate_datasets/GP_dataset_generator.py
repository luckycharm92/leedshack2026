import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# --- PATH LOGIC ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'leeds_gp_dataset.csv')

def get_female_names_from_gemini():
    fallback_names = [
        "Olivia", "Amelia", "Isla", "Ava", "Ivy", "Freya", "Lily", 
        "Florence", "Mia", "Willow", "Alice", "Sophie", "Ella", "Grace", "Zoe"
    ]

    try:
        api_key = os.getenv("GEMINI_API_KEY") 
        
        if not api_key:
            print("⚠️ API Key not found in .env. Using fallback list.")
            return ["Olivia", "Amelia", "Isla", "Ava", "Mia"]

        # New Client syntax for 'google-genai'
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents="List 100 common UK female names, comma separated. No intro text."
        )
        names = [n.strip() for n in response.text.split(',')]
        return names if len(names) > 10 else fallback_names
    
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return ["Olivia", "Amelia", "Isla", "Ava", "Mia"]

def generate_leeds_gp_dataset(n_patients=100):
    np.random.seed(42)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    # 1. Get names from Gemini (or fallback)
    uk_names = get_female_names_from_gemini()
    
    # Basic Identifiers
    nhs_numbers = [f"{np.random.randint(400, 500)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}" for _ in range(n_patients)]
    practice_codes = ['B82005', 'B82021', 'B82081'] 
    
    # Medical Features
    age = np.random.randint(18, 90, n_patients)
    sex = np.random.choice([0, 1], n_patients, p=[0.02, 0.98])
    imd_score = np.random.randint(1, 11, n_patients)
    
    genetics_snomed = np.random.choice(['0', '765057007', '412734009', '442525003'], 
                                        n_patients, p=[0.92, 0.03, 0.03, 0.02])
    
    mother_history = np.random.choice([0, 1], n_patients, p=[0.85, 0.15])
    sister_history = np.random.choice([0, 1], n_patients, p=[0.90, 0.10])
    relative_under_50 = np.random.choice([0, 1], n_patients, p=[0.80, 0.20])
    
    bmi_observation = np.random.normal(27.5, 5.5, n_patients).clip(17, 48)
    bmi_observation[np.random.choice(n_patients, int(n_patients*0.2), replace=False)] = np.nan
    
    smoking_status = np.random.choice(['266919005', '77176002'], n_patients, p=[0.78, 0.22])

    # --- ASSEMBLE DATAFRAME ---
    df = pd.DataFrame({
        'Patient_Name': [np.random.choice(uk_names) for _ in range(n_patients)], # Added Column
        'Registered_Practice': [np.random.choice(practice_codes) for _ in range(n_patients)],
        'Patient_NHS_Number': nhs_numbers,
        'patient_email': 'gracemboothe@gmail.com',
        'Last_Consultation_Date': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, n_patients), unit='D'),
        'age': age,
        'sex': sex,
        'imd_score': imd_score,
        'genetics_snomed': genetics_snomed,
        'mother_history': mother_history,
        'sister_history': sister_history,
        'relative_under_50': relative_under_50,
        'bmi_observation': np.round(bmi_observation, 1),
        'smoking_status': smoking_status
    })

    # Saving to the datasets folder
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Created '{DATA_PATH}' with {n_patients} patients and Gemini-generated names.")

if __name__ == "__main__":
    generate_leeds_gp_dataset(100)