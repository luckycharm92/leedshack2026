import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, 'datasets', 'quiz_training_data.csv')
VAL_PATH = os.path.join(BASE_DIR, 'datasets', 'quiz_validation_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'quiz_risk_model.json')

def train_quiz_ai():
    # 1. Load Data
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    # 2. Preprocess: Map strings to numbers for the AI
    mappings = {
        'density': {'A': 0, 'B': 1, 'C': 2, 'D': 3},
        'alcohol': {'Light': 0, 'Moderate': 1, 'Heavy': 2}
    }
    
    for df in [train_df, val_df]:
        df['density'] = df['density'].map(mappings['density'])
        df['alcohol'] = df['alcohol'].map(mappings['alcohol'])

    # 3. Separate Features and Target
    X_train = train_df.drop('quiz_risk_multiplier', axis=1)
    y_train = train_df['quiz_risk_multiplier']
    X_val = val_df.drop('quiz_risk_multiplier', axis=1)
    y_val = val_df['quiz_risk_multiplier']

    # 4. Initialize & Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=3,
        objective='reg:squarederror'
    )

    print("🚀 Training Quiz-Specific Risk Model...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 5. Evaluate & Save
    preds = model.predict(X_val)
    print(f"✅ Training Complete! R² Score: {r2_score(y_val, preds):.4f}")
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)

if __name__ == "__main__":
    train_quiz_ai()