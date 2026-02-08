import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, 'datasets', 'general_train_data.csv')
VAL_PATH = os.path.join(BASE_DIR, 'datasets', 'general_validation_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'breast_cancer_model.json')

def train_model():
    if not os.path.exists(TRAIN_PATH):
        print(f"❌ Error: {TRAIN_PATH} not found. Check your generator script!")
        return

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    # 2. Updated SNOMED Mapping
    # We use strings because CSVs often load long codes as objects/strings
    genetics_map = {
        '0': 0,               # None
        '765057007': 1,       # BRCA1
        '412734009': 2,       # BRCA2
        '442525003': 3        # PALB2
    }
    
    for df in [train_df, val_df]:
        df['genetics_snomed'] = df['genetics_snomed'].astype(str).map(genetics_map)

    # 3. Separate Features and Target
    X_train = train_df.drop('target_relative_risk', axis=1)
    y_train = train_df['target_relative_risk']
    
    X_val = val_df.drop('target_relative_risk', axis=1)
    y_val = val_df['target_relative_risk']

    # 4. XGBoost Training (Automatic Missing Value Handling)
    # We no longer need SimpleImputer. XGBoost learns the 'default' path for NaNs.
    model = xgb.XGBRegressor(
        n_estimators=1000, # how many learning rounds my model gets
        learning_rate=0.03, # rate is 0.03 is low so medical patterns without over-reacting to outliers
        max_depth=2, #Shallow trees prevent the model from memorizing specific rows and keep it focused on major patterns
        early_stopping_rounds=50, # if model doesnt improve for 50 rounds it stops
        objective='reg:squarederror'
    )

    print("🚀 Training Professional GP Model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=200 # Shows progress every 100 rounds
    )

    # 5. Evaluate
    predictions = model.predict(X_val)
    print(f"\n✅ Training Complete!")
    print(f"📊 MAE: {mean_absolute_error(y_val, predictions):.4f}")
    print(f"📊 R² Score: {r2_score(y_val, predictions):.4f}")

    # 6. Save & Visualise
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"💾 Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    # %%
    train_model()