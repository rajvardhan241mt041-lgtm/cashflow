import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "latest_xgboost_model.json")

def train_and_save_model(df: pd.DataFrame):
    """
    Trains XGBoost on the cleaned dataframe and saves the model.
    Target: Days_Overdue_Delay
    """
    if 'Days_Overdue_Delay' not in df.columns:
        print("Error: Target column 'Days_Overdue_Delay' not found in dataset.")
        return False
        
    y = df['Days_Overdue_Delay']
    X = df.drop(columns=['Days_Overdue_Delay'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    print("Training XGBoost Regressor...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model Training Complete. Mean Absolute Error: {mae}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return True
