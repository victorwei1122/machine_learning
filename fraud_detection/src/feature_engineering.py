import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib
import os

def engineer_features(input_path: str, output_path: str, scaler_path: str):
    """Scales features using RobustScaler (very important for outliers)."""
    print("Loading cleaned data for feature engineering...")
    df = pd.read_csv(input_path)
    
    # The V1-V28 features are already PCA transformed.
    # Amount is not. We should scale it. 
    # RobustScaler is less prone to outliers.
    scaler = RobustScaler()
    
    # Fit and transform 'Amount'
    # Reshaping is necessary for scaler
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Save scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Saved preprocessor to {scaler_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    engineer_features(
        os.path.join(base_dir, 'data', 'creditcard_cleaned.csv'),
        os.path.join(base_dir, 'data', 'creditcard_engineered.csv'),
        os.path.join(base_dir, 'models', 'scaler.joblib')
    )
