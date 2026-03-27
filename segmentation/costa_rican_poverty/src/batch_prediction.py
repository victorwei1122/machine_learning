import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
from .data_cleaning import clean_data
from .feature_engineering import engineer_features

def batch_predict(input_csv, output_csv, model_path):
    """Perform batch prediction on an unseen dataset."""
    print(f"Loading models from {model_path}...")
    classifier = lgb.Booster(model_file=os.path.join(model_path, 'lgbm_model.txt'))
    kmeans = joblib.load(os.path.join(model_path, 'kmeans_model.joblib'))
    scaler = joblib.load(os.path.join(model_path, 'scaler.joblib'))
    
    # 1. Cleaning & Engineering
    # We call our pipeline functions to process the new data
    # (Simplified for the demonstration - assume input_csv is raw individual data)
    print("Preprocessing data...")
    # (Logic to handle individual -> household aggregation)
    # For now, we assume the preprocessing pipeline is run separately or we wrap it here
    
    # Let's assume the user points us to a raw CSV
    # We'll need a temp directory for processed intermediate files
    temp_dir = "/tmp/costa_poverty_batch"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy input_csv to temp_dir as 'train.csv' (since our scripts look for that)
    # This is a bit hacky, in production we would refactor the cleaning/engineering to be more flexible.
    
    # (Skipping the file copy for brevity, assuming df is already engineered)
    df = pd.read_csv(input_csv)
    
    # 2. Extract Features
    X = df.drop(columns=['Id', 'idhogar', 'Target'], errors='ignore')
    
    # 3. Predict
    print("Generating predictions...")
    # Clusters
    X_scaled = scaler.transform(X.fillna(0))
    clusters = kmeans.predict(X_scaled)
    
    # Poverty Levels
    y_prob = classifier.predict(X)
    y_pred = np.argmax(y_prob, axis=1) + 1 # 1-4 scale
    
    # 4. Save results
    results = pd.DataFrame({
        'Id': df['Id'],
        'Household_Id': df['idhogar'],
        'Poverty_Prediction': y_pred,
        'Cluster_Assignment': clusters
    })
    
    results.to_csv(output_csv, index=False)
    print(f"Batch prediction results saved to {output_csv}")

if __name__ == "__main__":
    model_path = "/Users/zihanwei/Desktop/projects/machine_learning/segmentation/costa_rican_poverty/models"
    test_path = "/Users/zihanwei/Desktop/projects/machine_learning/segmentation/costa_rican_poverty/data/processed/test_engineered.csv"
    output_path = "/Users/zihanwei/Desktop/projects/machine_learning/segmentation/costa_rican_poverty/data/processed/test_predictions.csv"
    
    if os.path.exists(test_path):
        batch_predict(test_path, output_path, model_path)
    else:
        print("Engineered test data not found. Run clean_data and engineer_features first.")
