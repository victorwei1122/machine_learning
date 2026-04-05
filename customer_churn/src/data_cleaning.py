import pandas as pd
import numpy as np
import os

def load_and_clean_data(input_path: str, output_path: str):
    print("Loading raw dataset...")
    df = pd.read_csv(input_path)
    
    # 1. TotalCharges is object, convert to numeric
    # Coerce errors to NaN and fill with 0 (since they represent new customers with 0 tenure usually)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # 2. Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    load_and_clean_data(
        os.path.join(base_dir, 'data', 'customer_churn.csv'), 
        os.path.join(base_dir, 'data', 'customer_churn_cleaned.csv')
    )
