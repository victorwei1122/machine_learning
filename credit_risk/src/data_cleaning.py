import pandas as pd
import numpy as np
import os

def clean_data(input_path, output_path):
    """
    Cleans the credit risk dataset:
    - Removes rows with suspicious age (> 90).
    - Removes rows with suspicious employment length (> 60).
    - Fills missing interest rates with median.
    - Fills missing employment length with 0.
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Outlier detection and removal
    # Age > 90 is suspicious in this context
    df = df[df['person_age'] <= 90]
    
    # Employment length > 60 is also suspicious (assuming years)
    df = df[df['person_emp_length'] <= 60]
    
    # 2. Handling Missing Values
    # person_emp_length: fill with 0
    df['person_emp_length'] = df['person_emp_length'].fillna(0)
    
    # loan_int_rate: fill with median
    median_int_rate = df['loan_int_rate'].median()
    df['loan_int_rate'] = df['loan_int_rate'].fillna(median_int_rate)
    
    # 3. Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    raw_path = os.path.join(project_root, "data", "credit_risk_dataset.csv")
    processed_path = os.path.join(project_root, "data", "processed", "credit_risk_cleaned.csv")
    
    clean_df = clean_data(raw_path, processed_path)
    print("\nSummary Statistics of Cleaned Data:")
    print(clean_df.describe())
    print("\nNull Values Check:")
    print(clean_df.isnull().sum())
