import pandas as pd
import os

def load_and_clean_data(input_path: str, output_path: str):
    """Loads raw data, performs basic cleaning, and saves."""
    print("Loading raw dataset...")
    df = pd.read_csv(input_path)
    
    # 1. Drop duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"Dropped {initial_shape[0] - df.shape[0]} duplicate rows.")
    
    # 2. Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        df = df.fillna(df.median())
        print(f"Filled {missing} missing values with median.")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    load_and_clean_data(
        os.path.join(base_dir, 'data', 'creditcard.csv'), 
        os.path.join(base_dir, 'data', 'creditcard_cleaned.csv')
    )
