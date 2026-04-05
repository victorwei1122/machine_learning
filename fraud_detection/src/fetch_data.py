import ssl
import pandas as pd
from sklearn.datasets import fetch_openml
import os

def fetch_credit_card_fraud():
    print("Downloading Credit Card Fraud dataset from OpenML (this may take a minute)...")
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # ID 1597 corresponds to the Kaggle Credit Card Fraud Detection dataset
    data = fetch_openml(data_id=1597, as_frame=True)
    df = data.frame
    
    os.makedirs(os.path.join(os.path.dirname(__file__), '../data'), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), '../data/creditcard.csv')
    
    df.to_csv(out_path, index=False)
    print(f"✅ Downloaded successfully and saved to {os.path.abspath(out_path)}")

if __name__ == "__main__":
    fetch_credit_card_fraud()
