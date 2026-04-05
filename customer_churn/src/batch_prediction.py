import pandas as pd
import joblib
import mlflow.xgboost
import os

def run_batch_prediction(input_csv: str, output_csv: str):
    """Simulates a nightly batch job scoring the customer base."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', 'xgboost_churn_model')
    preprocessor_path = os.path.join(base_dir, 'models', 'preprocessor.joblib')
    
    print("Loading model and preprocessor...")
    model = mlflow.xgboost.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    print(f"Loading raw data for inference from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Normally we would have IDs to attach predictions back to
    # For this dataset, if customerID exists, we keep it for output
    
    # 1. Clean
    df_clean = df.copy()
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)
    if 'customerID' in df_clean.columns:
        ids = df_clean['customerID']
        df_clean.drop('customerID', axis=1, inplace=True)
    else:
        ids = pd.Series(range(len(df_clean)), name='customerID')
        
    if 'Churn' in df_clean.columns:
        df_clean.drop('Churn', axis=1, inplace=True)
        
    # 2. Engineer
    X_processed = preprocessor.transform(df_clean)
    
    # Retrieve feature names
    num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_feature_names)
    
    # Note: xgboost may need a DataFrame to match column names if trained with them
    # But usually it accepts numpy arrays if features match exactly in order.
    # We will pass the dataframe to be safe
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # 3. Predict
    print("Generating churn predictions...")
    probabilities = model.predict_proba(X_df)[:, 1]
    
    # 4. Save results
    results = pd.DataFrame({
        'customerID': ids,
        'Churn_Probability': probabilities,
        'Risk_Segment': ['High Risk' if p > 0.6 else ('Medium Risk' if p > 0.3 else 'Low Risk') for p in probabilities]
    })
    
    results.to_csv(output_csv, index=False)
    print(f"Batch prediction complete. Found {len(results[results['Risk_Segment'] == 'High Risk'])} High Risk customers.")
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_file = os.path.join(base_dir, 'data', 'customer_churn.csv')
    output_file = os.path.join(base_dir, 'data', 'batch_predictions_output.csv')
    
    run_batch_prediction(input_file, output_file)
