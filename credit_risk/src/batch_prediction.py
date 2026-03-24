import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report

def run_batch_prediction(input_path, output_path, model_path, preprocessor_path):
    """
    Runs batch prediction on a dataset.
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Separate target if present for evaluation
    if 'loan_status' in df.columns:
        X = df.drop('loan_status', axis=1)
        y_true = df['loan_status']
    else:
        X = df
        y_true = None
        
    print("Loading model and preprocessor...")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    print("Preprocessing and predicting...")
    X_processed = preprocessor.transform(X)
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1]
    
    # Add results to original dataframe
    result_df = df.copy()
    result_df['predicted_loan_status'] = predictions
    result_df['default_probability'] = probabilities
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Batch prediction results saved to {output_path}")
    
    # Evaluation (if target exists)
    if y_true is not None:
        print("\nBatch Evaluation Results:")
        print(f"Accuracy: {accuracy_score(y_true, predictions):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, predictions))
        
    return result_df

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Using the cleaned data as a "batch" for demonstration
    batch_input = os.path.join(project_root, "data", "processed", "credit_risk_cleaned.csv")
    batch_output = os.path.join(project_root, "data", "processed", "batch_results.csv")
    mod_path = os.path.join(project_root, "models", "model.joblib")
    prep_path = os.path.join(project_root, "data", "processed", "preprocessor.joblib")
    
    run_batch_prediction(batch_input, batch_output, mod_path, prep_path)
