import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import os

def train_model(features_path, target_path, model_dir):
    """
    Trains a Random Forest model and logs experiments to MLflow.
    """
    print(f"Loading features from {features_path}...")
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).values.ravel()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MLflow tracking
    mlflow.set_experiment("Credit Risk Project")
    
    with mlflow.start_run():
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Print results
        print("\nModel Evaluation:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Log model artifact
        mlflow.sklearn.log_model(model, "model", registered_model_name="CreditRiskRFModel")
        
        # Save locally as well
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "model.joblib"))
        print(f"Model saved locally to {model_dir}")
        
    return model

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    feat_path = os.path.join(project_root, "data", "processed", "features.csv")
    targ_path = os.path.join(project_root, "data", "processed", "target.csv")
    model_directory = os.path.join(project_root, "models")
    
    train_model(feat_path, targ_path, model_directory)
