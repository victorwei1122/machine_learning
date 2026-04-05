import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

def train_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'customer_churn_engineered.csv')
    
    print("Loading engineered data...")
    df = pd.read_csv(data_path)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(base_dir, 'mlflow.db')}")
    mlflow.set_experiment("Customer_Churn")
    
    with mlflow.start_run():
        print("Training XGBoost...")
        clf = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        
        clf.fit(X_train, y_train)
        
        # Inference
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        
        # MLflow Tracking
        mlflow.log_params({
            "model_type": "XGBoost",
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.05
        })
        
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc
        })
        
        # Save model
        model_dir = os.path.join(base_dir, 'models', 'xgboost_churn_model')
        mlflow.xgboost.save_model(clf, model_dir)
        print(f"Model saved locally at: {model_dir}")

if __name__ == "__main__":
    train_model()
