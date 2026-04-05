import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, f1_score
import os

def train_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'creditcard_engineered.csv')
    
    print("Loading engineered data...")
    df = pd.read_csv(data_path)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle massive class imbalance via scale_pos_weight
    # Ratio of negative class to positive class
    scale_pos_weight = y_train.value_counts()[0] / (y_train.value_counts()[1] + 1)
    
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(base_dir, 'mlflow.db')}")
    mlflow.set_experiment("Fraud_Detection")
    
    with mlflow.start_run():
        print("Training XGBoost with scale_pos_weight...")
        clf = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        clf.fit(X_train, y_train)
        
        # Inference
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Metrics
        auprc = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print(f"AUPRC (Average Precision): {auprc:.4f}")
        
        # MLflow Tracking
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_metric("auprc", auprc)
        mlflow.log_metric("f1_score", f1)
        
        # Save model
        model_dir = os.path.join(base_dir, 'models', 'xgboost_fraud_model')
        mlflow.xgboost.save_model(clf, model_dir)
        print(f"Model saved locally at: {model_dir}")

if __name__ == "__main__":
    train_model()
