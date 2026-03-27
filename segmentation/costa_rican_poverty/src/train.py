import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb

def train_segmentation(df, n_clusters=4):
    """Perform K-Means clustering for household segmentation."""
    print(f"Training K-Means with {n_clusters} clusters...")
    
    # Select features for clustering (exclude ID and Target)
    features = df.drop(columns=['Id', 'idhogar', 'Target'], errors='ignore')
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.fillna(0))
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    
    return kmeans, scaler, clusters

def train_classification(df):
    """Train a LightGBM classifier to predict poverty levels."""
    print("Training LightGBM classifier...")
    
    # Prepare data
    X = df.drop(columns=['Id', 'idhogar', 'Target'], errors='ignore')
    y = df['Target'] - 1 # LightGBM expects 0-indexed labels (0, 1, 2, 3)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize MLflow
    mlflow.set_experiment("Costa_Rican_Poverty_Prediction")
    
    with mlflow.start_run():
        # Parameters for LightGBM
        params = {
            'objective': 'multiclass',
            'num_class': 4,
            'metric': 'multi_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1,
            'seed': 42
        }
        
        # Train model
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        # Predictions
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Metrics
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"Macro F1 Score: {f1:.4f}")
        
        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("macro_f1", f1)
        mlflow.lightgbm.log_model(model, "model")
        
        return model, f1

def main(input_path, model_path):
    """Main training pipeline."""
    df = pd.read_csv(os.path.join(input_path, 'train_engineered.csv'))
    
    # 1. Segmentation
    kmeans, scaler, clusters = train_segmentation(df)
    df['cluster'] = clusters
    
    # 2. Classification
    model, f1 = train_classification(df)
    
    # Save artifacts
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(kmeans, os.path.join(model_path, 'kmeans_model.joblib'))
    joblib.dump(scaler, os.path.join(model_path, 'scaler.joblib'))
    # model is already saved/logged via mlflow, but we can save it locally too
    model.save_model(os.path.join(model_path, 'lgbm_model.txt'))
    
    print(f"Models saved to {model_path}")

if __name__ == "__main__":
    base_path = "/Users/zihanwei/Desktop/projects/machine_learning/segmentation/costa_rican_poverty"
    main(os.path.join(base_path, 'data/processed'), os.path.join(base_path, 'models'))
