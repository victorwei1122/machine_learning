from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
import lightgbm as lgb
from .data_cleaning import clean_mixed_types, handle_missing_values
from .feature_engineering import aggregate_household_features, create_derived_features

app = FastAPI(title="Costa Rican Poverty Prediction API")

# Global variables for models
MODEL_PATH = "/Users/zihanwei/Desktop/projects/machine_learning/segmentation/costa_rican_poverty/models"
classifier = None
kmeans = None
scaler = None

@app.on_event("startup")
async def load_models():
    global classifier, kmeans, scaler
    try:
        classifier = lgb.Booster(model_file=os.path.join(MODEL_PATH, 'lgbm_model.txt'))
        kmeans = joblib.load(os.path.join(MODEL_PATH, 'kmeans_model.joblib'))
        scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.joblib'))
    except Exception as e:
        print(f"Error loading models: {e}")

class IndividualData(BaseModel):
    # This should include all 142 features
    # For brevity in the example, we use a Dict[str, Any]
    data: Dict[str, Any]

class HouseholdData(BaseModel):
    members: List[Dict[str, Any]]

@app.post("/predict")
async def predict_poverty(household: HouseholdData):
    """Predict poverty level and cluster for a household."""
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # 1. Convert to DataFrame
    df = pd.DataFrame(household.members)
    
    # 2. Preprocess (Cleaning + Engineering)
    df = clean_mixed_types(df)
    df = handle_missing_values(df)
    df_eng = aggregate_household_features(df)
    df_eng = create_derived_features(df_eng)
    
    # Prepare features for model (drop IDs and Target)
    X = df_eng.drop(columns=['Id', 'idhogar', 'Target'], errors='ignore')
    
    # 3. Segmentation (Cluster)
    X_scaled = scaler.transform(X.fillna(0))
    cluster = kmeans.predict(X_scaled)[0]
    
    # 4. Classification (Poverty Level) - note: classifier is a Booster object
    # If using lgb.Booster, we use .predict()
    y_prob = classifier.predict(X)
    y_pred = int(np.argmax(y_prob, axis=1)[0]) + 1 # Convert back to 1-4 scale
    
    return {
        "poverty_level": y_pred,
        "cluster": int(cluster),
        "confidence": float(np.max(y_prob))
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": classifier is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
