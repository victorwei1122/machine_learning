from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import mlflow.xgboost
import os

app = FastAPI(title="Fraud Detection API")

base_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_dir, 'models', 'xgboost_fraud_model')
scaler_path = os.path.join(base_dir, 'models', 'scaler.joblib')

class TransactionItem(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

try:
    model = mlflow.xgboost.load_model(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Warning: Model or scaler not found. {e}")

@app.post("/predict")
def predict_fraud(transaction: TransactionItem):
    data = pd.DataFrame([transaction.dict()])
    
    # Scale Amount
    data['Amount'] = scaler.transform(data['Amount'].values.reshape(-1, 1))
    
    try:
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]
        
        return {
            "fraud_prediction": int(prediction),
            "fraud_probability": float(probability),
            "risk_status": "HIGH RISK" if prediction == 1 else "LOW RISK"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
