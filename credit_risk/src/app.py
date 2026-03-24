from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Define the input schema
class CreditRequest(BaseModel):
    person_age: float
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float

# Initialize FastAPI app
app = FastAPI(title="Credit Risk Prediction API")

# Load models
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

MODEL_PATH = os.path.join(project_root, "models", "model.joblib")
PREPROCESSOR_PATH = os.path.join(project_root, "data", "processed", "preprocessor.joblib")

if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
else:
    model = None
    preprocessor = None
    print("Warning: Model or preprocessor not found!")

@app.post("/predict")
def predict(request: CreditRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Create a DataFrame from the request
    input_data = pd.DataFrame([request.dict()])
    
    # Preprocess the input
    try:
        input_processed = preprocessor.transform(input_data)
        
        # Predict
        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0][1]
        
        return {
            "loan_status_prediction": int(prediction),
            "default_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
