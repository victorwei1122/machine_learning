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
    loan_percent_income: float = None  # Will be calculated if not provided
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
    
    # Create a dictionary from the request
    data_dict = request.dict()
    
    # Auto-calculate loan_percent_income if not provided
    if data_dict.get('loan_percent_income') is None:
        data_dict['loan_percent_income'] = data_dict['loan_amnt'] / data_dict['person_income']
    
    # Create a DataFrame from the dictionary
    input_data = pd.DataFrame([data_dict])
    
    # Ensure columns are in the exact order the preprocessor expects
    # (The order must match the training set features)
    ordered_cols = [
        'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
        'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 
        'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length'
    ]
    input_data = input_data[ordered_cols]
    
    # Preprocess the input
    try:
        input_processed = preprocessor.transform(input_data)
        
        # Prepare for prediction (the model was fitted on named features)
        # We wrap in a DataFrame with clean names to avoid the UserWarning/ValueError
        numerical_cols = input_data.select_dtypes(exclude=['object']).columns.tolist()
        categorical_cols = input_data.select_dtypes(include=['object']).columns.tolist()
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        all_feature_names = numerical_cols + list(cat_feature_names)
        
        input_processed_df = pd.DataFrame(input_processed, columns=all_feature_names)
        
        # Predict
        prediction = model.predict(input_processed_df)[0]
        probability = model.predict_proba(input_processed_df)[0][1]
        
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
