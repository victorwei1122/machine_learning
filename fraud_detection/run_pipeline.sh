#!/bin/bash
# run_pipeline.sh - Transaction Fraud Detection End-to-End Execution

set -e # Exit immediately if a command exits with a non-zero status.

echo "================================================="
echo "🕵️‍♂️ RUNNING FRAUD DETECTION MLOPS PIPELINE"
echo "================================================="

echo -e "\n[1/4] Fetching Raw Dataset..."
python src/fetch_data.py

echo -e "\n[2/4] Cleaning Data..."
python src/data_cleaning.py

echo -e "\n[3/4] Engineering Features & Fitting Scaler..."
python src/feature_engineering.py

echo -e "\n[4/4] Training XGBoost Model & Logging to MLflow..."
python src/train.py

echo -e "\n================================================="
echo "✅ BATCH PIPELINE COMPLETE!"
echo "To start the real-time inference FastAPI server, run:"
echo "uvicorn src.app:app --reload"
echo "================================================="
