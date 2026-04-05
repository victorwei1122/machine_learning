#!/bin/bash
# run_pipeline.sh - Customer Churn Prediction End-to-End Execution

set -e # Exit immediately if a command exits with a non-zero status.

echo "================================================="
echo "📉 RUNNING CUSTOMER CHURN MLOPS PIPELINE"
echo "================================================="

echo -e "\n[1/5] Fetching Raw Dataset..."
python src/fetch_data.py

echo -e "\n[2/5] Cleaning Data..."
python src/data_cleaning.py

echo -e "\n[3/5] Engineering Features & Fitting Preprocessor..."
python src/feature_engineering.py

echo -e "\n[4/5] Training XGBoost Model & Logging to MLflow..."
python src/train.py

echo -e "\n[5/5] Running Nightly Batch Prediction..."
python src/batch_prediction.py

echo -e "\n================================================="
echo "✅ PIPELINE COMPLETE!"
echo "Check data/batch_predictions_output.csv for results."
echo "================================================="
