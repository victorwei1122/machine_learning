# 🕵️‍♂️ Transaction Fraud Detection Pipeline

## 📖 Overview
In banking, monitoring transactions for fraud is critical for minimizing financial losses and protecting customer trust. This project implements a real-time transaction fraud detection MLOps pipeline. The dataset includes highly imbalanced credit card transactions, mimicking real-world banking environments where fraudulent transactions represent <0.5% of total volume.

## 🎯 Business Use Case
- **Problem**: Detecting fraudulent transactions instantly while minimizing false positives (which degrade customer experience).
- **Solution**: A robust Machine Learning model deployed via FastAPI for real-time scoring.
- **Value**: Prevents immediate financial loss and automates the transaction review process.

---

## 🧠 Why XGBoost? (Model Selection & Intuition)

### How XGBoost Works
XGBoost (eXtreme Gradient Boosting) is an ensemble learning method that builds a series of decision trees sequentially. Unlike Random Forest (which builds independent trees), XGBoost ensures that each new tree specifically tries to correct the residual errors made by all the previous trees combined. 

### Why It Excels in Fraud Detection
Fraud is notoriously difficult to model because it represents an **extreme class imbalance** (in this dataset, only ~0.17% of transactions are fraud). Normal machine learning algorithms will often achieve 99% accuracy by simply predicting "Not Fraud" every single time, rendering them useless.

We chose XGBoost because it allows us to handle this imbalance natively without artificially inflating the dataset (e.g., using SMOTE, which is very slow on large datasets). We do this using the `scale_pos_weight` parameter. 
- During training, if the model misclassifies a normal transaction, the penalty is small.
- If the model misclassifies an actual Fraudulent transaction, `scale_pos_weight` acts as a multi-hundredfold multiplier on the error penalty.
- This mathematically forces the gradient descent algorithm to prioritize learning the microscopic fraudulent patterns.

---

## pipeline ⚙️ MLOps Pipeline Steps

### 1. Data Cleaning (`data_cleaning.py`)
- **Duplicates**: Raw transactional data often contains duplicate logs. We identify and drop these to prevent data leakage during train/test splits.
- **Imputation**: Any rare missing values are securely filled using the median to prevent code breakage in production.

### 2. Feature Engineering (`feature_engineering.py`)
- **Handling Outliers (RobustScaler vs StandardScaler)**: Standard averages are highly sensitive to outliers (if one millionaire enters a room, the average net worth skyrockets, crushing the mathematical scale of all normal people). Financial transactions inherently have these massive outliers (e.g., a $100,000 corporate purchase vs. a $2 coffee), meaning `StandardScaler` fails. Instead, we use `RobustScaler` on the `Amount` feature. It relies on the Median and Interquartile Range, meaning it completely ignores outliers. The normal data retains its healthy shape, while the outlier gets scaled to an enormous number, highlighting it beautifully as an anomaly to the model.
- **Persistence**: The fitted scaler is saved to `models/scaler.joblib` so that real-time API requests are scaled using the exact same mathematics.

### 3. Model Training (`train.py`)
- We utilize **MLflow** to track the experiment metadata.
- We evaluate the model using **Average Precision (AUPRC)** rather than standard accuracy or ROC-AUC, because AUPRC is highly sensitive to the minority Class precision/recall trade-off. 
- The champion model is serialized into an MLflow artifact folder.

### 4. Real-Time Deployment (`app.py`)
- A **FastAPI** web server loads the MLflow-tracked XGBoost model upon startup.
- It exposes a `/predict` JSON endpoint. As transactions stream in, the API transforms the raw amounts via the `scaler.joblib`, immediately queries the model, and returns a binary `High Risk` or `Low Risk` flag in milliseconds.

---

## 📊 Evaluation Metrics Explained
When testing the model, you will see a Classification Report with specific results for **Class 1 (Fraud)**. Here is how to interpret those metrics operationally:

- **AUPRC (0.76)**: The Area Under the Precision-Recall Curve. For a highly imbalanced dataset (where fraud is <0.2%), an AUPRC of 0.76 is outstanding and proves the model is genuinely learning fraud patterns, not just guessing "not fraud" to cheat the accuracy score.
- **Recall (0.85)**: Out of all the transactions that were *actually* fraudulent, the model successfully caught 85% of them. This protects the bank from major financial theft (False Negatives).
- **Precision (0.42)**: When the model explicitly flags a transaction as Fraud, it is correct 42% of the time (meaning the other 58% were legitimate transactions). In banking operations, freezing 58 legitimate cards to successfully catch 42 actual fraudsters is generally considered a highly successful tradeoff to stop widespread financial bleeding.
- **Support (95)**: The raw count of actual fraudulent transactions in the test set.

---

## 💼 Executive Presentation & Business Buy-In
If presenting this pipeline to non-technical stakeholders or risk committees, focus on the **cost-benefit trade-off** rather than the machine learning architecture:

1. **The Cost of False Positives**: Emphasize that declining a legitimate customer's card at the register causes severe friction and brand damage. A model tuned blindly for accuracy might freeze too many healthy accounts.
2. **The Cost of False Negatives**: Allowing fraudulent transactions through results in direct financial chargeback losses for the bank.
3. **The Compromise**: Explain that this MLOps pipeline does not rely on a rigid "one-size-fits-all" rule. By utilizing advanced probabilistic scoring (`scale_pos_weight`), the business is empowered to choose the exact dollar-value risk threshold where they want to trigger an alert. This allows the bank to safely catch >85% of fraud while only marginally increasing the manual review queues, protecting both the P&L and customer loyalty.

---

## 🛠 Project Structure
- `data/`: Contains the down-loaded Kaggle `creditcard.csv` dataset.
- `src/`: Modular code for the pipeline described above.
- `app.py`: FastAPI application.
- `notebooks/`: Interactive walkthroughs combining all steps for educational tracing.

## 🚀 Running the Pipeline
1. Fetch data: `python src/fetch_data.py`
2. Run notebooks: Load `notebooks/01_fraud_detection_mlops.ipynb` inside Jupyter
3. Train model: `python src/train.py`
4. Serve model: `uvicorn src.app:app --reload`

### 🐳 Enterprise Deployment (Docker)
To deploy this cleanly and scale it out in production, we have fully containerized the API. From the `fraud_detection` directory, run:
1. **Build the Image**: `docker build -t fraud-api .`
2. **Run the Container**: `docker run -p 8000:8000 fraud-api`
The FastAPI application is now safely isolated and running!

---

### 🧪 Testing the Real-Time API
Whether you are running it natively via `uvicorn` (Step 4) or via the **Docker container**, you can open a new terminal window and send a simulated transaction to the API to see the model score it in real-time. 

Run this `curl` command to test a sample (normal) transaction:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37, "V5": -0.33,
  "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36, "V10": 0.09,
  "V11": -0.55, "V12": -0.61, "V13": -0.99, "V14": -0.31, "V15": 1.46,
  "V16": -0.47, "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
  "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06, "V25": 0.12,
  "V26": -0.18, "V27": 0.13, "V28": -0.02, "Amount": 149.62
}'
```

You should instantly receive a JSON response back that looks like this:
```json
{
  "fraud_prediction": 0,
  "fraud_probability": 0.0001245,
  "risk_status": "LOW RISK"
}
```
