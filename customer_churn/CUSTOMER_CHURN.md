# 📉 Customer Churn Prediction

## 📖 Overview
In banking and telecommunications, acquiring a new customer is significantly more expensive than retaining an existing one. This project demonstrates an MLOps pipeline for Customer Churn Prediction, allowing a business to proactively identify at-risk customers and offer targeted retention campaigns.

## 🎯 Business Use Case
- **Problem**: Customers leaving the bank/service due to high friction, better competitor offers, or low engagement.
- **Solution**: A binary classification model trained on behavioral, demographic, and account data to predict churn probability.
- **Value**: Enables marketing and customer success teams to run highly targeted retention campaigns, lowering churn rates and increasing CLV (Customer Lifetime Value).

---

## 🧠 Why XGBoost? (Model Selection & Intuition)

### How XGBoost Works
XGBoost (eXtreme Gradient Boosting) is an advanced ensemble technique. It iteratively builds decision trees, where later trees focus entirely on correcting the mistakes made by earlier trees. It uses regularization math to ensure it doesn't overfit to the training data.

### Why It Excels in Customer Churn
Customer behavior is highly complex and non-linear. A customer rarely churns for one simple reason. Churn happens through complex interactions known as "Feature Crosses". For example:
- *Condition A*: The customer has Month-to-Month billing.
- *Condition B*: The customer has high monthly charges.
- *Condition C*: The customer has poor tech support.

A standard linear model (like Logistic Regression) struggles to link these conditions unless you manually engineer complex interaction terms. XGBoost's deep, hierarchical trees naturally discover these hidden `A AND B AND C` behavioral conditions automatically. Because the dataset contains a mix of continuous financial data and discrete categorical groupings, XGBoost provides best-in-class predictive accuracy. 

---

## pipeline ⚙️ MLOps Pipeline Steps

### 1. Data Cleaning (`data_cleaning.py`)
- **Data Typing**: The raw extraction often casts monetary values (like `TotalCharges`) as text if a customer is brand new (0 tenure) and hasn't collected a bill yet. We coerce these edge cases to `NaN` and fill them accurately with `0`.
- **Anonymization**: We strip out `customerID` identifiers to prevent the model from accidentally memorizing individual users (preventing data leakage and overfitting).

### 2. Feature Engineering (`feature_engineering.py`)
Because our customer data is highly heterogeneous, we use `scikit-learn`'s **ColumnTransformer**:
- **Numerical Scaling**: We isolate the quantitative variables (tenure length, monthly charges) and apply `StandardScaler`. We explicitly chose `StandardScaler` (which uses Mean and Standard Deviation) because this data is relatively bounded without insane outliers (e.g., `tenure` is capped at 72 months, and `MonthlyCharges` naturally peak around $120). By standardizing, we ensure the algorithm doesn't artificially bias toward metrics just because their raw numbers are larger.
- **Categorical Encoding**: We take descriptive variables (Contract Type, Payment Method, Multi-Lines) and run them through `OneHotEncoder`. This converts string labels into clean `[0,1]` binary columns safely.
- The fitted architecture is saved to `models/preprocessor.joblib`. 

### 3. Model Training (`train.py`)
- Using **MLflow**, we track hyperparameters and log validation performance metrics.
- We evaluate the model using a myriad of metrics (*Accuracy, Precision, Recall, F1, ROC-AUC*). In churn contexts, tracking the **ROC-AUC** is highly valuable because we want a model that generates excellent probablistic ranked scores (0.0 to 1.0) rather than a rigid Yes/No barrier. 
- Features importances are automatically calculated and reviewed.

### 4. Nightly Batch Prediction (`batch_prediction.py`)
- Unlike Fraud Detection (which requires instantaneous decisions), retention marketing executes campaigns on delayed timelines. 
- We built a **Cron-style Batch Script**. This script loads the active model and preprocessor, reads in the entire banking customer base from the data lake, scores them, and outputs a `batch_predictions_output.csv`. Marketing teams can simply filter this output for "High Risk" segments (>60% churn probability) to trigger email workflows the next morning! 

---

## 📊 Evaluation Metrics Explained
When you run the training pipeline or the notebook, you will see a Classification Report. Here is how to interpret those metrics through a business lens for the **Churn Class (1)**:

- **ROC-AUC (e.g., 0.84)**: There is an 84% chance the model will score a randomly chosen actual churner higher than a randomly chosen retained customer. Excellent for ranking marketing target lists.
- **Precision (e.g., 0.65)**: When the model flags a customer as "At Risk", it is correct ~65% of the time. The remaining 35% are false alarms, which is highly acceptable since accidentally offering a retention discount to a loyal customer is rarely a net-negative.
- **Recall (e.g., 0.52)**: Out of all the customers who truly churned in reality, the model successfully caught ~52% of them. If the business needed to stop all bleeding at all costs, the prediction threshold could be lowered to increase this metric.
- **F1-Score (e.g., 0.58)**: The harmonic mean of Precision and Recall, used as a balanced single-number barometer of model health.
- **Support**: The raw count of instances in the test set.

---

## 💼 Executive Presentation & Business Buy-In
If presenting this pipeline to marketing executives or business leaders, focus on the **ROI of Retention** rather than model hyperparameters:

1. **The Problem with Blanket Marketing**: Offering a generic $50 retention discount to all 100,000 customers costs $5,000,000. It is a massive waste of budget to discount customers who were never going to leave in the first place.
2. **The Power of Precision Targeting**: This pipeline acts as a surgical tool. Instead of blanket marketing, the model precisely identifies the 3,000 specific customers who are currently exhibiting flight-risk behavior.
3. **The Financial Impact**: By targeting only the At-Risk segment, Marketing can reduce their promotional spend by over 90%, while directly saving millions in future recurring revenue by intercepting the exact people who were on the fence. 
4. **Actionability**: End the presentation by showing the `batch_predictions_output.csv` file. Explain to marketing that they don't need to learn Python; they just need to load this spreadsheet into their email software (like HubSpot or Salesforce), and the proactive retention workflow is completely ready for production.

---

## 🛠 Project Structure
- `data/`: Contains the Telco Customer Churn dataset.
- `src/`: Modular feature engineering and training pipelines.
- `notebooks/`: Exploratory Data Analysis and interactively structured training logic.

## 🚀 Running the Pipeline
1. Fetch data: `python src/fetch_data.py`
2. Run notebooks: Try interactively executing `notebooks/01_customer_churn_mlops.ipynb`
3. Train model: `python src/train.py`
4. Run batch scoring to simulate a marketing pipeline: `python src/batch_prediction.py`
