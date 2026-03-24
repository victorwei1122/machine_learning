# 🚀 Machine Learning & MLOps Portfolio

This repository serves as a centralized hub for diverse personal machine learning projects, demonstrating end-to-end MLOps (Machine Learning Operations) capabilities. Each project is designed to solve specific business or technical use cases while adhering to professional engineering standards.

## 🛠 MLOps Lifecycle & Best Practices

Each use case in this repository follows a structured MLOps pipeline to ensure scalability, reproducibility, and reliability.

| Phase | Description | Key Techniques/Tools |
| :--- | :--- | :--- |
| **1. Data Cleaning** | Handling missing values, outliers, and data quality issues. | Pandas, SQL, Data Validation |
| **2. Feature Engineering** | Selecting and transforming variables to improve model performance. | Scaling, Encoding, PCA, Feature Selection |
| **3. Training & Registry** | Controlled experimentation and model versioning. | Scikit-learn, MLflow, Experiment Tracking |
| **4. Deployment** | Serving models for real-time or batch inferencing. | Docker, FastAPI, Model Serving |
| **5. Batch Prediction** | Running efficient inference on large datasets. | Cron Jobs, Spark, Batch Scripts |
| **6. Model Evaluation** | Continuous monitoring of drift and performance metrics. | Accuracy/F1, MSE/MAE, Monitoring Dashboards |

## 📂 Repository Structure

```text
machine_learning/
├── credit_risk/           # Credit default prediction use case
│   ├── data/              # Raw and processed datasets
│   ├── notebooks/         # Exploratory Data Analysis (EDA)
│   ├── src/               # Reusable source code (cleaning, training, etc.)
│   ├── models/            # Serialized model artifacts
│   └── README.md          # Use-case specific documentation
├── templates/             # Starter templates for new ML projects
└── README.md              # Main repository overview (you are here)
```

## 🚀 Projects Overview

| Project | Domain | Status | Key Tech |
| :--- | :--- | :--- | :--- |
| **[Credit Risk](./credit_risk)** | Finance | 🏗 In Progress | Scikit-Learn |

## 🛠 Tech Stack

- **Core**: Python (Pandas, NumPy, Scikit-learn)
- **Tracking**: MLflow (Proposed)
- **Serving**: FastAPI / Docker (Proposed)
- **DevOps**: GitHub Actions for CI/CD

## 🏁 Getting Started

1. **Clone the repository**:

   ```bash
   git clone [your-repo-url]
   cd machine_learning
   ```

2. **Environment Setup**:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---
*Created by Victor Wei - Focused on building production-ready ML systems.*
