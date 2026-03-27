# 🚀 Machine Learning & MLOps Portfolio

This repository serves as a centralized hub for diverse personal machine learning projects, demonstrating end-to-end MLOps (Machine Learning Operations) capabilities. Each project is designed to solve specific business or technical use cases while adhering to professional engineering standards.

## 🛠 MLOps Lifecycle & Best Practices

Each use case in this repository follows a structured MLOps pipeline to ensure scalability, reproducibility, and reliability.

| Phase | Description | Key Techniques/Tools |
| :--- | :--- | :--- |
| **1. Data Cleaning** | Handling missing values, outliers, and data quality issues. | Pandas, SQL, Data Validation |
| **2. Feature Engineering** | Selecting and transforming variables to improve model performance. | Scaling, Encoding, PCA, Feature Selection |
| **3. Training & Registry** | Controlled experimentation and model versioning. | Scikit-learn, LightGBM, MLflow |
| **4. Deployment** | Serving models for real-time or batch inferencing. | Docker, FastAPI, Model Serving |
| **5. Batch Prediction** | Running efficient inference on large datasets. | Cron Jobs, Spark, Batch Scripts |
| **6. Model Evaluation** | Continuous monitoring of drift and performance metrics. | Accuracy/F1 (Macro), Precision/Recall |

## 📂 Repository Structure

```text
machine_learning/
├── credit_risk/           # Credit default prediction (Supervised)
│   ├── data/              # Raw and processed datasets
│   ├── notebooks/         # Interactive MLOps Step-by-Step
│   ├── src/               # Modular components (cleaning, training, etc.)
│   └── README.md          # Use-case documentation
├── segmentation/          # Household Poverty & Clustering
│   └── costa_rican_poverty/ # Hybrid Clustering + Classification
│       ├── notebooks/      # Visual EDA & Self-Contained logic
│       ├── src/            # Production-ready MLOps code
│       └── POVERTY_PREDICTION.md # Data Dictionary & Insights
├── templates/             # Starter templates for new ML projects
└── README.md              # Main repository overview (you are here)
```

## 🚀 Projects Overview

| Project | Domain | Status | Key Tech |
| :--- | :--- | :--- | :--- |
| **[Credit Risk](./credit_risk)** | Finance | ✅ Completed | Scikit-Learn, MLflow, FastAPI |
| **[Poverty Prediction](./segmentation/costa_rican_poverty)** | Social Science | ✅ Completed | LightGBM, K-Means, Docker |

## 🛠 Tech Stack

- **Core**: Python (Pandas, NumPy, Scikit-learn, LightGBM)
- **Tracking**: MLflow (Experiment tracking and model registry)
- **Serving**: FastAPI / Docker (Containerized real-time inference)
- **Clustering**: K-Means for natural socio-economic segmentation
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
   pip install -r segmentation/costa_rican_poverty/requirements.txt
   ```

---
*Created by Victor Wei - Focused on building production-ready ML systems.*
