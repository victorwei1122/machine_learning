# Costa Rican Household Poverty Prediction

This project explores high-dimensional household data to segment and predict poverty levels in Costa Rica. With 142 features per household, it offers a rich playground for feature engineering, dimensionality reduction, and clustering.

## 📊 Project Overview

The goal is to classify households into four levels of income/well-being:

1. **Extreme Poverty**
2. **Moderate Poverty**
3. **Vulnerable Households**
4. **Non-vulnerable Households**

### Key Challenges

* **Household vs. Individual:** Data is provided per individual, but the target is consistent within a household. Feature aggregation is critical.
* **Feature Volume:** 142 columns including housing materials, education, and household composition.
* **Mixed Labels:** Some columns like `dependency` and `edjefe` mix numeric values with 'yes'/'no' strings.

## 📖 Data Dictionary (Non-English Features)

To make sense of the 142 features, here is a guide to the common Spanish-based column names:

| Prefix | Spanish | English Meaning |
| :--- | :--- | :--- |
| **`pared`** | Pared | Wall material (e.g., `paredblig` = brick) |
| **`piso`** | Piso | Floor material (e.g., `pisocemento` = cement) |
| **`techo`** | Techo | Roof material (e.g., `techozinc` = zinc) |
| **`abasta`** | Abastecimiento | Water supply (e.g., `abastaguadentro` = water inside) |
| **`sanitario`** | Sanitario | Toilet/Sanitary conditions |
| **`energcocinar`** | Energía Cocinar | Cooking energy source (Electricity, Gas, Wood) |
| **`elimbasu`** | Eliminación Basura | Waste disposal method |
| **`epared/etecho/eviv`** | Estado | Condition of Wall/Roof/House (1=Bad, 2=Regular, 3=Good) |
| **`instlevel`** | Instrucción | Level of education attained |
| **`tipovivi`** | Tipo Vivienda | Housing status (Own, Rent, Precarious, etc.) |
| **`lugar`** | Lugar | Region (1-6) |
| **`area`** | Área | Area type (1=Urban, 2=Rural) |

**Common Odd Columns:**

* **`v2a1`**: Monthly rent payment.
* **`v18q1`**: Number of tablets owned.
* **`hacdor`**: Overcrowding (more than 3 persons per bedroom).
* **`escolari`**: Years of schooling.
* **`rez_esc`**: Years behind in school.

## 🧠 Algorithm Insights

This project employs a **Hybrid Learning Architecture** combining Unsupervised and Supervised techniques:

### 1. Unsupervised Segmentation (K-Means)

Why? In poverty prediction, the official "Target" 1-4 is often a noisy, human-assigned label. By using **K-Means Clustering**, we find "natural" groups based solely on infrastructure (walls, floors, water) and education.

* **Insight**: We discovered that Cluster 3 almost perfectly aligns with the non-vulnerable (Target 4) population, while Cluster 2 captures a specific "Urban/High Rent" vulnerable group.

### 2. Supervised Classification (LightGBM)

Why? With 142 features, traditional Decision Trees would overfit. **LightGBM** (Light Gradient Boosting Machine) is chosen for:

* **Histogram-based learning**: Faster training and lower memory usage.
* **Native handling of imbalanced data**: Since vulnerable households are rarer than wealthy ones, LightGBM's gradient-based approach focuses on these hard-to-predict cases.

## 💡 Key Findings

* **Education is the strongest Predictor**: Average household schooling (`meaneduc`) has a direct correlation with poverty level. Every extra year of adult schooling significantly reduces the probability of being in Target 1 or 2.
* **The "Tablet Gap"**: Tablets (`v18q1`) are a more precise luxury indicator than mobile phones. While almost every household has a phone, only the most stable (Target 4) own tablets.
* **Overcrowding Dynamics**: The `hacdor` feature (more than 3 people per bedroom) is a better predictor of moderate poverty than total household size alone. It indicates a lack of living space relative to the family's needs.

## 📁 Project Structure

Following the modular MLOps pattern:

* `data/`: Raw and processed CSV files.
* `notebooks/`: Exploratory Data Analysis (EDA) and Model Prototypes.
* `src/`: Production-ready modular code.
  * `data_cleaning.py`: Initial scrubbing and label fixing.
  * `feature_engineering.py`: Household-level aggregation and encoding.
  * `train.py`: Model training and MLflow tracking.
  * `app.py`: FastAPI service for real-time segmented predictions.
* `requirements.txt`: Project dependencies.
* `Dockerfile`: Containerization setup.

## 🚀 Getting Started

1. **Environment Setup**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**:
   Place `train.csv` and `test.csv` in the `data/` directory.

3. **Cleaning & Engineering**:
   Run the preprocessing pipeline to generate the household-level feature set.

## 📡 Live Prediction Example

Once the FastAPI app is running (via Docker or `uvicorn`), you can send a **JSON** request to the `/predict` endpoint to get an live poverty classification and cluster assignment.

### Example Request (using `curl`)

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "idhogar": "h_12345",
           "meaneduc": 12.5,
           "v2a1": 250000,
           "hacdor": 0,
           "rooms": 4,
           "tamhog": 3,
           "v18q1": 1
         }'
```

### Example Response

```json
{
  "household_id": "h_12345",
  "poverty_prediction": 4,
  "poverty_label": "Non-vulnerable",
  "cluster_assignment": 3,
  "confidence_score": 0.92
}
```

*Note: The API handles the mapping of individual data to the household-level model automatically.*
