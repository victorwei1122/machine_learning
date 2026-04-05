import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

def engineer_features(input_path: str, output_path: str, preprocessor_path: str):
    print("Loading cleaned data for feature engineering...")
    df = pd.read_csv(input_path)
    
    # Target encoding
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    X = df.drop('Churn', axis=1)
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    # Fit transform
    X_processed = preprocessor.fit_transform(X)
    
    # Retrieve feature names
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_feature_names)
    
    # Create final dataframe
    df_engineered = pd.DataFrame(X_processed, columns=feature_names)
    df_engineered['Churn'] = df['Churn'].values
    
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_engineered.to_csv(output_path, index=False)
    
    print(f"Feature engineering complete. Preprocessor saved to {preprocessor_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    engineer_features(
        os.path.join(base_dir, 'data', 'customer_churn_cleaned.csv'),
        os.path.join(base_dir, 'data', 'customer_churn_engineered.csv'),
        os.path.join(base_dir, 'models', 'preprocessor.joblib')
    )
