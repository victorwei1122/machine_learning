import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

def engineer_features(input_path, output_dir):
    """
    Performs feature engineering:
    - Categorical encoding (One-Hot for ownership, intent, grade, default)
    - Scaling numerical features
    - Saving processed features and the preprocessor for inference.
    """
    print(f"Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Separate features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Define preprocessing pipeline
    # Using OneHotEncoder for categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Fit and transform features
    X_processed = preprocessor.fit_transform(X)
    
    # Convert to DataFrame to keep track of column names (optional, but good for inspection)
    # OneHotEncoder generates new feature names
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_feature_names = numerical_cols + list(cat_feature_names)
    X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed data
    X_processed_df.to_csv(os.path.join(output_dir, 'features.csv'), index=False)
    y.to_csv(os.path.join(output_dir, 'target.csv'), index=False)
    
    # Save the preprocessor for production (MLOps step)
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
    
    print(f"Feature engineering complete. Saved to {output_dir}")
    return X_processed_df, y

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    cleaned_path = os.path.join(project_root, "data", "processed", "credit_risk_cleaned.csv")
    output_directory = os.path.join(project_root, "data", "processed")
    
    X_feat, y_feat = engineer_features(cleaned_path, output_directory)
    print("\nFeature Matrix Shape:", X_feat.shape)
    print("\nSample Features (First 5 rows):")
    print(X_feat.head())
