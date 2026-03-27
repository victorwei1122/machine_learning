import pandas as pd
import numpy as np
import os

def clean_poverty_data(df):
    \"\"\"Perform initial cleaning and type conversion.\"\"\"
    df = df.copy()
    
    # 1. Map mixed-type columns (yes=1, no=0)
    mapping = {'yes': 1, 'no': 0}
    df['edjefe'] = df['edjefe'].replace(mapping).astype(float)
    df['edjefa'] = df['edjefa'].replace(mapping).astype(float)
    df['dependency'] = df['dependency'].replace(mapping).astype(float)
    
    # 2. Impute missing values based on domain logic
    # v2a1: monthly rent payment (if tipovivi1=1, they own the house, so rent is 0)
    df.loc[(df['tipovivi1'] == 1), 'v2a1'] = 0
    
    # v18q1: number of tablets (if v18q=0, they have 0 tablets)
    df.loc[(df['v18q'] == 0), 'v18q1'] = 0
    
    # rez_esc: Years behind in school (fill with 0 if missing)
    df['rez_esc'] = df['rez_esc'].fillna(0)
    
    # meaneduc: average years of education for adults (18+) (fill with 0 if missing)
    df['meaneduc'] = df['meaneduc'].fillna(0)
    df['SQBmeaned'] = df['SQBmeaned'].fillna(0)
    
    return df

def main():
    DATA_PATH = 'data/'
    PROCESSED_PATH = 'data/processed/'
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    for split in ['train', 'test']:
        file_path = os.path.join(DATA_PATH, f'{split}.csv')
        if os.path.exists(file_path):
            print(f\"Cleaning {split} data...\")
            df = pd.read_csv(file_path)
            cleaned_df = clean_poverty_data(df)
            
            output_path = os.path.join(PROCESSED_PATH, f'{split}_cleaned.csv')
            cleaned_df.to_csv(output_path, index=False)
            print(f\"Saved cleaned {split} data to {output_path}\")
        else:
            print(f\"Warning: {file_path} not found.\")

if __name__ == \"__main__\":
    main()
