import pandas as pd
import numpy as np
import os

def aggregate_household_features(df):
    \"\"\"Aggregate individual-level features to the household level.\"\"\"
    ind_features = ['escolari', 'age', 'rez_esc', 'dis', 'male', 'female']
    
    agg_funcs = {
        'escolari': ['mean', 'max', 'min', 'std', 'sum'],
        'age': ['mean', 'max', 'min', 'std', 'sum'],
        'rez_esc': ['mean', 'max', 'sum'],
        'dis': ['sum', 'mean'], 
        'male': ['sum', 'mean'],
        'female': ['sum', 'mean']
    }
    
    house_agg = df.groupby('idhogar')[ind_features].agg(agg_funcs)
    house_agg.columns = ['_'.join(col).strip() for col in house_agg.columns.values]
    house_agg.reset_index(inplace=True)
    
    # We take the head of household's data as the base and merge with aggregates
    heads = df[df['parentesco1'] == 1].copy()
    final_df = heads.merge(house_agg, on='idhogar', how='left')
    
    return final_df

def create_derived_features(df):
    \"\"\"Create new socio-economic indicators.\"\"\"
    df['phones_per_person'] = df['qmobilephone'] / df['tamhog']
    df['tablets_per_person'] = df['v18q1'] / df['tamhog']
    df['rooms_per_person'] = df['rooms'] / df['tamhog']
    df['rent_per_person'] = df['v2a1'] / df['tamhog']
    df['children_ratio'] = df['hogar_nin'] / df['hogar_total']
    df['elderly_ratio'] = df['hogar_mayor'] / df['hogar_total']
    df['adult_ratio'] = df['hogar_adul'] / df['hogar_total']
    
    return df

def main():
    PROCESSED_PATH = 'data/processed/'
    
    for split in ['train', 'test']:
        input_file = os.path.join(PROCESSED_PATH, f'{split}_cleaned.csv')
        if os.path.exists(input_file):
            print(f\"Engineering {split} features...\")
            df = pd.read_csv(input_file)
            df_eng = aggregate_household_features(df)
            df_eng = create_derived_features(df_eng)
            
            output_file = os.path.join(PROCESSED_PATH, f'{split}_engineered.csv')
            df_eng.to_csv(output_file, index=False)
            print(f\"Saved engineered {split} data to {output_file}\")
        else:
            print(f\"Warning: {input_file} not found. Run data_cleaning.py first.\")

if __name__ == \"__main__\":
    main()
