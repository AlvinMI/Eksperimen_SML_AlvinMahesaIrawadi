import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def run_automation():
    # Path dari root repository
    input_path = 'heart_disease_raw/heart.csv'
    output_dir = 'preprocessing/heart_disease_preprocessing'
    output_file = os.path.join(output_dir, 'heart_clean.csv')

    if not os.path.exists(input_path):
        print("File mentah gak ada!")
        return

    df = pd.read_csv(input_path)
    
    # Cleaning Logic
    df['Cholesterol'] = df['Cholesterol'].replace(0, df['Cholesterol'].median())
    df['RestingBP'] = df['RestingBP'].replace(0, df['RestingBP'].median())
    
    # Encoding
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Otomatisasi kelar! Data bersih ada di: {output_file}")

if __name__ == "__main__":
    run_automation()