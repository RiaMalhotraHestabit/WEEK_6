#!/usr/bin/env python
"""
Data Cleaning Pipeline
Handles: duplicates, missing values, type conversions, target encoding
Does NOT handle: feature encoding or scaling (that's in build_features.py)
"""
import os
import pandas as pd

# Paths
RAW_DATA_PATH = "src/data/raw/Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "src/data/processed/final.csv"

def load_data(path=RAW_DATA_PATH):
    """Load raw CSV"""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Raw data shape: {df.shape}")
    return df

def clean_data(df):
    """Clean data: remove duplicates, handle missing values, encode target"""
    print("\nCleaning data...")
    
    # 1️⃣ Drop duplicates
    initial_rows = len(df)
    df = df.drop_duplicates(subset="customerID")
    duplicates_removed = initial_rows - len(df)
    print(f"   Duplicates removed: {duplicates_removed}")
    
    # 2️⃣ Drop identifier column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print(f"  Dropped customerID column")
    
    # 3️⃣ Convert TotalCharges to numeric (spaces become NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    total_charges_nulls = df["TotalCharges"].isna().sum()
    print(f"  TotalCharges converted to numeric ({total_charges_nulls} NaN values found)")
    
    # 4️⃣ Fill missing TotalCharges (new customers with 0 tenure) with 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    print(f"  Missing TotalCharges filled with 0")
    
    # 5️⃣ Encode target column (Yes/No → 1/0)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    print(f"  Target 'Churn' encoded (Yes→1, No→0)")
    print(f"  Class distribution:\n{df['Churn'].value_counts()}")
    
    return df

def main():
    # Ensure processed folder exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    
    # Load and clean (NO SCALING - that happens in build_features.py)
    df = load_data()
    df = clean_data(df)
    
    # Save cleaned dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print(f"\n Cleaned dataset saved to {PROCESSED_DATA_PATH}")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\n Next step: Run features/build_features.py to encode and engineer features")

if __name__ == "__main__":
    main()