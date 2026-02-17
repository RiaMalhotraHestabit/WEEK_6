#!/usr/bin/env python

import os
import pandas as pd

RAW_DATA_PATH = "src/data/raw/Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "src/data/processed/final.csv"

def load_data(path=RAW_DATA_PATH):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Raw data shape: {df.shape}")
    return df

def clean_data(df):
    print("\nCleaning data...")
    
    # 1. Drop duplicates
    initial_rows = len(df)
    df = df.drop_duplicates(subset="customerID")
    duplicates_removed = initial_rows - len(df)
    print(f"   Duplicates removed: {duplicates_removed}")
    
    # 2. Drop identifier column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print(f"  Dropped customerID column")
    
    # 3. Convert TotalCharges to numeric (spaces become NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    total_charges_nulls = df["TotalCharges"].isna().sum()
    print(f"  TotalCharges converted to numeric ({total_charges_nulls} NaN values found)")
    
    # 4. Fill missing TotalCharges (new customers with 0 tenure) with 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    print(f"  Missing TotalCharges filled with 0")
    
    # 5. Encode target column (Yes/No → 1/0)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    print(f"  Target 'Churn' encoded (Yes→1, No→0)")
    print(f"  Class distribution:\n{df['Churn'].value_counts()}")
    
    return df

def main():
    # Ensure processed folder exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    
    df = load_data()
    df = clean_data(df)
    
    # Save cleaned dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print(f"\n Cleaned dataset saved to {PROCESSED_DATA_PATH}")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    main()