#!/usr/bin/env python
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Paths
RAW_DATA_PATH = "src/data/raw/Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "src/data/processed/final.csv"

def load_data(path=RAW_DATA_PATH):
    """Load raw CSV"""
    df = pd.read_csv(path)
    return df

def clean_data(df):
    """Clean data: remove duplicates, handle missing values, encode target"""
    # 1️⃣ Drop duplicates
    df = df.drop_duplicates(subset="customerID")
    
    # 2️⃣ Drop identifier column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    
    # 3️⃣ Convert TotalCharges to numeric (spaces become NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    
    # 4️⃣ Fill missing TotalCharges (new customers) with 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    
    # 5️⃣ Encode target column
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    
    return df

def scale_features(df):
    """Scale numeric columns: MonthlyCharges, TotalCharges"""
    scaler = StandardScaler()
    numeric_cols = ["MonthlyCharges", "TotalCharges"]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def main():
    # Ensure processed folder exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    
    # Load, clean, and scale
    df = load_data()
    df = clean_data(df)
    df = scale_features(df)
    
    # Save final cleaned dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Cleaned dataset saved to {PROCESSED_DATA_PATH} with shape {df.shape}")

if __name__ == "__main__":
    main()
