import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


RAW_DATA_PATH = "src/data/raw/Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "src/data/processed/final.csv"

def load_data(path=RAW_DATA_PATH):
    """Load raw CSV"""
    df = pd.read_csv(path)
    return df

def clean_data(df):
    """Clean data: handle duplicates, missing values, encode target"""

    df = df.drop_duplicates(subset="customerID")

    # Convert TotalCharges to numeric (some are spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

    # Fill missing TotalCharges with median (safe assignment)
    median_total = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total)

    # Encode target Churn column
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df

def remove_outliers(df):
    """Remove outliers for numeric columns"""
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    return df

def scale_features(df):
    """Scale numeric columns"""
    scaler = StandardScaler()
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def main():
    os.makedirs("src/data/processed", exist_ok=True)

    # Run pipeline
    df = load_data()
    df = clean_data(df)
    df = remove_outliers(df)
    df = scale_features(df)

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Cleaned dataset saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
