#!/usr/bin/env python
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from feature_selector import select_features

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "final.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
FEATURE_LIST_PATH = os.path.join(BASE_DIR, "features", "feature_list.json")

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    
    return df

def create_new_features(df):
    # 1. Average charge per month
    df["AvgChargePerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)  
    # 2. Tenure group
    df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1,12,24,48,72], labels=["0-1yr","1-2yr","2-4yr","4-6yr"])
    # 3. Long-term customer
    df["IsLongTerm"] = (df["tenure"] > 24).astype(int)

    # 4. High monthly charges
    df["HighMonthlyCharges"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
    
    # 5. Streaming services
    df["HasStreaming"] = ((df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")).astype(int)
    
    # 6 Online security
    df["HasOnlineSecurity"] = (df["OnlineSecurity"] == "Yes").astype(int)
    
    # 7. Tech support
    df["HasTechSupport"] = (df["TechSupport"] == "Yes").astype(int)
    
    # 8. Total services subscribed
    df["TotalServices"] = (
        (df["PhoneService"] == "Yes").astype(int) +
        (df["MultipleLines"] == "Yes").astype(int) +
        (df["OnlineSecurity"] == "Yes").astype(int) +
        (df["OnlineBackup"] == "Yes").astype(int) +
        (df["DeviceProtection"] == "Yes").astype(int) +
        (df["TechSupport"] == "Yes").astype(int) +
        (df["StreamingTV"] == "Yes").astype(int) +
        (df["StreamingMovies"] == "Yes").astype(int)
    )
    
    # 9. Month-to-month contract flag
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    
    # 10. Senior citizen with partner
    df["SeniorWithPartner"] = ((df["SeniorCitizen"]==1) & (df["Partner"]=="Yes")).astype(int)
    
    # 11. Partner & Dependents combo
    df["PartnerAndDependents"] = ((df["Partner"]=="Yes") & (df["Dependents"]=="Yes")).astype(int)
    
    # Handle missing values properly after feature creation

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    return df

def build_pipeline(df):
    target = "Churn"
    X = df.drop(columns=[target])
    y = df[target]

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object", "category","string"]).columns
    numeric_cols = X.select_dtypes(include=["int64","float64"]).columns

    # Column transformer for scaling and encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    # Feature names
    feature_names = list(numeric_cols) + \
                    list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))

    X_processed = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
        columns=feature_names
    )

    return X_processed, y, feature_names

def main():
    print("Loading data...")
    df = load_data()

    print("Creating new features...")
    df = create_new_features(df)

    print("Building preprocessing pipeline...")
    X, y, feature_names = build_pipeline(df)

    print("Selecting best features...")
    X_selected = select_features(X, y)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Saving datasets...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

    # Save selected features
    os.makedirs(os.path.dirname(FEATURE_LIST_PATH), exist_ok=True)
    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(list(X_selected.columns), f, indent=4)

    print("Feature engineering pipeline completed successfully.")

if __name__ == "__main__":
    main()
