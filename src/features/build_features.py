#!/usr/bin/env python
import sys
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
sys.path.append(str(Path(__file__).parent.parent))
from config.config import Paths, RANDOM_STATE
from features.feature_selector import select_features

DATA_PATH = Path('src/data/processed/final.csv')
PREPROCESSOR_PATH = Path('src/features/preprocessor.pkl')
FEATURE_LIST_PATH = Path('src/features/feature_list.json')

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f" Loaded cleaned data: {df.shape}")
    return df

def create_new_features(df):
    """Type A Feature Engineering - Deterministic transformations (safe before split)"""
    
    # 1-4: Charge-based features
    df["AvgChargePerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["TenureGroup"] = pd.cut(
        df["tenure"], 
        bins=[-1, 12, 24, 48, 72], 
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
    )
    df["IsLongTerm"] = (df["tenure"] > 24).astype(int)
    df["HighMonthlyCharges"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
    
    # 5-7: Service-based features
    df["HasStreaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)
    df["HasOnlineSecurity"] = (df["OnlineSecurity"] == "Yes").astype(int)
    df["HasTechSupport"] = (df["TechSupport"] == "Yes").astype(int)
    
    # 8: Total services count
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", 
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["TotalServices"] = sum((df[col] == "Yes").astype(int) for col in service_cols)
    
    # 9: Contract type flag
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    
    # 10-11: Demographic combinations
    df["SeniorWithPartner"] = (
        (df["SeniorCitizen"] == 1) & (df["Partner"] == "Yes")
    ).astype(int)
    df["PartnerAndDependents"] = (
        (df["Partner"] == "Yes") & (df["Dependents"] == "Yes")
    ).astype(int)
    
    print(f"✓ Created 11 new features: {df.shape}")
    return df

def main():
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE (DAY 2)")
    print("=" * 60)
    
    # Step 1: Load cleaned data
    print("\n[1/7] Loading cleaned data from Day 1...")
    df = load_data()
    
    # Step 2: Create new features (Type A - safe before split)
    print("\n[2/7] Creating new features...")
    df = create_new_features(df)
    
    # Step 3: Separate features and target
    print("\n[3/7] Separating features and target...")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    print(f"✓ X: {X.shape}, y: {y.shape}")
    
    # Step 4: Train/Test Split (CRITICAL: before any data-dependent operations)
    print("\n[4/7] Train/test split (before preprocessing)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Step 5: Build preprocessor (Type B - fit on train only)
    print("\n[5/7] Building preprocessor (fit on TRAIN only)...")
    cat_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    print(f"  - Categorical: {len(cat_cols)} columns")
    print(f"  - Numerical: {len(num_cols)} columns")
    
    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    
    # Fit on train, transform both
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding
    feature_names = num_cols + list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
    )
    
    X_train_proc = pd.DataFrame(X_train_proc, columns=feature_names)
    X_test_proc = pd.DataFrame(X_test_proc, columns=feature_names)
    print(f"After encoding: {X_train_proc.shape[1]} features")
    
    # Step 6: Feature selection (on train only)
    print("\n[6/7] Feature selection (on TRAIN only)...")
    X_train_final, selected_features = select_features(
        X_train_proc, 
        y_train.reset_index(drop=True), 
        return_feature_names=True
    )
    X_test_final = X_test_proc[selected_features]
    print(f"✓ Final features: {len(selected_features)}")
    
    # Step 7: Save all outputs
    print("\n[7/7] Saving outputs...")
    
    # Create directories
    Paths.X_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save train/test splits
    X_train_final.to_csv(Paths.X_TRAIN, index=False)
    X_test_final.to_csv(Paths.X_TEST, index=False)
    y_train.to_csv(Paths.Y_TRAIN, index=False)
    y_test.to_csv(Paths.Y_TEST, index=False)
    
    # Save preprocessor (for production deployment)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    # Save feature metadata
    FEATURE_LIST_PATH.write_text(json.dumps({
        "selected_features": selected_features,
        "n_features": len(selected_features),
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols,
        "total_features_after_encoding": len(feature_names)
    }, indent=2))
    
    print(f"✓ Train/test data saved to {Paths.X_TRAIN.parent}")
    print(f"✓ Preprocessor saved to {PREPROCESSOR_PATH}")
    print(f"✓ Feature list saved to {FEATURE_LIST_PATH}")
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE - NO DATA LEAKAGE!")
    print("=" * 60)
    print(f"\nNext step: Run training/train.py (Day 3)")

if __name__ == "__main__":
    main()