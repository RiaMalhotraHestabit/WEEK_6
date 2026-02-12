# Feature Engineering – Telecom Customer Churn

## Objective

Build a feature engineering pipeline that:

- Encodes categorical features  
- Normalizes numerical features  
- Generates 10+ new features  
- Applies feature selection  
- Produces:
  - `X_train`, `X_test`
  - `y_train`, `y_test`
- Saves selected features  
- Plots feature importance  

---

## 1. Data Preprocessing

### Numerical Handling
- Converted `TotalCharges` to numeric  
- Handled missing values  
- Applied **StandardScaler** for normalization  

### Categorical Handling
- Applied **OneHotEncoder(handle_unknown="ignore")**
- Ensures safe handling of unseen categories  

---

## 2. Manual Feature Engineering (11 Features)

The following features were created:

1. `AvgChargePerMonth`
2. `TenureGroup`
3. `IsLongTerm`
4. `HighMonthlyCharges`
5. `HasStreaming`
6. `HasOnlineSecurity`
7. `HasTechSupport`
8. `TotalServices`
9. `IsMonthToMonth`
10. `SeniorWithPartner`
11. `PartnerAndDependents`


---

## 3. Encoding Expansion

Categorical columns were transformed using One-Hot Encoding.


This generated ~30–50 binary features depending on unique category values.

---

## 4. Feature Selection Strategy

Two-stage feature selection:

### Step 1: Correlation Filtering
- Removed highly correlated features  

### Step 2: Mutual Information (MI)
- Ranked features based on predictive power  
- Selected top informative features  

### Generated:
- `feature_list.json`
- `feature_importance.png`

---

## 5. Train-Test Split

- Applied **Stratified Split**
- Maintained churn class balance  

Generated files:
- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

---




