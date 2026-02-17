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


## Pipeline Overview
```
Raw Data (7,043 samples)
    ↓
Create 11 Engineered Features
    ↓
Train/Test Split (80/20) ← CRITICAL: Before encoding!
    ↓
OneHot Encode Categoricals (fit on train only)
    ↓
Feature Selection (train only)
    ↓
Final: 30 selected features
```
---


## 1. Data Source
- **Dataset**: Telco Customer Churn (from Day 1 cleaned data)
- **Samples**: 7,043 customers
- **Target**: Churn (binary: 0=No, 1=Yes)
- **Churn rate**: 26.54% (imbalanced)

---

## 2. Feature Engineering (11 New Features)

### Charge-Based (4)
1. **AvgChargePerMonth**: `TotalCharges / (tenure + 1)`
2. **TenureGroup**: Binned tenure into 4 categories (0-1yr, 1-2yr, 2-4yr, 4-6yr)
3. **IsLongTerm**: Binary flag for `tenure > 24 months`
4. **HighMonthlyCharges**: Binary flag for above-median charges

### Service-Based (4)
5. **HasStreaming**: Combined StreamingTV OR StreamingMovies
6. **HasOnlineSecurity**: Binary flag
7. **HasTechSupport**: Binary flag  
8. **TotalServices**: Count of subscribed services (0-8)

### Contract/Demographic (3)
9. **IsMonthToMonth**: Binary flag for month-to-month contracts
10. **SeniorWithPartner**: Senior citizen AND has partner
11. **PartnerAndDependents**: Has both partner AND dependents

---

## 3. Preprocessing Strategy

### Train/Test Split (FIRST!)
- **Method**: Stratified split (preserves 26.54% churn ratio)
- **Split**: 80% train (5,634) / 20% test (1,409)
- **Critical**: Split BEFORE encoding to prevent data leakage

### Encoding (After Split)
- **Categorical**: OneHotEncoder (handle_unknown='ignore')
  - Fitted on train only
  - Transformed both train and test
- **Numerical**: Passthrough (no scaling applied)
- **Result**: 30 base features → 59 encoded features

---

## 4. Feature Selection

### Two-Stage Selection:

**Stage 1: Correlation Filtering**
- Removed features with correlation > 0.9
- Dropped: 16 highly correlated features
- Remaining: 43 features

**Stage 2: Mutual Information**
- Ranked features by MI score with target
- Selected top 30 features
- Saved to `feature_list.json` 

## 5. Output Files

### Data Files
- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

### Artifacts
- `preprocessor.pkl`: Fitted OneHotEncoder for production use
- `feature_list.json`: Selected feature names and metadata
- `feature_selection_report.txt`: Detailed selection process
- `feature_importance.png`: Top 20 features by MI score

---

## 6. Key Statistics

| Metric | Value |
|--------|-------|
| Original features | 30 |
| After encoding | 59 |
| After correlation filter | 43 |
| **Final selected** | **30** |
| Engineered features kept | 5 out of 11 (45%) |

---

## 7. Business-Relevant Features Created

5 of 11 engineered features were selected:
- **IsLongTerm**: Customer retention indicator
- **HighMonthlyCharges**: Revenue risk flag
- **HasOnlineSecurity**: Service usage pattern
- **TotalServices**: Customer engagement metric
- **PartnerAndDependents**: Demographic stability

These features proved more predictive than some raw features.

---



