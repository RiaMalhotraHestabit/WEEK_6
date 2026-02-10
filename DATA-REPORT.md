# DATA-REPORT.md — Day 1 (Telco Customer Churn)

## 1. Dataset Overview
- **Rows:** 7043  
- **Columns:** 21  
- **Numeric columns:** tenure, MonthlyCharges, TotalCharges, SeniorCitizen, Churn  
- **Categorical columns:** 16 columns including gender, Partner, Dependents, InternetService, etc.  
- **Missing values:** None after cleaning  

**Screenshot:**  

![Dataset Overview](screenshots/dataset_info.png)  

---

## 2. Target Variable Distribution (`Churn`)
- `0` = No churn  
- `1` = Churn  

**Observations:**  
- Majority of customers did not churn  
- Dataset is slightly imbalanced  

**Screenshot:**  

![Churn Distribution](screenshots/churn_distribution.png)  

---

## 3. Numeric Feature Distributions
- **Columns:** tenure, MonthlyCharges, TotalCharges  
- Observed distributions:  
  - tenure slightly skewed  
  - MonthlyCharges roughly uniform  
  - TotalCharges skewed due to customers with short tenure  

**Screenshots:** 

- Histogram of tenure:

 ![tenure](screenshots/tenure_hist.png)  

- Histogram of MonthlyCharges: 

![MonthlyCharges](screenshots/monthlycharges_hist.png)  

- Histogram of TotalCharges: 

![TotalCharges](screenhsots/totalcharges_hist.png)  


---

## 4. Categorical Feature Distributions
- Features like Contract, PaymentMethod, InternetService show variations in customer behavior  
- Most common contract: Month-to-month  
- Most common payment method: Electronic check  

**Screenshots:**  
- Contract distribution: ![Contract](screenshots/contract_count.png)  
- PaymentMethod distribution: ![PaymentMethod](screenshots/payment_count.png)  
- InternetService distribution: ![InternetService](screenshots/internet_count.png)  

---

## 5. Missing Values Heatmap
- No missing values remain in the cleaned dataset  

**Screenshot:**  
![Missing Values Heatmap](screenshots/missing_heatmap.png)  

---

## 6. Correlation Matrix (Numeric Features)
- Positive correlation observed between tenure and TotalCharges  
- MonthlyCharges weakly correlated with TotalCharges  
- Target variable Churn weakly correlated with numeric features  

**Screenshot:**  
![Correlation Heatmap](screenshots/correlation_heatmap.png)  

---

## 7. Summary
- Dataset is cleaned: duplicates removed, TotalCharges converted, missing values handled  
- Basic EDA performed: distributions, correlations, and missing value checks  
- Dataset ready for next steps (feature engineering, scaling, train/test split)
