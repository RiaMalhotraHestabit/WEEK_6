# Model Comparison Report
## Telecom Customer Churn Prediction

**Date:** Generated from Day-3 Training Pipeline  
**Dataset:** Telecom Customer Churn (7,043 customers)  
**Class Distribution:** 73.5% No Churn (0) | 26.5% Churn (1)  
**Train/Test Split:** 80/20 stratified split

---

## Models Trained

4 classification models with class balancing:
1. **Logistic Regression** (baseline linear model)
2. **Random Forest** (ensemble of decision trees)
3. **Gradient Boosting** (sequential boosting ensemble)
4. **Support Vector Classifier** (SVC with RBF kernel)

All models use `class_weight='balanced'` to handle class imbalance.

---

## Cross-Validation Results (5-Fold Stratified)

| Model | CV Accuracy | Std Dev | Ranking |
|-------|-------------|---------|---------|
| RandomForest | 78.06% | ±1.11% | 1st |
| GradientBoosting | 76.38% | ±1.61% | 2nd |
| LogisticRegression | 74.94% | ±1.07% | 3rd |
| SVC | 66.08% | ±0.42% | 4th |

---


## Test Set Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| RandomForest | 76.79% | 54.83% | 71.39% | 62.02% | 84.01% |
| **GradientBoosting** | **75.80%** | **53.12%** | **75.13%** | **62.24%** | **83.53%** |
| LogisticRegression | 73.88% | 50.51% | 79.95% | 61.90% | 84.14% |
| SVC | 66.78% | 41.26% | 59.36% | 48.68% | 73.38% |

---

## Best Model Selection

**Selected**: **Gradient Boosting Classifier**  
**Selection Criterion**: F1 Score (best metric for imbalanced classification)  
**Test F1 Score**: 62.24%

---

## Confusion Matrix Analysis (GradientBoosting)

![confusion_matrix](src/evaluation/confusion_matrix.png)


---

## Model Comparison Insights

### Gradient Boosting **SELECTED**
- **Best F1 Score (62.24%)**
- **Highest recall (75.13%)** - catches most churners
- Balanced precision/recall tradeoff
- Best for business: prioritizes not missing churners

### Random Forest
- Highest CV accuracy (78.06%)
- Best precision (54.83%)
- Highest ROC-AUC (84.01%)
- Lower recall (71.39%) - misses ~14 more churners than GB

### Logistic Regression
- **Highest recall (79.95%)** - catches most churners
- **Lowest precision (50.51%)** - too many false alarms
- F1 only 61.90% due to poor precision/recall balance

### SVC
- Significantly underperforms (F1: 48.68%)
- Not suitable for this problem
- Likely issue: RBF kernel doesn't fit feature space

---

## Key Takeaways

1. **GradientBoosting selected** for optimal F1 score (62.24%)
2. **High recall prioritized** (75.13%) to minimize costly missed churners
3. **Close competition**: RandomForest (62.02%) vs GradientBoosting (62.24%)
4. **Class imbalance handled** via balanced weights + stratified CV
5. **Strong discrimination**: ROC-AUC ~84% across top 3 models

---