import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# CONFIGURATION

class Config:
    # Paths
    X_TRAIN_PATH = Path('src/data/processed/X_train.csv')
    X_TEST_PATH = Path('src/data/processed/X_test.csv')
    Y_TRAIN_PATH = Path('src/data/processed/y_train.csv')
    Y_TEST_PATH = Path('src/data/processed/y_test.csv')
    
    MODEL_DIR = Path('src/models')
    EVAL_DIR = Path('src/evaluation')
    
    # Training config
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


# DATA LOADING
def load_data():
    print("\n" + "="*60)
    print("LOADING DATA FROM DAY 2 PIPELINE")
    print("="*60)
    
    # Load train and test sets (already split, encoded, and feature-selected)
    X_train = pd.read_csv(Config.X_TRAIN_PATH)
    X_test = pd.read_csv(Config.X_TEST_PATH)
    y_train = pd.read_csv(Config.Y_TRAIN_PATH)['Churn']
    y_test = pd.read_csv(Config.Y_TEST_PATH)['Churn']
    
    print(f" Train set: {X_train.shape}")
    print(f" Test set: {X_test.shape}")
    print(f" Train churn rate: {y_train.mean():.2%}")
    print(f" Test churn rate: {y_test.mean():.2%}")
    print(f" Number of features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


# MODEL DEFINITIONS WITH CLASS BALANCING

def get_models():
    """Initialize 4 models with class_weight='balanced' for imbalance handling"""
    models = {
        'LogisticRegression': LogisticRegression(
            class_weight='balanced',  
            max_iter=1000,
            random_state=Config.RANDOM_STATE,
            solver='liblinear'
        ),
        'RandomForest': RandomForestClassifier(
            class_weight='balanced',  
            n_estimators=100,
            max_depth=10,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=Config.RANDOM_STATE
            # Note: GradientBoosting doesn't have class_weight parameter
        ),
        'SVC': SVC(
            class_weight='balanced',  
            probability=True,  # Needed for predict_proba()
            kernel='rbf',
            random_state=Config.RANDOM_STATE
        )
    }
    
    print(f"\n Initialized {len(models)} models with class balancing")
    return models


# CROSS-VALIDATION TRAINING
def train_with_cv(models, X_train, y_train):
    """Train all models with stratified 5-fold cross-validation"""
    cv = StratifiedKFold(
        n_splits=Config.CV_FOLDS, 
        shuffle=True, 
        random_state=Config.RANDOM_STATE
    )
    cv_results = {}
    
    print("\n" + "="*60)
    print(f"CROSS-VALIDATION TRAINING ({Config.CV_FOLDS}-FOLD)")
    print("="*60)
    
    # Calculate sample weights for GradientBoosting
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y_train)
    
    for name, model in models.items():
        print(f"\n[{list(models.keys()).index(name)+1}/{len(models)}] Training {name}...")
        
        # Special handling for GradientBoosting (doesn't support class_weight)
        if name == 'GradientBoosting':
            # Manual CV with sample weights
            cv_scores = []
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                sw_cv = sample_weights[train_idx]
                
                temp_model = GradientBoostingClassifier(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=5, 
                    random_state=Config.RANDOM_STATE
                )
                temp_model.fit(X_cv_train, y_cv_train, sample_weight=sw_cv)
                score = temp_model.score(X_cv_val, y_cv_val)
                cv_scores.append(score)
                print(f"  Fold {fold}: {score:.4f}")
            
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            # Fit on full training data with sample weights
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            # Standard cross-validation for other models
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv, 
                scoring='accuracy', 
                n_jobs=-1
            )
            
            for fold, score in enumerate(scores, 1):
                print(f"  Fold {fold}: {score:.4f}")
            
            cv_mean = scores.mean()
            cv_std = scores.std()
            
            # Fit on full training data
            model.fit(X_train, y_train)
        
        cv_results[name] = {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'model': model
        }
        
        print(f"  ✓ CV Accuracy: {cv_mean:.4f} (±{cv_std:.4f})")
    
    return cv_results

# MODEL EVALUATION
def evaluate_models(cv_results, X_test, y_test):
    """Evaluate all models on test set and compute all required metrics"""
    metrics = {}
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    for name, result in cv_results.items():
        model = result['model']
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {metrics[name]['accuracy']:.4f}")
        print(f"  Precision: {metrics[name]['precision']:.4f}")
        print(f"  Recall:    {metrics[name]['recall']:.4f}")
        print(f"  F1 Score:  {metrics[name]['f1']:.4f}")
        if metrics[name]['roc_auc'] is not None:
            print(f"  ROC-AUC:   {metrics[name]['roc_auc']:.4f}")
    
    return metrics


# BEST MODEL SELECTION

def select_best_model(metrics, cv_results):
    """Select best model based on F1 score (best for imbalanced datasets)"""
    # Sort by F1 score (balances precision and recall)
    best_model_name = max(metrics.items(), key=lambda x: x[1]['f1'])[0]
    best_model = cv_results[best_model_name]['model']
    
    print("\n" + "="*60)
    print("BEST MODEL SELECTION")
    print("="*60)
    print(f"✓ Selected: {best_model_name}")
    print(f"  Criteria: Highest F1 Score")
    print(f"  F1 Score: {metrics[best_model_name]['f1']:.4f}")
    print(f"  Recall:   {metrics[best_model_name]['recall']:.4f}")
    print(f"  Precision: {metrics[best_model_name]['precision']:.4f}")
    print("="*60)
    
    return best_model_name, best_model


# SAVE RESULTS
def save_model(model, name):
    """Save the best model as pickle"""
    model_path = Config.MODEL_DIR / 'best_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(Config.MODEL_DIR / 'best_model_name.txt', 'w') as f:
        f.write(name)
    
    print(f"\n Best model saved to: {model_path}")

def save_metrics(metrics, best_model_name):
    """Save metrics as JSON"""
    metrics_path = Config.EVAL_DIR / 'metrics.json'
    
    # Convert numpy types to native Python types for JSON serialization
    metrics_clean = {}
    for model, metric_dict in metrics.items():
        metrics_clean[model] = {
            k: float(v) if v is not None else None 
            for k, v in metric_dict.items()
        }
    
    # Add best model indicator
    metrics_output = {
        'best_model': best_model_name,
        'models': metrics_clean
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    print(f"Metrics saved to: {metrics_path}")

def plot_confusion_matrix(model, X_test, y_test, model_name):
    """Generate and save confusion matrix"""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn'],
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    
    cm_path = Config.EVAL_DIR / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to: {cm_path}")
    
    # Print confusion matrix breakdown
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix Breakdown:")
    print(f"    True Negatives:  {tn:4d} (Correctly predicted No Churn)")
    print(f"    False Positives: {fp:4d} (Incorrectly predicted Churn)")
    print(f"    False Negatives: {fn:4d} (Missed Churn cases)")
    print(f"    True Positives:  {tp:4d} (Correctly predicted Churn)")
    print(f"\n  Recall (Sensitivity):    {tp/(tp+fn):.4f}")
    print(f"  Specificity:             {tn/(tn+fp):.4f}")

def save_classification_report(model, X_test, y_test, model_name):
    """Save detailed classification report"""
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, 
        target_names=['No Churn', 'Churn'],
        digits=4
    )
    
    report_path = Config.EVAL_DIR / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"✓ Classification report saved to: {report_path}")


# MAIN PIPELINE

def main():
    """Complete training pipeline"""
    print("\n" + "="*60)
    print("TELECOM CHURN PREDICTION - TRAINING PIPELINE (DAY 3)")
    print("="*60)
    
    # 1. Load pre-split data from Day 2
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Initialize 4 models with class balancing
    models = get_models()
    
    # 3. Train with 5-fold cross-validation
    cv_results = train_with_cv(models, X_train, y_train)
    
    # 4. Evaluate on test set
    metrics = evaluate_models(cv_results, X_test, y_test)
    
    # 5. Select best model (by F1 score)
    best_model_name, best_model = select_best_model(metrics, cv_results)
    
    # 6. Save all results
    save_model(best_model, best_model_name)
    save_metrics(metrics, best_model_name)
    plot_confusion_matrix(best_model, X_test, y_test, best_model_name)
    save_classification_report(best_model, X_test, y_test, best_model_name)
    
    # 7. Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Model:     {best_model_name}")
    print(f"Test Accuracy:  {metrics[best_model_name]['accuracy']:.4f}")
    print(f"Test Precision: {metrics[best_model_name]['precision']:.4f}")
    print(f"Test Recall:    {metrics[best_model_name]['recall']:.4f}")
    print(f"Test F1 Score:  {metrics[best_model_name]['f1']:.4f}")
    print(f"Test ROC-AUC:   {metrics[best_model_name]['roc_auc']:.4f}")
    print("\n All outputs saved:")
    print(f"   • {Config.MODEL_DIR}/best_model.pkl")
    print(f"   • {Config.EVAL_DIR}/metrics.json")
    print(f"   • {Config.EVAL_DIR}/confusion_matrix.png")
    print(f"   • {Config.EVAL_DIR}/classification_report.txt")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()