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

# ============================================
# CONFIGURATION
# ============================================
class Config:
    DATA_PATH = Path('src/data/processed/final.csv')
    MODEL_DIR = Path('src/models')
    EVAL_DIR = Path('src/evaluation')
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# DATA LOADING
# ============================================
def load_data():
    """Load pre-processed and split data from build_features.py"""
    print("Loading pre-processed data...")
    
    # Load train and test sets (already encoded and split)
    X_train = pd.read_csv('src/data/processed/X_train.csv')
    X_test = pd.read_csv('src/data/processed/X_test.csv')
    y_train = pd.read_csv('src/data/processed/y_train.csv')['Churn']
    y_test = pd.read_csv('src/data/processed/y_test.csv')['Churn']
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# ============================================
# TRAIN/TEST SPLIT
# ============================================
def train_test_split_custom(X, y, test_size=0.2):
    """Stratified train-test split to preserve class distribution"""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=Config.RANDOM_STATE,
        stratify=y  # Ensures same class ratio in train and test
    )
    
    print(f"\nTrain set: {X_train.shape}, Churn rate: {y_train.mean():.3f}")
    print(f"Test set: {X_test.shape}, Churn rate: {y_test.mean():.3f}")
    
    return X_train, X_test, y_train, y_test

# ============================================
# MODEL DEFINITIONS WITH CLASS BALANCING
# ============================================
def get_models():
    """Initialize models with class_weight='balanced' for imbalance handling"""
    models = {
        'LogisticRegression': LogisticRegression(
            class_weight='balanced',  # Handles class imbalance
            max_iter=1000,
            random_state=Config.RANDOM_STATE,
            solver='liblinear'
        ),
        'RandomForest': RandomForestClassifier(
            class_weight='balanced',  # handles class imbalance
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
            # Note: GB doesn't have class_weight, we'll use sample_weight
        ),
        'SVC': SVC(
            class_weight='balanced',  # Handles class imbalance
            probability=True,
            kernel='rbf',
            random_state=Config.RANDOM_STATE
        )
    }
    return models

# ============================================
# CROSS-VALIDATION TRAINING
# ============================================
def train_with_cv(models, X_train, y_train):
    """Train all models with stratified k-fold cross-validation"""
    cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    cv_results = {}
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION TRAINING (5-Fold)")
    print("="*60)
    
    # Calculate sample weights for GradientBoosting (since it doesn't support class_weight)
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y_train)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use sample_weight for GradientBoosting
        if name == 'GradientBoosting':
            # Train with sample weights
            model.fit(X_train, y_train, sample_weight=sample_weights)
            # Manual CV with sample weights
            cv_scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                sw_cv = sample_weights[train_idx]
                
                temp_model = GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=5, 
                    random_state=Config.RANDOM_STATE
                )
                temp_model.fit(X_cv_train, y_cv_train, sample_weight=sw_cv)
                cv_scores.append(temp_model.score(X_cv_val, y_cv_val))
            
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
        else:
            # Standard cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_mean = scores.mean()
            cv_std = scores.std()
            
            # Fit on full training data
            model.fit(X_train, y_train)
        
        cv_results[name] = {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'model': model
        }
        
        print(f"  CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    return cv_results

# ============================================
# MODEL EVALUATION
# ============================================
def evaluate_models(cv_results, X_test, y_test):
    """Evaluate all models on test set and compute metrics"""
    metrics = {}
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    for name, result in cv_results.items():
        model = result['model']
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }
        
        print(f"\n{name}:")
        for metric, value in metrics[name].items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
    
    return metrics

# ============================================
# BEST MODEL SELECTION
# ============================================
def select_best_model(metrics, cv_results):
    """Select best model based on F1 score (balances precision and recall)"""
    # Sort by F1 score (better for imbalanced datasets than accuracy)
    best_model_name = max(metrics.items(), key=lambda x: x[1]['f1'])[0]
    best_model = cv_results[best_model_name]['model']
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print(f"F1 Score: {metrics[best_model_name]['f1']:.4f}")
    print(f"Recall: {metrics[best_model_name]['recall']:.4f}")
    print("="*60)
    
    return best_model_name, best_model

# ============================================
# SAVE RESULTS
# ============================================
def save_model(model, name):
    """Save the best model as pickle"""
    model_path = Config.MODEL_DIR / 'best_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save model name
    with open(Config.MODEL_DIR / 'best_model_name.txt', 'w') as f:
        f.write(name)
    
    print(f"\nBest model saved to: {model_path}")

def save_metrics(metrics, best_model_name):
    """Save metrics as JSON"""
    metrics_path = Config.EVAL_DIR / 'metrics.json'
    
    # Convert numpy types to native Python types for JSON serialization
    metrics_clean = {}
    for model, metric_dict in metrics.items():
        metrics_clean[model] = {k: float(v) if v is not None else None 
                                for k, v in metric_dict.items()}
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_clean, f, indent=4)
    
    print(f"Metrics saved to: {metrics_path}")

def plot_confusion_matrix(model, X_test, y_test, model_name):
    """Generate and save confusion matrix"""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    cm_path = Config.EVAL_DIR / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Print confusion matrix values
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives: {tp}")
    print(f"\n  Recall (Sensitivity): {tp/(tp+fn):.4f}")
    print(f"  Specificity: {tn/(tn+fp):.4f}")

def save_classification_report(model, X_test, y_test, model_name):
    """Save detailed classification report"""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, 
                                    target_names=['No Churn', 'Churn'],
                                    digits=4)
    
    report_path = Config.EVAL_DIR / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("="*60 + "\n")
        f.write(report)
    
    print(f"Classification report saved to: {report_path}")

# ============================================
# MAIN PIPELINE
# ============================================
def main():
    """Complete training pipeline"""
    print("\n" + "="*60)
    print("TELECOM CHURN PREDICTION - TRAINING PIPELINE")
    print("="*60)
    
    # 1. Train-test split
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Initialize models with class balancing
    models = get_models()
    
    # 3. Train with cross-validation
    cv_results = train_with_cv(models, X_train, y_train)
    
    # 4. Evaluate on test set
    metrics = evaluate_models(cv_results, X_test, y_test)
    
    # 5. Select best model
    best_model_name, best_model = select_best_model(metrics, cv_results)
    
    # 6. Save results
    save_model(best_model, best_model_name)
    save_metrics(metrics, best_model_name)
    plot_confusion_matrix(best_model, X_test, y_test, best_model_name)
    save_classification_report(best_model, X_test, y_test, best_model_name)
    
    # 7. Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {metrics[best_model_name]['accuracy']:.4f}")
    print(f"Test Recall: {metrics[best_model_name]['recall']:.4f}")
    print(f"Test F1 Score: {metrics[best_model_name]['f1']:.4f}")
    print(f"Test ROC-AUC: {metrics[best_model_name]['roc_auc']:.4f}")
    print("\nAll outputs saved to models/ and evaluation/ directories")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()