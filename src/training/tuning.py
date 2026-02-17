import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
import pickle
import warnings
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from optuna.samplers import TPESampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    f1_score, recall_score, precision_score, 
    accuracy_score, roc_auc_score, confusion_matrix
)
from config.config import Paths, RANDOM_STATE, N_TRIALS, CV_FOLDS, N_JOBS

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Create directories
Paths.TUNING_RESULTS.parent.mkdir(exist_ok=True, parents=True)
Paths.CONFUSION_MATRIX_TUNED.parent.mkdir(exist_ok=True, parents=True)

def load_data():
    """Load pre-processed data from Day 2"""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    X_train = pd.read_csv(Paths.X_TRAIN)
    X_test = pd.read_csv(Paths.X_TEST)
    y_train = pd.read_csv(Paths.Y_TRAIN)["Churn"]
    y_test = pd.read_csv(Paths.Y_TEST)["Churn"]
    
    print(f"✓ Train: {X_train.shape}")
    print(f"✓ Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def load_baseline_model():
    """Load the best model from Day 3"""
    print("\n" + "="*60)
    print("LOADING BASELINE MODEL (DAY 3)")
    print("="*60)
    
    if not Paths.BASELINE_MODEL.exists():
        print("⚠ Warning: No baseline model found!")
        return None, None
    
    with open(Paths.BASELINE_MODEL, "rb") as f:
        baseline_model = pickle.load(f)
    
    # Load model name
    model_name_path = Paths.BASELINE_MODEL.parent / "best_model_name.txt"
    if model_name_path.exists():
        best_model_name = model_name_path.read_text().strip()
        print(f"✓ Loaded baseline model: {best_model_name}")
    else:
        best_model_name = "Unknown"
        print(f"✓ Loaded baseline model (name unknown)")
    
    return baseline_model, best_model_name

def objective(trial, X, y):
    """Optuna objective function for GradientBoosting"""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": RANDOM_STATE,
    }
    
    model = GradientBoostingClassifier(**params)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Compute sample weights for class imbalance
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y)
    
    # Manual CV with sample weights
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        sw_cv = sample_weights[train_idx]
        
        temp_model = GradientBoostingClassifier(**params)
        temp_model.fit(X_train_cv, y_train_cv, sample_weight=sw_cv)
        y_pred = temp_model.predict(X_val_cv)
        scores.append(f1_score(y_val_cv, y_pred))
    
    return sum(scores) / len(scores)

def evaluate(model, X_test, y_test):
    """Evaluate model on test set"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    
    return metrics, y_pred

def plot_confusion_matrix(y_test, y_pred, title="Tuned Model"):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn']
    )
    plt.title(f'Confusion Matrix - {title}', fontsize=14, fontweight='bold')
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    plt.savefig(Paths.CONFUSION_MATRIX_TUNED, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Confusion matrix saved to: {Paths.CONFUSION_MATRIX_TUNED}")
    
    # Print breakdown
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives:  {tn:4d}")
    print(f"  False Positives: {fp:4d}")
    print(f"  False Negatives: {fn:4d}")
    print(f"  True Positives:  {tp:4d}")

def print_comparison(baseline_metrics, tuned_metrics):
    """Print side-by-side comparison"""
    print("\n" + "="*60)
    print("BASELINE vs TUNED COMPARISON")
    print("="*60)
    print(f"{'Metric':<15} {'Baseline':<12} {'Tuned':<12} {'Change':<12}")
    print("-"*60)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        baseline_val = baseline_metrics.get(metric, 0)
        tuned_val = tuned_metrics.get(metric, 0)
        change = tuned_val - baseline_val
        change_str = f"{change:+.4f}" if change != 0 else "0.0000"

        print(f"{metric:<15} {baseline_val:<12.4f} {tuned_val:<12.4f} {change_str:<8}")

def main():
    """Complete hyperparameter tuning pipeline"""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING PIPELINE (DAY 4)")
    print("="*60)
    
    # 1. Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Load baseline model
    baseline_model, baseline_name = load_baseline_model()
    baseline_metrics = {}
    if baseline_model:
        baseline_metrics, _ = evaluate(baseline_model, X_test, y_test)
        print(f"✓ Baseline F1 Score: {baseline_metrics['f1']:.4f}")
    
    # 3. Optuna tuning
    print("\n" + "="*60)
    print(f"OPTUNA HYPERPARAMETER SEARCH ({N_TRIALS} trials)")
    print("="*60)
    print("Optimizing GradientBoosting for F1 score...")
    print("This may take 10-30 minutes...\n")
    
    study = optuna.create_study(
        direction="maximize", 
        sampler=TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train), 
        n_trials=N_TRIALS,
        show_progress_bar=True
    )
    
    print(f"\n Best CV F1 Score: {study.best_value:.4f}")
    print(f" Best parameters found:")
    for param, value in study.best_params.items():
        print(f"    {param}: {value}")
    
    # 4. Train final model with best parameters
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("="*60)
    
    best_params = study.best_params.copy()
    best_params["random_state"] = RANDOM_STATE
    
    # Compute sample weights
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y_train)
    
    model = GradientBoostingClassifier(**best_params)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    print(" Model trained")
    
    # 5. Evaluate
    tuned_metrics, y_pred = evaluate(model, X_test, y_test)
    print(f" Tuned F1 Score: {tuned_metrics['f1']:.4f}")
    
    # 6. Save model
    with open(Paths.TUNED_MODEL, "wb") as f:
        pickle.dump(model, f)
    print(f" Tuned model saved to: {Paths.TUNED_MODEL}")
    
    # 7. Save results
    results = {
        "model_type": "GradientBoostingClassifier",
        "baseline_model": baseline_name,
        "best_params": best_params,
        "best_cv_f1": float(study.best_value),
        "n_trials": N_TRIALS,
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
        "improvement": {
            metric: float(tuned_metrics[metric] - baseline_metrics.get(metric, 0))
            for metric in tuned_metrics.keys()
        }
    }
    
    with open(Paths.TUNING_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {Paths.TUNING_RESULTS}")
    
    # 8. Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, "Tuned GradientBoosting")
    
    # 9. Print comparison
    if baseline_metrics:
        print_comparison(baseline_metrics, tuned_metrics)
    
    # 10. Final summary
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("="*60)
    print(f"Model Type: GradientBoostingClassifier")
    print(f"F1 Score Improvement: {tuned_metrics['f1'] - baseline_metrics.get('f1', 0):+.4f}")
    print(f"Recall Improvement: {tuned_metrics['recall'] - baseline_metrics.get('recall', 0):+.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()