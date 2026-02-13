"""
Hyperparameter tuning with Optuna for RandomForest
"""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, confusion_matrix

from config.config import Paths, RANDOM_STATE, N_TRIALS, CV_FOLDS, N_JOBS

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

Paths.TUNING_RESULTS.parent.mkdir(exist_ok=True)
Paths.CONFUSION_MATRIX_TUNED.parent.mkdir(exist_ok=True)


def load_data():
    X_train = pd.read_csv(Paths.X_TRAIN)
    X_test = pd.read_csv(Paths.X_TEST)
    y_train = pd.read_csv(Paths.Y_TRAIN)["Churn"]
    y_test = pd.read_csv(Paths.Y_TEST)["Churn"]
    return X_train, X_test, y_train, y_test


def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
    }

    model = RandomForestClassifier(**params)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=N_JOBS).mean()


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics, y_pred


def main():
    X_train, X_test, y_train, y_test = load_data()

    # baseline comparison
    baseline_metrics = {}
    if Paths.BASELINE_MODEL.exists():
        with open(Paths.BASELINE_MODEL, "rb") as f:
            baseline_model = pickle.load(f)
        baseline_metrics, _ = evaluate(baseline_model, X_test, y_test)

    # Optuna tuning
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=N_TRIALS)

    best_params = study.best_params
    best_params.update({
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS
    })

    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)

    tuned_metrics, y_pred = evaluate(model, X_test, y_test)

    # Save model
    with open(Paths.TUNED_MODEL, "wb") as f:
        pickle.dump(model, f)

    # Save results
    results = {
        "best_params": best_params,
        "best_cv_f1": float(study.best_value),
        "baseline": baseline_metrics,
        "tuned": tuned_metrics,
    }

    with open(Paths.TUNING_RESULTS, "w") as f:
        json.dump(results, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(Paths.CONFUSION_MATRIX_TUNED, dpi=300)
    plt.close()

    print("Tuning complete. Model and results saved.")


if __name__ == "__main__":
    main()
