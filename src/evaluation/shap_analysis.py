import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from config.config import Paths, SHAP_SAMPLES


def load():
    with open(Paths.TUNED_MODEL, "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_csv(Paths.X_TEST)
    y_test = pd.read_csv(Paths.Y_TEST)["Churn"]

    # Sample for SHAP (speed optimization)
    sample_size = min(SHAP_SAMPLES, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)

    return model, X_test, X_sample, y_test


def generate_shap_summary(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):  # binary classification
        shap_values = shap_values[1]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(Paths.SHAP_SUMMARY, dpi=300, bbox_inches="tight")
    plt.close()


def generate_feature_importance(model, X_sample):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices][::-1])
    plt.yticks(range(len(indices)), X_sample.columns[indices][::-1])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(Paths.FEATURE_IMPORTANCE, dpi=300, bbox_inches="tight")
    plt.close()


def generate_error_analysis(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = pd.crosstab(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(Paths.ERROR_ANALYSIS, dpi=300)
    plt.close()

def print_top_features(model, X):
    import numpy as np
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    print("\nTOP 10 FEATURES DRIVING CHURN")
    print("=" * 50)

    for rank, idx in enumerate(indices, 1):
        print(f"{rank:2d}. {X.columns[idx]:30s} {importances[idx]:.4f}")

def main():
    model, X_test, X_sample, y_test = load()

    generate_shap_summary(model, X_sample)
    generate_feature_importance(model, X_sample)
    print_top_features(model, X_sample)
    generate_error_analysis(model, X_test, y_test)

    print("Explainability artifacts generated.")


if __name__ == "__main__":
    main()
