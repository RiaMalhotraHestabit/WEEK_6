import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import os

def remove_high_correlation(X, threshold=0.9):
    """Remove features with correlation above threshold"""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_reduced = X.drop(columns=to_drop)
    return X_reduced

def select_features(X, y):
    """Select top features using correlation and mutual information"""
    # Remove highly correlated features
    X = remove_high_correlation(X)
    
    # Compute mutual information
    mi = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({"Feature": X.columns, "MI Score": mi}).sort_values(by="MI Score", ascending=False)
    
    # Keep top 30 features
    top_features = mi_df.head(30)["Feature"].values
    X_selected = X[top_features]
    
    # Plot top 20 features
    plt.figure(figsize=(10,5))
    mi_df.head(20).plot(kind="bar", x="Feature", y="MI Score", legend=False)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation", "feature_importance.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    
    return X_selected
