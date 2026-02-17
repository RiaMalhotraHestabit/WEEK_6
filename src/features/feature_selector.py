import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

def remove_high_correlation(X, threshold=0.9):
    """Remove features with correlation above threshold"""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    print(f"  - Dropping {len(to_drop)} highly correlated features (>{threshold})")
    if to_drop:
        print(f"    Dropped: {to_drop[:5]}{'...' if len(to_drop) > 5 else ''}")
    
    X_reduced = X.drop(columns=to_drop)
    return X_reduced, to_drop

def select_features(X, y, top_k=30, return_feature_names=True):
    """
    Select top features using correlation and mutual information
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix (TRAIN data only!)
    y : pd.Series
        Target variable (TRAIN data only!)
    top_k : int
        Number of top features to keep
    return_feature_names : bool
        If True, returns (X_selected, feature_names)
        If False, returns only X_selected (for backward compatibility)
    
    Returns:
    --------
    X_selected : pd.DataFrame
        Selected features
    feature_names : list (if return_feature_names=True)
        List of selected feature names
    """
    
    print(f"  - Starting with {X.shape[1]} features")
    
    # Step 1: Remove highly correlated features
    X_reduced, dropped_corr = remove_high_correlation(X, threshold=0.9)
    print(f"  - After correlation filter: {X_reduced.shape[1]} features")
    
    # Adjust top_k if needed (edge case handling)
    if top_k > X_reduced.shape[1]:
        print(f"  ⚠ Warning: top_k ({top_k}) > available features ({X_reduced.shape[1]})")
        top_k = X_reduced.shape[1]
        print(f"  - Adjusted top_k to {top_k}")
    
    # Step 2: Compute mutual information
    print(f"  - Computing mutual information scores...")
    mi = mutual_info_classif(X_reduced, y, random_state=42)
    mi_df = pd.DataFrame({
        "Feature": X_reduced.columns, 
        "MI_Score": mi
    }).sort_values(by="MI_Score", ascending=False)
    
    # Step 3: Keep top K features
    top_features = mi_df.head(top_k)["Feature"].tolist()
    X_selected = X_reduced[top_features]
    
    print(f"  - Selected top {len(top_features)} features by MI score")
    print(f"  - Top 5 features: {top_features[:5]}")
    
    # Step 4: Plot feature importance
    plot_feature_importance(mi_df, top_n=20)
    
    # Step 5: Save feature selection report
    save_feature_report(mi_df, dropped_corr, top_features)
    
    if return_feature_names:
        return X_selected, top_features
    else:
        return X_selected

def plot_feature_importance(mi_df, top_n=20):
    """Plot top N features by mutual information score"""
    plt.figure(figsize=(12, 6))
    top_mi = mi_df.head(top_n)
    
    plt.barh(range(len(top_mi)), top_mi["MI_Score"].values)
    plt.yticks(range(len(top_mi)), top_mi["Feature"].values)
    plt.xlabel("Mutual Information Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"Top {top_n} Features by Mutual Information", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest score at top
    plt.tight_layout()
    
    # Save plot using Path
    plot_path = Path(__file__).parent.parent / "evaluation" / "feature_importance.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Feature importance plot saved: {plot_path}")

def save_feature_report(mi_df, dropped_corr, selected_features):
    """Save detailed feature selection report"""
    report_path = Path(__file__).parent / "feature_selection_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FEATURE SELECTION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"1. Correlation Filtering\n")
        f.write(f"   - Dropped {len(dropped_corr)} highly correlated features\n")
        if dropped_corr:
            f.write(f"   - Dropped features: {', '.join(dropped_corr)}\n")
        f.write("\n")
        
        f.write(f"2. Mutual Information Selection\n")
        f.write(f"   - Selected top {len(selected_features)} features\n")
        f.write(f"   - Selected features:\n")
        for feat in selected_features:
            score = mi_df[mi_df['Feature'] == feat]['MI_Score'].values[0]
            f.write(f"     • {feat}: {score:.4f}\n")
        f.write("\n")
        
        f.write(f"3. All Features Ranked by MI Score\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Rank':<6} {'Feature':<40} {'MI_Score':<10}\n")
        f.write("-" * 70 + "\n")
        for idx, row in mi_df.iterrows():
            rank = mi_df.index.get_loc(idx) + 1
            f.write(f"{rank:<6} {row['Feature']:<40} {row['MI_Score']:<10.4f}\n")
    
    print(f"  ✓ Feature selection report saved: {report_path}")