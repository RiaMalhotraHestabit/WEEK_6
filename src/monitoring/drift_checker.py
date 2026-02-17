import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timezone

# PATHS
PREDICTION_LOG_PATH  = Path("prediction_logs.csv")
REFERENCE_DATA_PATH  = Path("src/data/processed/X_train.csv")
DRIFT_REPORT_PATH    = Path("src/monitoring/drift_report.json")
DRIFT_THRESHOLD      = 0.05  # p-value threshold for KS test

# CORE DRIFT DETECTION
def load_reference_data() -> pd.DataFrame:
    """Load training data as reference distribution"""
    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Reference data not found: {REFERENCE_DATA_PATH}")
    df = pd.read_csv(REFERENCE_DATA_PATH)
    print(f" Reference data loaded: {df.shape}")
    return df

def load_prediction_logs() -> pd.DataFrame:
    """Load recent prediction logs as incoming distribution"""
    if not PREDICTION_LOG_PATH.exists():
        raise FileNotFoundError(f"No prediction logs found: {PREDICTION_LOG_PATH}")
    df = pd.read_csv(PREDICTION_LOG_PATH)
    print(f"✓ Prediction logs loaded: {df.shape}")
    return df

def detect_numerical_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str
) -> dict:
    """
    Detect drift in numerical features using Kolmogorov-Smirnov test

    KS test compares two distributions:
    - p-value < 0.05 = significant drift detected
    - p-value >= 0.05 = no significant drift
    """
    # Remove NaN values
    ref_clean = reference.dropna()
    cur_clean = current.dropna()

    if len(cur_clean) < 10:
        return {
            "feature": feature_name,
            "type": "numerical",
            "drift_detected": False,
            "reason": "insufficient data (< 10 samples)"
        }

    # KS Test
    ks_stat, p_value = stats.ks_2samp(ref_clean, cur_clean)
    drift_detected = bool(p_value < DRIFT_THRESHOLD)

    return {
        "feature": feature_name,
        "type": "numerical",
        "drift_detected": drift_detected,
        "ks_statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_value), 4),
        "reference_mean": round(float(ref_clean.mean()), 4),
        "current_mean": round(float(cur_clean.mean()), 4),
        "reference_std": round(float(ref_clean.std()), 4),
        "current_std": round(float(cur_clean.std()), 4),
        "mean_shift": round(float(cur_clean.mean() - ref_clean.mean()), 4)
    }

def detect_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str
) -> dict:
    """
    Detect drift in categorical features using Chi-Square test

    Compares category distributions between reference and current data
    """
    ref_clean = reference.dropna()
    cur_clean = current.dropna()

    if len(cur_clean) < 10:
        return {
            "feature": feature_name,
            "type": "categorical",
            "drift_detected": False,
            "reason": "insufficient data (< 10 samples)"
        }

    # Get all categories
    all_categories = set(ref_clean.unique()) | set(cur_clean.unique())

    # Count distributions
    ref_counts = ref_clean.value_counts()
    cur_counts = cur_clean.value_counts()

    ref_dist = {cat: ref_counts.get(cat, 0) / len(ref_clean) for cat in all_categories}
    cur_dist = {cat: cur_counts.get(cat, 0) / len(cur_clean) for cat in all_categories}

    # Chi-square test
    ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
    cur_freq = [cur_counts.get(cat, 0) for cat in all_categories]

    # Avoid zero expected frequencies
    if min(ref_freq) == 0:
        drift_detected = False
        chi2_stat = 0
        p_value = 1.0
    else:
        chi2_stat, p_value = stats.chisquare(
            f_obs=cur_freq,
            f_exp=[f * len(cur_clean) / len(ref_clean) for f in ref_freq]
        )
        drift_detected = p_value < DRIFT_THRESHOLD

    return {
        "feature": feature_name,
        "type": "categorical",
        "drift_detected": drift_detected,
        "chi2_statistic": round(float(chi2_stat), 4),
        "p_value": round(float(p_value), 4),
        "reference_distribution": {k: round(v, 4) for k, v in ref_dist.items()},
        "current_distribution": {k: round(v, 4) for k, v in cur_dist.items()}
    }

# ACCURACY DECAY MONITORING
def check_prediction_drift(logs_df: pd.DataFrame) -> dict:
    """
    Monitor prediction distribution drift over time.
    """
    if len(logs_df) < 10:
        return {
            "status": "insufficient data for prediction drift analysis",
            "overall_churn_rate": None,        
            "older_predictions_churn_rate": None,
            "recent_predictions_churn_rate": None,
            "churn_rate_shift": None,
            "drift_detected": False,
            "alert": f"Need 10+ predictions, only {len(logs_df)} so far"
        }
    
    overall_churn_rate = logs_df["prediction"].mean()

    # Recent vs older predictions 
    split = int(len(logs_df) * 0.8)
    older = logs_df.iloc[:split]["prediction"].mean() if split > 0 else overall_churn_rate
    recent = logs_df.iloc[split:]["prediction"].mean() if split < len(logs_df) else overall_churn_rate

    drift_detected = abs(recent - older) > 0.1  # 10% shift threshold

    return {
        "overall_churn_rate": round(float(overall_churn_rate), 4),
        "older_predictions_churn_rate": round(float(older), 4),
        "recent_predictions_churn_rate": round(float(recent), 4),
        "churn_rate_shift": round(float(recent - older), 4),
        "drift_detected": drift_detected,
        "alert": "Prediction distribution shifted! Consider retraining." if drift_detected else "Stable"
    }

# MAIN DRIFT REPORT
def run_drift_check():
    """Run complete drift detection and generate report"""

    print("\n" + "="*60)
    print("DATA DRIFT MONITORING REPORT")
    print("="*60)
    try:
        reference_df = load_reference_data()
        logs_df      = load_prediction_logs()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Get numerical features
    numerical_features = reference_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Run drift detection per feature
    print(f"\n{'='*60}")
    print(f"FEATURE DRIFT ANALYSIS ({len(numerical_features)} features)")
    print(f"{'='*60}")

    drift_results = []
    drifted_features = []

    for feature in numerical_features:
        if feature in logs_df.columns:
            result = detect_numerical_drift(
                reference_df[feature],
                logs_df[feature],
                feature
            )
            drift_results.append(result)

            status = "DRIFT" if result["drift_detected"] else "OK"
            print(f"  {feature:<35} {status}")

            if result["drift_detected"]:
                drifted_features.append(feature)

    # Prediction distribution drift
    print(f"\n{'='*60}")
    print("PREDICTION DISTRIBUTION DRIFT")
    print(f"{'='*60}")
    prediction_drift = check_prediction_drift(logs_df)
    if prediction_drift["overall_churn_rate"] is None:
        print(f"  Status: {prediction_drift['alert']}")
    else:
        print(f"  Overall churn rate:  {prediction_drift['overall_churn_rate']:.2%}")
        print(f"  Recent churn rate:   {prediction_drift['recent_predictions_churn_rate']:.2%}")
        print(f"  Older churn rate:    {prediction_drift['older_predictions_churn_rate']:.2%}")
        print(f"  Status: {prediction_drift['alert']}")

    # Build final report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_samples": len(reference_df),
        "current_samples": len(logs_df),
        "drift_threshold": float(DRIFT_THRESHOLD),
        "total_features_checked": len(drift_results),
        "features_with_drift": len(drifted_features),
        "drifted_features": drifted_features,
        "overall_drift_detected": bool(len(drifted_features) > 0),
        "prediction_drift": {
            k: bool(v) if isinstance(v, bool) else v 
            for k, v in prediction_drift.items()
        },
        "feature_results": [
            {k: bool(v) if isinstance(v, bool) else v 
            for k, v in result.items()}
            for result in drift_results
        ],
        "recommendation": (
            "RETRAIN MODEL - Significant drift detected!"
            if len(drifted_features) > 3
            else "Model is stable - no retraining needed"
        )
    }

    # Save report
    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("DRIFT SUMMARY")
    print(f"{'='*60}")
    print(f"  Features checked:    {len(drift_results)}")
    print(f"  Features drifted:    {len(drifted_features)}")
    print(f"  Drifted features:    {drifted_features if drifted_features else 'None'}")
    print(f"  Recommendation:      {report['recommendation']}")
    print(f"  Report saved to:     {DRIFT_REPORT_PATH}")
    print(f"{'='*60}\n")

    return report

if __name__ == "__main__":
    run_drift_check()