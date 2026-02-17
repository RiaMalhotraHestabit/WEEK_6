from pathlib import Path

class Paths:
    # Data
    X_TRAIN = Path('src/data/processed/X_train.csv')
    X_TEST = Path('src/data/processed/X_test.csv')
    Y_TRAIN = Path('src/data/processed/y_train.csv')
    Y_TEST = Path('src/data/processed/y_test.csv')
    
    # Models
    BASELINE_MODEL = Path('src/models/best_model.pkl')
    TUNED_MODEL = Path('src/models/best_tuned_model.pkl')
    
    # Outputs
    TUNING_RESULTS = Path('src/tuning/results.json')
    CONFUSION_MATRIX_TUNED = Path('src/evaluation/confusion_matrix_tuned.png')
    SHAP_SUMMARY = Path('src/evaluation/shap_summary.png')
    FEATURE_IMPORTANCE = Path('src/evaluation/feature_importance.png')
    ERROR_ANALYSIS = Path('src/evaluation/error_analysis.png')

# SETTINGS
RANDOM_STATE = 42
N_TRIALS = 100
CV_FOLDS = 5
N_JOBS = -1
SHAP_SAMPLES = 100