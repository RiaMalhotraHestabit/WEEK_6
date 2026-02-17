import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import uuid
import pickle
import joblib
import logging
import pandas as pd
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from config.config import Paths, RANDOM_STATE

#LOGGING SETUP
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

#PATHS
PREDICTION_LOG_PATH = Path("prediction_logs.csv")
FEATURE_LIST_PATH   = Path("src/features/feature_list.json")
PREPROCESSOR_PATH   = Path("src/features/preprocessor.pkl")
MODEL_VERSION       = "1.0.0-tuned-gradientboosting"

# APP INITIALIZATION
app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Predict customer churn using a tuned GradientBoosting model",
    version=MODEL_VERSION
)


# INPUT SCHEMA (Pydantic Validation)

class ChurnPredictionRequest(BaseModel):

    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 if senior citizen, 0 otherwise")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    gender: str = Field(..., description="Male or Female")
    tenure: int = Field(..., ge=0, le=72, description="Months with company (0-72)")
    Contract: str = Field(..., description="Month-to-month, One year, Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(..., description="Payment method used")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charge amount")
    TotalCharges: float = Field(..., ge=0, description="Total charges to date")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")

    @field_validator("Partner", "Dependents", "PhoneService", "PaperlessBilling")
    @classmethod
    def validate_yes_no(cls, v):
        if v not in ["Yes", "No"]:
            raise ValueError(f"Must be 'Yes' or 'No', got '{v}'")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v):
        valid = ["Month-to-month", "One year", "Two year"]
        if v not in valid:
            raise ValueError(f"Must be one of {valid}")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet(cls, v):
        valid = ["DSL", "Fiber optic", "No"]
        if v not in valid:
            raise ValueError(f"Must be one of {valid}")
        return v

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in ["Male", "Female"]:
            raise ValueError("Must be 'Male' or 'Female'")
        return v

    model_config={
        "json_schema_extra ": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.35,
                "TotalCharges": 904.2
            }
        }

    }
        
# OUTPUT SCHEMA
class ChurnPredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    request_id: str
    timestamp: str
    model_version: str
    prediction: int
    prediction_label: str
    churn_probability: float
    risk_level: str

# MODEL LOADING
def load_model():
    """Load tuned model with version tracking"""
    if not Paths.TUNED_MODEL.exists():
        raise FileNotFoundError(f"Model not found: {Paths.TUNED_MODEL}")
    with open(Paths.TUNED_MODEL, "rb") as f:
        model = pickle.load(f)
    logger.info(f"✓ Model loaded: {Paths.TUNED_MODEL}")
    return model

def load_preprocessor():
    """Load fitted preprocessor from Day 2"""
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info(f"✓ Preprocessor loaded: {PREPROCESSOR_PATH}")
    return preprocessor

def load_feature_list():
    """Load selected feature list from Day 2"""
    import json
    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(f"Feature list not found: {FEATURE_LIST_PATH}")
    with open(FEATURE_LIST_PATH) as f:
        feature_config = json.load(f)
    logger.info(f"Feature list loaded: {len(feature_config['selected_features'])} features")
    return feature_config

# Load at startup
model        = load_model()
preprocessor = load_preprocessor()
feature_config = load_feature_list()

# FEATURE ENGINEERING 
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same feature engineering as Day 2 pipeline"""

    df["AvgChargePerMonth"]   = df["TotalCharges"] / (df["tenure"] + 1)
    df["IsLongTerm"]          = (df["tenure"] > 24).astype(int)
    df["HighMonthlyCharges"]  = (df["MonthlyCharges"] > 64.76).astype(int)  # train median
    df["HasStreaming"]        = ((df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")).astype(int)
    df["HasOnlineSecurity"]   = (df["OnlineSecurity"] == "Yes").astype(int)
    df["HasTechSupport"]      = (df["TechSupport"] == "Yes").astype(int)
    df["IsMonthToMonth"]      = (df["Contract"] == "Month-to-month").astype(int)
    df["SeniorWithPartner"]   = ((df["SeniorCitizen"] == 1) & (df["Partner"] == "Yes")).astype(int)
    df["PartnerAndDependents"]= ((df["Partner"] == "Yes") & (df["Dependents"] == "Yes")).astype(int)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
    )

    # TotalServices
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    df["TotalServices"] = sum((df[col] == "Yes").astype(int) for col in service_cols)

    return df

def preprocess_input(data: ChurnPredictionRequest) -> pd.DataFrame:
    df = pd.DataFrame([data.model_dump()])
    df = engineer_features(df)
    X_processed = preprocessor.transform(df)
    cat_cols = feature_config["categorical_cols"]
    num_cols = feature_config["numeric_cols"]
    feature_names = num_cols + list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
    )

    X_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
        columns=feature_names
    )
    selected = feature_config["selected_features"]
    available = [f for f in selected if f in X_df.columns]
    X_final = X_df[available]

    return X_final

# PREDICTION LOGGING
def log_prediction(request_id, timestamp, input_data, prediction, probability):
    """Log prediction to CSV file"""
    log_entry = {
        "request_id": request_id,
        "timestamp": timestamp,
        "model_version": MODEL_VERSION,
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        **input_data.model_dump()
    }

    log_df = pd.DataFrame([log_entry])

    if PREDICTION_LOG_PATH.exists():
        log_df.to_csv(PREDICTION_LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(PREDICTION_LOG_PATH, mode="w", header=True, index=False)

    logger.info(f"Prediction logged | ID: {request_id} | Churn: {prediction} | P: {probability:.4f}")

def get_risk_level(probability: float) -> str:
    """Convert probability to risk level"""
    if probability >= 0.75:
        return "HIGH"
    elif probability >= 0.50:
        return "MEDIUM"
    elif probability >= 0.25:
        return "LOW"
    else:
        return "VERY LOW"

# API ENDPOINTS
@app.get("/",include_in_schema=False)
def root():
    return {
        "message": "Telecom Churn Prediction API",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/model-info",include_in_schema=False)
def model_info():
    """Return model version and feature information"""
    return {
        "model_version": MODEL_VERSION,
        "model_type": "GradientBoostingClassifier (Tuned)",
        "n_features": len(feature_config["selected_features"]),
        "selected_features": feature_config["selected_features"],
        "training_recall": 0.8048,
        "training_f1": 0.6357,
        "training_roc_auc": 0.8457
    }

@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(data: ChurnPredictionRequest):
    # Generate unique request ID and timestamp
    request_id = str(uuid.uuid4())
    timestamp  = datetime.now(timezone.utc).isoformat()

    try:
        X = preprocess_input(data)

        # Make prediction
        prediction   = int(model.predict(X)[0])
        probability  = float(model.predict_proba(X)[0][1])
        risk_level   = get_risk_level(probability)

        log_prediction(request_id, timestamp, data, prediction, probability)

        return ChurnPredictionResponse(
            request_id=request_id,
            timestamp=timestamp,
            model_version=MODEL_VERSION,
            prediction=prediction,
            prediction_label="Churn" if prediction == 1 else "No Churn",
            churn_probability=round(probability, 4),
            risk_level=risk_level
        )

    except Exception as e:
        logger.error(f"Prediction failed | ID: {request_id} | Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/logs/summary",include_in_schema=False)
def logs_summary():
    """Return summary of prediction logs"""
    if not PREDICTION_LOG_PATH.exists():
        return {"message": "No predictions logged yet"}

    df = pd.read_csv(PREDICTION_LOG_PATH)
    return {
        "total_predictions": len(df),
        "churn_predictions": int(df["prediction"].sum()),
        "no_churn_predictions": int((df["prediction"] == 0).sum()),
        "churn_rate": round(df["prediction"].mean(), 4),
        "avg_churn_probability": round(df["churn_probability"].mean(), 4),
        "latest_prediction": df["timestamp"].iloc[-1] if len(df) > 0 else None
    }