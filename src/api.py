"""
src/api.py
----------
Step 3 of MLOps Pipeline — FastAPI REST API for Model Serving

Endpoints:
  GET  /health           → Returns API status + model metrics
  GET  /info             → Returns model feature info and input format
  POST /predict          → Single prediction (returns churn probability)
  POST /predict/batch    → Batch prediction (list of customers)

Run locally:  uvicorn src.api:app --reload --port 8000
Docs UI:      http://localhost:8000/docs   (Swagger auto-generated)
"""

import json
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator     # ✅ FIXED: field_validator (V2)
from pydantic import ConfigDict                            # ✅ FIXED: ConfigDict (V2)
from typing import List
from pathlib import Path

# ─── Load Artifacts ───────────────────────────────────────────────────────────
MODEL_DIR = Path("artifacts")

def load_artifacts():
    try:
        with open(MODEL_DIR / "model.pkl",    "rb") as f: model    = pickle.load(f)
        with open(MODEL_DIR / "encoders.pkl", "rb") as f: encoders = pickle.load(f)
        with open(MODEL_DIR / "scaler.pkl",   "rb") as f: scaler   = pickle.load(f)
        with open(MODEL_DIR / "metrics.json")       as f: metrics  = json.load(f)
        return model, encoders, scaler, metrics
    except FileNotFoundError:
        raise RuntimeError("Model artifacts not found. Run `python src/train.py` first.")

model, encoders, scaler, train_metrics = load_artifacts()

CATEGORICAL_COLS = ["contract", "internet_service", "payment_method"]
NUMERIC_COLS     = ["tenure", "monthly_charges", "total_charges",
                    "senior_citizen", "dependents", "tech_support",
                    "online_security", "num_services"]

VALID_CONTRACT = ["Month-to-month", "One year", "Two year"]
VALID_INTERNET = ["DSL", "Fiber optic", "No"]
VALID_PAYMENT  = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]


# ─── Pydantic Schema (Request Validation) ────────────────────────────────────
class CustomerInput(BaseModel):

    # ✅ FIXED: model_config replaces class Config (Pydantic V2)
    model_config = ConfigDict(
        json_schema_extra={                            # ✅ FIXED: json_schema_extra (was schema_extra)
            "example": {
                "tenure": 12,
                "monthly_charges": 85.5,
                "total_charges": 1026.0,
                "senior_citizen": 0,
                "dependents": 0,
                "tech_support": 1,
                "online_security": 0,
                "num_services": 3,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check"
            }
        }
    )

    tenure:           int   = Field(..., ge=0, le=120, description="Months with company (0–120)")
    monthly_charges:  float = Field(..., ge=0, le=200, description="Monthly bill amount in USD")
    total_charges:    float = Field(..., ge=0,          description="Total amount charged")
    senior_citizen:   int   = Field(..., ge=0, le=1,   description="Is senior citizen? (0 or 1)")
    dependents:       int   = Field(..., ge=0, le=1,   description="Has dependents? (0 or 1)")
    tech_support:     int   = Field(..., ge=0, le=1,   description="Has tech support? (0 or 1)")
    online_security:  int   = Field(..., ge=0, le=1,   description="Has online security? (0 or 1)")
    num_services:     int   = Field(..., ge=1, le=8,   description="Number of subscribed services")
    contract:         str   = Field(...,                description="Contract type")
    internet_service: str   = Field(...,                description="Internet service type")
    payment_method:   str   = Field(...,                description="Payment method")

    # ✅ FIXED: @field_validator + @classmethod (Pydantic V2 way)
    @field_validator("contract")
    @classmethod
    def validate_contract(cls, v):
        if v not in VALID_CONTRACT:
            raise ValueError(f"contract must be one of {VALID_CONTRACT}")
        return v

    @field_validator("internet_service")
    @classmethod
    def validate_internet(cls, v):
        if v not in VALID_INTERNET:
            raise ValueError(f"internet_service must be one of {VALID_INTERNET}")
        return v

    @field_validator("payment_method")
    @classmethod
    def validate_payment(cls, v):
        if v not in VALID_PAYMENT:
            raise ValueError(f"payment_method must be one of {VALID_PAYMENT}")
        return v


# ─── Prediction Response Schema ───────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_prediction:  int
    churn_probability: float
    risk_level:        str
    recommendation:    str


class BatchResponse(BaseModel):
    total_customers:    int
    predicted_churners: int
    churn_rate:         str
    predictions:        List[PredictionResponse]


# ─── Preprocessing Helper ─────────────────────────────────────────────────────
def preprocess_input(customer: CustomerInput) -> np.ndarray:
    # encoders is a single LabelEncoder saved from training
    # We re-encode each categorical value using the saved encoder
    cat_encoded = [
        encoders["contract"].transform([customer.contract])[0],
        encoders["internet_service"].transform([customer.internet_service])[0],
        encoders["payment_method"].transform([customer.payment_method])[0],
    ]
    num_features = [
        customer.tenure, customer.monthly_charges, customer.total_charges,
        customer.senior_citizen, customer.dependents, customer.tech_support,
        customer.online_security, customer.num_services,
    ]
    features = np.array([cat_encoded + num_features])
    return scaler.transform(features)


def make_prediction(customer: CustomerInput) -> PredictionResponse:
    X     = preprocess_input(customer)
    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    if proba >= 0.70:
        risk, rec = "HIGH",   "Immediate retention call + discount offer recommended"
    elif proba >= 0.40:
        risk, rec = "MEDIUM", "Send loyalty offer and check satisfaction score"
    else:
        risk, rec = "LOW",    "Customer is likely to stay — standard engagement"

    return PredictionResponse(
        churn_prediction  = pred,
        churn_probability = round(proba, 4),
        risk_level        = risk,
        recommendation    = rec,
    )


# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Customer Churn Prediction API",
    description = "MLOps Pipeline — XGBoost model served via FastAPI",
    version     = "1.0.0",
)


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status"       : "healthy",
        "model"        : "XGBoost Churn Classifier",
        "train_metrics": train_metrics,
    }


@app.get("/info", tags=["System"])
def model_info():
    return {
        "features": {
            "categorical": {col: list(encoders[col].classes_) for col in CATEGORICAL_COLS},
            "numeric"    : NUMERIC_COLS,
        },
        "output": {
            "churn_prediction"  : "0 = No Churn, 1 = Churn",
            "churn_probability" : "0.0 to 1.0",
            "risk_level"        : "LOW | MEDIUM | HIGH",
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    """
    Predict churn probability for a single customer.
    Returns prediction, probability, risk level, and a recommendation.
    """
    try:
        return make_prediction(customer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(customers: List[CustomerInput]):
    """
    Predict churn for a list of customers (max 100).
    Returns summary stats + individual predictions.
    """
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Batch limit is 100 customers per request.")
    try:
        predictions = [make_prediction(c) for c in customers]
        churners    = sum(p.churn_prediction for p in predictions)
        return BatchResponse(
            total_customers    = len(predictions),
            predicted_churners = churners,
            churn_rate         = f"{churners / len(predictions):.1%}",
            predictions        = predictions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


report_text = "Patient" 


# V3 CODE (CURRENT) — From medical_summarizer.py

# 0. BioBERT — Named Entity Pre-Extraction (lines 30-45)
from transformers import pipeline as hf_pipeline
ner_pipeline = hf_pipeline("ner", model="dmis-lab/biobert-base-cased-v1.2",
                             aggregation_strategy="simple")
bio_entities = ner_pipeline(report_text)  # Extracts: Condition, Test, Value spans
structured_entities = {e["word"]: e["entity_group"] for e in bio_entities}
# structured_entities passed into _SYSTEM_PROMPT as pre-labeled context

# 1. Structured system prompt (lines 60-158)
_SYSTEM_PROMPT = """You are SwasthyaMitra, an expert medical report analyst
who explains reports in simple layman language...

You MUST follow this EXACT output format:
🩺 **What This Report Tests**
📊 **What Your Report Shows**
  ✅ **Normal Values**
  ⚠️ **Slightly Elevated / Low**
  ❌ **Abnormal / Concerning Values**
🧾 **Overall Simple Summary**
🥗 **General Suggestions**
🚨 **Important**
🇮🇳 **सरल सारांश (Hindi)**
🇮🇳🔸 **સरળ સારાંશ (Gujarati)**
🍎 **Customized Diet Plan**

RULES:
1. ALWAYS use the exact emoji headers
2. Be warm, reassuring but honest
3. Use plain English — no medical jargon
...10 strict rules..."""

# 2. Llama 3.3 70B — Primary Summarization Engine (line 190-199)
from transformers import pipeline
llama_pipe = pipeline("text-generation", model="meta-llama/Llama-3.3-70B-Instruct")
temperature=0.3,    # Low creativity for factual output
max_tokens=4096,    # Enough for full trilingual + diet plan
top_p=0.9,

# 3. Offline fallback (lines 237-345)
def _summarize_generic_labs_fallback(report_text):
    # Regex-based value extraction and classification
    pattern = r"^([A-Za-z][\w\s]+?)\s+(\d+\.?\d*)\s*(.*)$"
    # Extracts test name, value, reference range
    # Classifies as ✅ Normal / ⚠️ Above / ⚠️ Below