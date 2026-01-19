# main.py
# PRODUCTION-READY ESG RISK BACKEND (FASTAPI)
# FINAL VERSION – ANDROID SAFE

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uuid
import os

from esg_training import analyze_esg_data

# -------------------------------------------------
# PATH SAFETY (RENDER SAFE)
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SOURCE = os.path.join(BASE_DIR, "esg_extracted_data.csv")

if not os.path.exists(DATA_SOURCE):
    raise RuntimeError("ESG dataset missing. Deployment aborted.")

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------

ALLOWED_CATEGORIES = {"Environmental", "Social", "Governance"}
MODEL_VERSION = "ESG-RISK-v1.0"
DATA_FILE_NAME = "esg_extracted_data.csv"

# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------

app = FastAPI(
    title="Production ESG Risk Analysis API",
    version=MODEL_VERSION
)

# -------------------------------------------------
# CORS (ANDROID / WEB SAFE)
# -------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# SCHEMAS
# -------------------------------------------------

class AnalyzeRequest(BaseModel):
    category: str

class AnalyzeResponse(BaseModel):
    request_id: str
    timestamp: str
    category: str
    risk_level: str
    summary: str
    key_findings: list[str]
    data_source: str
    model_version: str

class EntryResponse(BaseModel):
    id: str
    category: str
    risk_level: str
    summary: str
    key_findings: list[str]
    timestamp: str
    model_version: str

class InsightResponse(BaseModel):
    category: str
    pillar: str
    score: float
    risk: str
    confidence: float
    model_version: str

# -------------------------------------------------
# ANALYZE ENDPOINT (SINGLE CATEGORY)
# -------------------------------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):

    if request.category not in ALLOWED_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Allowed: {list(ALLOWED_CATEGORIES)}"
        )

    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    try:
        result = analyze_esg_data(request.category)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="ESG analysis engine failed"
        )

    return AnalyzeResponse(
        request_id=request_id,
        timestamp=timestamp,
        category=request.category,
        risk_level=result.get("risk_level", "UNKNOWN"),
        summary=result.get("summary", ""),
        key_findings=result.get("key_findings", []),
        data_source=DATA_FILE_NAME,
        model_version=MODEL_VERSION
    )

# -------------------------------------------------
# ENTRIES ENDPOINT (REAL DATA)
# -------------------------------------------------

@app.get("/entries", response_model=list[EntryResponse])
def get_entries():

    entries = []

    for category in ALLOWED_CATEGORIES:
        try:
            result = analyze_esg_data(category)
        except Exception:
            continue

        entries.append(
            EntryResponse(
                id=str(uuid.uuid4()),
                category=category,
                risk_level=result.get("risk_level", "UNKNOWN"),
                summary=result.get("summary", ""),
                key_findings=result.get("key_findings", []),
                timestamp=datetime.utcnow().isoformat(),
                model_version=MODEL_VERSION
            )
        )

    return entries

# -------------------------------------------------
# INSIGHTS ENDPOINT (FOR DASHBOARD / MOBILE UI)
# -------------------------------------------------

@app.get("/insights", response_model=list[InsightResponse])
def get_insights():

    insights = []

    for category in ALLOWED_CATEGORIES:
        try:
            result = analyze_esg_data(category)
        except Exception:
            continue

        risk = result.get("risk_level", "UNKNOWN")

        score = {
            "LOW": 85.0,
            "MEDIUM": 60.0,
            "HIGH": 30.0
        }.get(risk, 50.0)

        confidence = round(len(result.get("key_findings", [])) / 10, 2)

        insights.append(
            InsightResponse(
                category=category,
                pillar=category[0],  # E / S / G
                score=score,
                risk=risk,
                confidence=min(confidence, 1.0),
                model_version=MODEL_VERSION
            )
        )

    return insights

# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "OK",
        "service": "ESG Risk Analysis Backend",
        "model_version": MODEL_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }
