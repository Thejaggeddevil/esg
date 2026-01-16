# main.py
# PRODUCTION-READY ESG RISK BACKEND (FASTAPI)
# This file REPLACES your existing main.py
# Other files (esg_training.py, CSV, requirements) remain unchanged

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uuid

from esg_training import analyze_esg_data  # existing logic you already have

# ---------------------------
# CONSTANTS (PRODUCTION)
# ---------------------------

ALLOWED_CATEGORIES = {"Environmental", "Social", "Governance"}
MODEL_VERSION = "ESG-RISK-v1.0"
DATA_SOURCE = "esg_extracted_data.csv"

# ---------------------------
# FASTAPI APP
# ---------------------------

app = FastAPI(
    title="Production ESG Risk Analysis API",
    version=MODEL_VERSION
)

# ---------------------------
# REQUEST / RESPONSE SCHEMAS
# ---------------------------

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

# ---------------------------
# API ENDPOINT
# ---------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    # 1. CATEGORY VALIDATION
    if request.category not in ALLOWED_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Allowed values: {list(ALLOWED_CATEGORIES)}"
        )

    # 2. GENERATE AUDIT METADATA
    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    # 3. RUN EXISTING ESG LOGIC (UNCHANGED)
    result = analyze_esg_data(request.category)

    # EXPECTED FROM esg_training.py:
    # result = {
    #   "risk_level": "HIGH / MEDIUM / LOW",
    #   "summary": "...",
    #   "key_findings": [...]
    # }

    # 4. STRONG, AUDIT-SAFE RESPONSE
    return AnalyzeResponse(
        request_id=request_id,
        timestamp=timestamp,
        category=request.category,
        risk_level=result.get("risk_level", "UNKNOWN"),
        summary=result.get("summary", ""),
        key_findings=result.get("key_findings", []),
        data_source=DATA_SOURCE,
        model_version=MODEL_VERSION
    )

# ---------------------------
# HEALTH CHECK (OPTIONAL BUT PRODUCTION-STANDARD)
# ---------------------------

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "service": "ESG Risk Analysis Backend",
        "model_version": MODEL_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }
