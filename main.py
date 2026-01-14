from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from esg_training import analyze_esg_risk, init_resources

app = FastAPI(title="ESG Risk Scanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    category: str

@app.on_event("startup")
def startup_event():
    init_resources()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: QueryRequest):
    try:
        result = analyze_esg_risk(req.category)
        return {
            "analysis": result
        }
    except Exception as e:
        return {"error": str(e)}
