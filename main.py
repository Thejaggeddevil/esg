from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from esg_training import rag_answer

app = FastAPI(title="ESG Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    state: str
    category: str
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: QueryRequest):
    answer, sources = rag_answer(
        req.question,
        req.state,
        req.category
    )
    return {
        "recommendation": answer,
        "sources": sources
    }
