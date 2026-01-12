from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from esg_training import rag_answer, init_resources

app = FastAPI(title="ESG Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    category: str
    question: str


@app.on_event("startup")
def startup_event():
    init_resources()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: QueryRequest):
    try:
        answer, sources = rag_answer(
            req.question,
            req.category
        )
        return {
            "recommendation": answer,
            "sources": sources
        }
    except Exception as e:
        return {"error": str(e)}
