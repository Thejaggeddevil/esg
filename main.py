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
    state: str
    category: str
    question: str


@app.on_event("startup")
def startup_event():
    # Load everything once at startup
    init_resources()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: QueryRequest):
    try:
        answer, sources = rag_answer(
            req.question,
            req.state,
            req.category
        )
        return {
            "recommendation": answer,
            "sources": sources
        }
    except Exception as e:
        # Never hide errors again
        return {"error": str(e)}
