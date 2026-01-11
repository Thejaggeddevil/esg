import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# =========================
# CONFIG
# =========================
INPUT_CSV = "esg_extracted_data.csv"

STATE_COL = "state"
CATEGORY_COL = "category"
QUESTION_COL = "question"
ANSWER_COL = "answer"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# =========================
# GLOBALS (loaded once)
# =========================
_df = None
_embeddings = None
_index = None
_model = None
_client = None


# =========================
# INIT RESOURCES (SAFE)
# =========================
def init_resources():
    global _df, _embeddings, _index, _model, _client

    if _df is not None:
        return  # already loaded

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"{INPUT_CSV} not found")

    # Load CSV
    df = pd.read_csv(INPUT_CSV)

    # Validate columns
    required = {STATE_COL, CATEGORY_COL, QUESTION_COL, ANSWER_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Normalize text
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()
    df[CATEGORY_COL] = df[CATEGORY_COL].astype(str).str.strip().str.lower()

    # Combine text for embeddings
    texts = (
        "Question: " + df[QUESTION_COL].astype(str) +
        " | Answer: " + df[ANSWER_COL].astype(str) +
        " | State: " + df[STATE_COL] +
        " | Category: " + df[CATEGORY_COL]
    ).tolist()

    # Embedding model
    _model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = _model.encode(texts, convert_to_numpy=True)

    # FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # LLM client
    _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    _df = df
    _embeddings = embeddings
    _index = index


# =========================
# RETRIEVAL
# =========================
def retrieve_similar_by_state(query, state, category, top_k=5):
    init_resources()

    state = state.strip().lower()
    category = category.strip().lower()

    mask = (
        (_df[STATE_COL] == state) &
        (_df[CATEGORY_COL] == category)
    )

    filtered = _df[mask]
    if filtered.empty:
        return []

    query_emb = _model.encode([query], convert_to_numpy=True)
    distances, indices = _index.search(query_emb, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(_df):
            row = _df.iloc[idx]
            if row[STATE_COL] == state and row[CATEGORY_COL] == category:
                results.append(row[ANSWER_COL])

    return results[:top_k]


# =========================
# RAG ANSWER
# =========================
def rag_answer(question, state, category):
    docs = retrieve_similar_by_state(question, state, category)

    if not docs:
        return (
            "No ESG-specific recommendations found for this state and category.",
            []
        )

    context = "\n".join(f"- {d}" for d in docs)

    prompt = f"""
You are an ESG policy advisor.

Context:
{context}

Question:
{question}

Give a concise, practical ESG recommendation.
"""

    response = _client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an expert ESG consultant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=300
    )

    return response.choices[0].message.content.strip(), docs
