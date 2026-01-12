import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

DATA_PATH = "esg_extracted_data.csv"

df = None
index = None
model = None
documents = None


def init_resources():
    global df, index, model, documents

    if df is not None:
        return  # already initialized

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")

    # Load CSV
    df = pd.read_csv(DATA_PATH)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"state", "category", "question", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Build documents
    documents = (
        "State: " + df["state"].astype(str) +
        " | Category: " + df["category"].astype(str) +
        " | Question: " + df["question"].astype(str) +
        " | Answer: " + df["answer"].astype(str)
    ).tolist()

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents, convert_to_numpy=True)

    # FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)


def retrieve_similar_by_state(query, state, category, k=3):
    init_resources()

    filtered = df[
        (df["state"].str.lower() == state.lower()) &
        (df["category"].str.lower() == category.lower())
    ]

    if filtered.empty:
        return []

    texts = (
        "Question: " + filtered["question"] +
        " | Answer: " + filtered["answer"]
    ).tolist()

    embeddings = model.encode(texts, convert_to_numpy=True)
    q_emb = model.encode([query], convert_to_numpy=True)

    temp_index = faiss.IndexFlatL2(embeddings.shape[1])
    temp_index.add(embeddings)

    _, I = temp_index.search(q_emb, min(k, len(texts)))
    return [texts[i] for i in I[0]]


def rag_answer(question, state, category):
    docs = retrieve_similar_by_state(question, state, category)

    if not docs:
        return "No ESG data found for given inputs.", []

    return docs[0], docs
