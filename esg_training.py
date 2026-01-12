import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

CSV_PATH = "esg_extracted_data.csv"

df = None
model = None
index = None
texts = None


def init_resources():
    global df, model, index, texts

    if df is not None:
        return

    # Load CSV
    df = pd.read_csv(CSV_PATH)

    # Validate required columns
    required = {"Category", "Extracted_Text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Clean data
    df = df.dropna(subset=["Extracted_Text", "Category"])

    texts = df["Extracted_Text"].astype(str).tolist()

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)


def retrieve_similar(question, category, top_k=3):
    init_resources()

    query_vec = model.encode([question]).astype("float32")
    _, indices = index.search(query_vec, top_k)

    results = []
    for i in indices[0]:
        row = df.iloc[i]
        if row["Category"].lower() == category.lower():
            results.append(row["Extracted_Text"])

    return results[:top_k]


def rag_answer(question, state, category):
    docs = retrieve_similar(question, category)

    if not docs:
        return "No relevant ESG data found for this category.", []

    answer = " ".join(docs[:2])
    return answer, docs
