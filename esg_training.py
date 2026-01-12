import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "esg_extracted_data.csv"

df = None
vectorizer = None
documents = None


def init_resources():
    global df, vectorizer, documents

    if df is not None:
        return

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"company", "category", "keyword", "extracted_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    documents = (
        "Company: " + df["company"].astype(str) +
        " Category: " + df["category"].astype(str) +
        " Keyword: " + df["keyword"].astype(str) +
        " Text: " + df["extracted_text"].astype(str)
    ).tolist()

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(documents)


def rag_answer(question: str, category: str):
    init_resources()

    filtered = df[df["category"].str.lower() == category.lower()]

    if filtered.empty:
        return "No ESG data found for this category.", []

    texts = (
        "Keyword: " + filtered["keyword"] +
        " Text: " + filtered["extracted_text"]
    ).tolist()

    query_vec = vectorizer.transform([question])
    text_vecs = vectorizer.transform(texts)

    similarities = cosine_similarity(query_vec, text_vecs)[0]
    best_idx = similarities.argmax()

    return texts[best_idx], texts
