import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "esg_extracted_data.csv"

df = None
vectorizer = None
tfidf_matrix = None
documents = None


def init_resources():
    global df, vectorizer, tfidf_matrix, documents

    if df is not None:
        return

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found in project root")

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"state", "category", "question", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    documents = (
        "State: " + df["state"].astype(str) +
        " Category: " + df["category"].astype(str) +
        " Question: " + df["question"].astype(str) +
        " Answer: " + df["answer"].astype(str)
    ).tolist()

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)


def rag_answer(question: str, state: str, category: str):
    init_resources()

    filtered = df[
        (df["state"].str.lower() == state.lower()) &
        (df["category"].str.lower() == category.lower())
    ]

    if filtered.empty:
        return "No ESG data found for the given inputs.", []

    texts = (
        "Question: " + filtered["question"] +
        " Answer: " + filtered["answer"]
    ).tolist()

    query_vec = vectorizer.transform([question])
    text_vecs = vectorizer.transform(texts)

    similarities = cosine_similarity(query_vec, text_vecs)[0]
    best_idx = similarities.argmax()

    return texts[best_idx], texts
