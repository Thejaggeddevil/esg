import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

CSV_FILE = "esg_extracted_data.csv"

df = None
model = None
index = None


def init_resources():
    global df, model, index

    if df is not None:
        return

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"{CSV_FILE} not found")

    df = pd.read_csv(CSV_FILE)

    REQUIRED = [
        "Company",
        "State",
        "Category",
        "Extracted_Text"
    ]

    for col in REQUIRED:
        if col not in df.columns:
            raise ValueError(f"CSV missing column: {col}")

    df["context"] = (
        "Company: " + df["Company"].astype(str) +
        ", State: " + df["State"].astype(str) +
        ", Category: " + df["Category"].astype(str) +
        ", Details: " + df["Extracted_Text"].astype(str)
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(df["context"].tolist(), show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))


def retrieve_docs(question, state, category, top_k=5):
    init_resources()

    filtered = df[
        (df["State"].str.lower() == state.lower()) &
        (df["Category"].str.lower() == category.lower())
    ]

    if filtered.empty:
        return []

    emb = model.encode(filtered["context"].tolist())
    temp_index = faiss.IndexFlatL2(emb.shape[1])
    temp_index.add(np.array(emb))

    q_emb = model.encode([question])
    _, idx = temp_index.search(np.array(q_emb), min(top_k, len(filtered)))

    return filtered.iloc[idx[0]]


client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def rag_answer(question, state, category):
    docs = retrieve_docs(question, state, category)

    if len(docs) == 0:
        return "No ESG data found for this state and category.", ""

    context = "\n".join(docs["Extracted_Text"].astype(str).tolist())

    prompt = f"""
You are an ESG consultant.

Based ONLY on the following ESG data:
{context}

Give 3–4 actionable ESG recommendations
for companies operating in {state}
under the category {category}.
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip(), context
