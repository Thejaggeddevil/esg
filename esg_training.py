import os
import re
import numpy as np
import pandas as pd
from time import sleep

from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from groq import Groq

# load .env variables
load_dotenv()

# ✅ Portable CSV path
INPUT_CSV = "esg_with_state.csv"

# Lazy-loaded globals
df = None
model = None
embeddings = None
index = None


def init_resources():
    global df, model, embeddings, index

    if df is not None:
        return

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"{INPUT_CSV} not found")

    df = pd.read_csv(INPUT_CSV)

    df["context"] = (
        "Company: " + df["Company"].astype(str) +
        ", State: " + df["State"].astype(str) +
        ", Category: " + df["Category"].astype(str) +
        ", Keyword: " + df["Keyword"].astype(str) +
        ", Extracted Info: " + df["Extracted_Text"].astype(str) +
        ", Marketing Perspective: " + df["Marketing_Perspective"].astype(str) +
        ", Cost Perspective: " + df["Cost_Perspective"].astype(str) +
        ", Ground Impact: " + df["Ground_Impact"].astype(str)
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(">>> Initializing embeddings & FAISS index")
    embeddings = model.encode(df["context"].tolist(), show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))


def retrieve_similar_by_state(query, state, category, top_k=5):
    df_state = df[
        (df["State"].str.lower() == state.lower()) &
        (df["Category"].str.lower() == category.lower())
    ]

    if df_state.empty:
        return pd.DataFrame()

    state_embeddings = model.encode(df_state["context"].tolist(), show_progress_bar=False)

    index_state = faiss.IndexFlatL2(state_embeddings.shape[1])
    index_state.add(np.array(state_embeddings))

    query_embedding = model.encode([query])
    _, indices = index_state.search(
        np.array(query_embedding),
        min(top_k, len(df_state))
    )

    return df_state.iloc[indices[0]][[
        "Company", "State", "Category", "Keyword",
        "Extracted_Text", "Marketing_Perspective", "Cost_Perspective"
    ]]


def summarize_achievements(docs):
    achievements = []

    for _, row in docs.iterrows():
        clean = re.sub(r"\s+", " ", str(row["Extracted_Text"])).strip()
        first_sentence = clean.split(".")[0]

        achievements.append({
            "summary": "• " + first_sentence,
            "marketing": row.get("Marketing_Perspective", ""),
            "cost": row.get("Cost_Perspective", "")
        })

    return achievements[:4]


# ✅ Groq client (NO base_url, NO proxy)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def rag_answer(query, state=None, category=None, top_k=5):
    init_resources()

    docs = retrieve_similar_by_state(query, state, category)

    if docs.empty:
        return (
            "No relevant ESG data found for this state and category.",
            "",
            [],
            []
        )

    context = "\n\n".join(docs["Extracted_Text"].astype(str).tolist())

    prompt = f"""
You are an ESG advisor.

Context below contains ESG achievements already done by other companies in {state}.

Context:
{context}

Task:
Convert these achievements into forward-looking recommendations that OTHER companies operating in {state} can adopt.

Rules:
- Use only the context
- 3–4 bullet points
- Each bullet starts with a verb
"""

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
    )

    generated_text = completion.choices[0].message.content.strip()
    achievements = summarize_achievements(docs)

    return generated_text, context, [], achievements
