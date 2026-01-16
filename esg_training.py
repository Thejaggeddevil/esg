# esg_training.py
# PRODUCTION-SAFE ESG ANALYSIS ENGINE (CSV-ALIGNED)

import os
import pandas as pd

# ---------------------------
# LOAD CSV SAFELY (RENDER SAFE)
# ---------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "esg_extracted_data.csv")

if not os.path.exists(CSV_PATH):
    raise RuntimeError("ESG CSV file not found")

df = pd.read_csv(CSV_PATH)

# Normalize column names once (IMPORTANT)
df.columns = [c.strip() for c in df.columns]

# ---------------------------
# ESG ANALYSIS LOGIC
# ---------------------------

def analyze_esg_data(category: str) -> dict:
    """
    category: Environmental / Social / Governance
    """

    if "Category" not in df.columns or "Extracted_Text" not in df.columns:
        raise RuntimeError("CSV schema mismatch")

    # Case-insensitive category match
    filtered = df[df["Category"].str.lower() == category.lower()]

    if filtered.empty:
        return {
            "risk_level": "LOW",
            "summary": f"No significant ESG risk indicators found for {category}.",
            "key_findings": []
        }

    risk_score = len(filtered)

    if risk_score >= 15:
        level = "HIGH"
    elif risk_score >= 7:
        level = "MEDIUM"
    else:
        level = "LOW"

    findings = (
        filtered["Extracted_Text"]
        .dropna()
        .astype(str)
        .head(5)
        .tolist()
    )

    return {
        "risk_level": level,
        "summary": f"{risk_score} ESG risk indicators detected for {category}.",
        "key_findings": findings
    }
