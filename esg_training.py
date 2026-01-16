import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "esg_extracted_data.csv")

df = pd.read_csv(CSV_PATH)


REQUIRED_COLUMNS = ["company", "category", "extracted_text"]

def init_resources():
    global df

    if df is not None:
        return

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")


def analyze_esg_risk(category: str) -> str:
    init_resources()

    pillar = category.strip().lower()
    filtered = df[df["category"].str.lower() == pillar]

    # No disclosure at all
    if filtered.empty:
        return f"""{category.upper()}:
- Risk Level: RED
- Finding:
  - No ESG disclosure data found
  - CSV column: category
- Impact:
  - Absence of disclosure represents compliance and reporting risk
- Data Evidence:
  - No rows found for category = {category}
"""

    findings = []
    risk_level = "GREEN"

    for col in REQUIRED_COLUMNS:
        missing_count = filtered[col].isnull().sum()
        if missing_count > 0:
            findings.append(f"- {col}: {missing_count} missing values")
            risk_level = "YELLOW"

    if not findings:
        return f"""{category.upper()}:
- Risk Level: GREEN
- Finding:
  - No missing or incomplete ESG disclosures detected
- Impact:
  - No compliance or reporting risk identified
- Data Evidence:
  - All required fields populated
"""

    return f"""{category.upper()}:
- Risk Level: {risk_level}
- Finding:
{chr(10).join(findings)}
- Impact:
  - Missing or incomplete ESG data may indicate compliance or reporting risk
- Data Evidence:
  - Null or empty values detected in listed columns
"""
