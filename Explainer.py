"""
explainer.py
------------
Loads the trained model artifacts, accepts a single employee's raw data,
produces a burnout risk prediction, and — if the employee is at risk —
calls the OpenAI API to generate a plain-language explanation of the
driving factors using their individual feature contributions.
"""

import os
import numpy as np
import pandas as pd
import joblib
from openai import OpenAI
from config import OPENAI_API_KEY

# ─────────────────────────────────────────────
# 1. LOAD SAVED ARTEFACTS
# ─────────────────────────────────────────────

MODEL_PATH         = "best_model.pkl"
PREPROCESSOR_PATH  = "preprocessor.pkl"
FEATURE_NAMES_PATH = "feature_names.npy"

model         = joblib.load(MODEL_PATH)
preprocessor  = joblib.load(PREPROCESSOR_PATH)
feature_names = np.load(FEATURE_NAMES_PATH, allow_pickle=True).tolist()

# ─────────────────────────────────────────────
# 2. FEATURE CONTRIBUTION HELPER
# ─────────────────────────────────────────────
# For logistic regression, the contribution of each feature to the
# log-odds is: coefficient × (scaled feature value).
# Sorting by magnitude tells us which features pushed the prediction
# most strongly toward or away from "at risk".

def get_top_contributions(encoded_row: np.ndarray, n: int = 6) -> list[dict]:
    """
    Returns the top-n features by |coefficient × value|, signed so that
    positive = increases burnout risk, negative = decreases burnout risk.
    """
    coefficients  = model.coef_[0]                       # shape: (n_features,)
    contributions = coefficients * encoded_row            # element-wise product
    indices       = np.argsort(np.abs(contributions))[::-1][:n]

    return [
        {
            "feature":      feature_names[i],
            "contribution": float(contributions[i]),
            "direction":    "increases risk" if contributions[i] > 0 else "decreases risk",
        }
        for i in indices
    ]

# ─────────────────────────────────────────────
# 3. HUMAN-READABLE FEATURE LABELS
# ─────────────────────────────────────────────
# Maps raw column names to plain English for the LLM prompt so the
# explanation reads naturally rather than echoing snake_case identifiers.

FEATURE_LABELS = {
    "years_experience":              "years of experience",
    "team_size":                     "team size",
    "salary_usd_k":                  "annual salary (USD thousands)",
    "hours_with_ai_assistance_daily":"hours per day using AI tools",
    "ai_replaces_my_tasks_pct":      "percentage of tasks replaced by AI",
    "weekly_ai_upskilling_hrs":      "hours per week spent upskilling in AI",
    "productivity_score":            "productivity score (0–100)",
    "job_satisfaction_1_5":          "job satisfaction (1–5 scale)",
    "education_level":               "education level",
    "ai_adoption_stage":             "AI adoption stage",
    "fear_of_ai_replacement":        "fear of AI replacement (Low/Medium/High)",
    "job_role":                      "job role",
    "industry":                      "industry",
    "remote_work_type":              "work arrangement",
}

def friendly_label(raw_name: str) -> str:
    """Strips OHE suffixes (e.g. 'job_role_Data Analyst') and looks up label."""
    for key in FEATURE_LABELS:
        if raw_name.startswith(key):
            return FEATURE_LABELS[key]
    return raw_name.replace("_", " ")

# ─────────────────────────────────────────────
# 4. CORE PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict_burnout_risk(employee: dict) -> dict:
    """
    Accepts a dict of raw employee features (matching original CSV columns,
    excluding employee_id, burnout_score, and attrition_risk).

    Returns a result dict with:
        at_risk       : bool
        probability   : float  (probability of being at risk)
        contributions : list[dict]  (top feature drivers)
    """
    df_input   = pd.DataFrame([employee])
    encoded    = preprocessor.transform(df_input)
    prob       = model.predict_proba(encoded)[0][1]    # P(at_risk=1)
    prediction = int(prob >= 0.5)

    contributions = get_top_contributions(encoded[0]) if prediction == 1 else []

    return {
        "at_risk":       bool(prediction),
        "probability":   round(prob, 4),
        "contributions": contributions,
    }

# ─────────────────────────────────────────────
# 5. LLM EXPLANATION
# ─────────────────────────────────────────────

def build_prompt(employee: dict, result: dict) -> str:
    """Constructs the user message sent to the OpenAI API."""

    contrib_lines = "\n".join([
        f"  - {friendly_label(c['feature'])}: {c['direction']} "
        f"(strength: {abs(c['contribution']):.3f})"
        for c in result["contributions"]
    ])

    profile_lines = "\n".join([
        f"  - {FEATURE_LABELS.get(k, k.replace('_',' '))}: {v}"
        for k, v in employee.items()
    ])

    return f"""
You are an occupational wellbeing advisor reviewing an employee's work profile.
Our predictive model has flagged this employee as at risk of burnout
with {result['probability']*100:.1f}% confidence.

Employee profile:
{profile_lines}

The model's top factors driving this prediction (positive = increases risk):
{contrib_lines}

Please write a 3–4 sentence explanation for the employee in plain, empathetic
language that:
  1. Briefly acknowledges the specific factors most responsible for the risk
  2. Explains why those factors tend to contribute to burnout
  3. Suggests one concrete, actionable step they could take this week

Do not use technical jargon or mention model coefficients. Write directly to
the employee using "you / your".
""".strip()


def explain_burnout_risk(employee: dict, result: dict, api_key: str = None) -> str:
    """
    Calls the OpenAI API and returns a plain-language burnout explanation.
    If the employee is not at risk, returns a reassurance message instead.
    """
    if not result["at_risk"]:
        return (
            f"Based on your current profile, you are not flagged as at risk of burnout "
            f"(risk score: {result['probability']*100:.1f}%). Keep up the healthy patterns!"
        )

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a compassionate occupational health advisor. "
                    "Your explanations are warm, specific, jargon-free, and actionable."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(employee, result),
            },
        ],
        temperature=0.5,      # low temperature for consistent, grounded explanations
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()

# ─────────────────────────────────────────────
# 6. FULL PIPELINE — PREDICT + EXPLAIN
# ─────────────────────────────────────────────

def run(employee: dict, api_key: str = None) -> None:
    """End-to-end: predict burnout risk and print an LLM explanation."""

    print("\n" + "=" * 55)
    print("  BURNOUT RISK ASSESSMENT")
    print("=" * 55)

    result = predict_burnout_risk(employee)

    status = "⚠  AT RISK" if result["at_risk"] else "✓  NOT AT RISK"
    print(f"\n  Status:      {status}")
    print(f"  Risk score:  {result['probability']*100:.1f}%")

    if result["at_risk"]:
        print(f"\n  Top contributing factors:")
        for c in result["contributions"]:
            arrow = "↑" if c["contribution"] > 0 else "↓"
            print(f"    {arrow} {friendly_label(c['feature']):<40} "
                  f"({c['direction']}, strength {abs(c['contribution']):.3f})")

    print(f"\n  Explanation:")
    print(f"  {'-' * 51}")
    explanation = explain_burnout_risk(employee, result, api_key=api_key)
    # Wrap explanation text for readability
    for line in explanation.split("\n"):
        print(f"  {line}")
    print(f"  {'-' * 51}\n")

# ─────────────────────────────────────────────
# 7. EXAMPLE — STANDALONE ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Sample employee — swap in any values matching the dataset's columns.
    # Note: burnout_score, attrition_risk, and employee_id are excluded
    # since they are not available at inference time.
    sample_employee = {
        "job_role":                       "Data Scientist",
        "years_experience":               4,
        "education_level":                "Master",
        "industry":                       "SaaS",
        "remote_work_type":               "Hybrid",
        "team_size":                      12,
        "salary_usd_k":                   110,
        "hours_with_ai_assistance_daily": 5.5,
        "ai_replaces_my_tasks_pct":       72,
        "ai_adoption_stage":              "Optimizing",
        "weekly_ai_upskilling_hrs":       1.5,
        "productivity_score":             61,
        "job_satisfaction_1_5":           2.4,
        "fear_of_ai_replacement":         "High",
    }

    # Set your key here or export OPENAI_API_KEY in your environment
    run(sample_employee, api_key=OPENAI_API_KEY)