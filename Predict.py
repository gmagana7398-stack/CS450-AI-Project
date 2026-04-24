"""
predict.py
----------
Interactive CLI that walks an employee through entering their profile,
runs the burnout risk model, and — if flagged at risk — generates a
plain-language LLM explanation of the driving factors.

Usage:
    python3 predict.py
"""

import os
import sys

from config import OPENAI_API_KEY

from Explainer import run as explain_and_run

# ─────────────────────────────────────────────
# VALID CATEGORICAL OPTIONS
# (must match categories seen during training)
# ─────────────────────────────────────────────

VALID = {
    "job_role": [
        "AI Ethics Officer", "AI Researcher", "Backend Engineer",
        "Cloud Architect", "Data Analyst", "Data Scientist",
        "DevOps Engineer", "Frontend Engineer", "ML Engineer",
        "Product Manager", "Prompt Engineer", "Software Engineer",
    ],
    "education_level": [
        "Bootcamp", "Self-taught", "Bachelor", "Master", "PhD",
    ],
    "industry": [
        "Automotive", "Consulting", "Cybersecurity", "E-commerce",
        "EdTech", "Fintech", "Gaming", "Healthtech", "Media", "SaaS",
    ],
    "remote_work_type": [
        "Fully Remote", "Hybrid", "On-site",
    ],
    "ai_adoption_stage": [
        "Experimenting", "Integrating", "Optimizing", "AI-First",
    ],
    "fear_of_ai_replacement": [
        "Low", "Medium", "High",
    ],
}

# ─────────────────────────────────────────────
# INPUT HELPERS
# ─────────────────────────────────────────────

def prompt_choice(label: str, options: list[str]) -> str:
    """Displays a numbered menu and returns the chosen option string."""
    print(f"\n  {label}:")
    for i, opt in enumerate(options, 1):
        print(f"    {i:>2}. {opt}")
    while True:
        raw = input("     → Enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"     Please enter a number between 1 and {len(options)}.")


def prompt_int(label: str, lo: int, hi: int) -> int:
    """Prompts for an integer in [lo, hi]."""
    while True:
        raw = input(f"\n  {label} ({lo}–{hi}): ").strip()
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        print(f"     Please enter a whole number between {lo} and {hi}.")


def prompt_float(label: str, lo: float, hi: float) -> float:
    """Prompts for a float in [lo, hi]."""
    while True:
        raw = input(f"\n  {label} ({lo}–{hi}): ").strip()
        try:
            val = float(raw)
            if lo <= val <= hi:
                return round(val, 2)
        except ValueError:
            pass
        print(f"     Please enter a number between {lo} and {hi}.")


# ─────────────────────────────────────────────
# MAIN INTERVIEW FLOW
# ─────────────────────────────────────────────

def collect_employee_data() -> dict:
    """
    Walks the employee through each field interactively.
    Returns a dict ready to pass to predict_burnout_risk().
    """
    print("\n" + "=" * 55)
    print("  EMPLOYEE BURNOUT RISK SELF-ASSESSMENT")
    print("=" * 55)
    print("""
  This tool uses a trained machine learning model to assess
  your personal burnout risk based on your current work
  situation. All inputs stay local — nothing is stored.

  Answer each question as accurately as you can.
  Press Ctrl+C at any time to exit.
""")

    data = {}

    # ── Role & background ──────────────────────────────
    print("  ── Role & background ─────────────────────────────")

    data["job_role"] = prompt_choice(
        "Job role", VALID["job_role"]
    )
    data["years_experience"] = prompt_int(
        "Years of professional experience", 0, 50
    )
    data["education_level"] = prompt_choice(
        "Highest education level", VALID["education_level"]
    )
    data["industry"] = prompt_choice(
        "Industry", VALID["industry"]
    )

    # ── Work setup ─────────────────────────────────────
    print("\n  ── Work setup ────────────────────────────────────")

    data["remote_work_type"] = prompt_choice(
        "Work arrangement", VALID["remote_work_type"]
    )
    data["team_size"] = prompt_int(
        "Number of people in your immediate team", 1, 200
    )
    data["salary_usd_k"] = prompt_int(
        "Annual salary in USD thousands (e.g. 95 for $95,000)", 10, 500
    )

    # ── AI interaction ─────────────────────────────────
    print("\n  ── AI interaction ────────────────────────────────")

    data["hours_with_ai_assistance_daily"] = prompt_float(
        "Hours per day you actively use AI tools", 0.0, 12.0
    )
    data["ai_replaces_my_tasks_pct"] = prompt_int(
        "Percentage of your tasks that AI now handles instead of you (0–100)", 0, 100
    )
    data["ai_adoption_stage"] = prompt_choice(
        "How would you describe your personal AI adoption stage",
        VALID["ai_adoption_stage"]
    )
    data["weekly_ai_upskilling_hrs"] = prompt_float(
        "Hours per week you spend learning or upskilling in AI", 0.0, 30.0
    )

    # ── Wellbeing ──────────────────────────────────────
    print("\n  ── Wellbeing ─────────────────────────────────────")

    data["productivity_score"] = prompt_int(
        "How productive do you feel on a typical day (0 = not at all, 100 = extremely)", 0, 100
    )
    data["job_satisfaction_1_5"] = prompt_float(
        "Overall job satisfaction (1.0 = very dissatisfied, 5.0 = very satisfied)", 1.0, 5.0
    )
    data["fear_of_ai_replacement"] = prompt_choice(
        "How concerned are you that AI could replace your role",
        VALID["fear_of_ai_replacement"]
    )

    return data


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    api_key = OPENAI_API_KEY
    if not api_key:
        print("\n  ⚠  OPENAI_API_KEY not set.")
        print("     Export it before running:")
        print("       export OPENAI_API_KEY=sk-...\n")
        sys.exit(1)

    try:
        employee = collect_employee_data()

        print("\n  Analysing your profile", end="", flush=True)
        for _ in range(3):
            import time; time.sleep(0.4); print(".", end="", flush=True)
        print()

        explain_and_run(employee, api_key=api_key)

    except KeyboardInterrupt:
        print("\n\n  Assessment cancelled. Take care!\n")
        sys.exit(0)