"""
preprocess.py
-------------
Loads the AI worker burnout dataset, engineers the binary target label,
encodes categorical features, scales numerical features, and produces an
80/20 stratified train/test split ready for model training.

Can be run standalone for a data audit, or imported by other modules
via load_and_preprocess().
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

DATA_PATH = "ai_worker_burnout_attrition_2026.csv"
BURNOUT_THRESHOLD = 60   # mean + 1 std dev — see analysis notes

NOMINAL_COLS = ["job_role", "industry", "remote_work_type"]

ORDINAL_COLS = {
    "education_level":        ["Bootcamp", "Self-taught", "Bachelor", "Master", "PhD"],
    "ai_adoption_stage":      ["Experimenting", "Integrating", "Optimizing", "AI-First"],
    "fear_of_ai_replacement": ["Low", "Medium", "High"],
}

DROP_COLS = ["burnout_score", "attrition_risk", "at_risk"]

# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def load_and_preprocess(data_path=DATA_PATH, verbose=True):
    """
    Loads the dataset, engineers the target, builds a ColumnTransformer
    preprocessor, and returns an 80/20 stratified train/test split.

    Returns
    -------
    X_train       : np.ndarray        — scaled/encoded training features
    X_test        : np.ndarray        — scaled/encoded test features
    y_train       : pd.Series         — binary training labels
    y_test        : pd.Series         — binary test labels
    preprocessor  : ColumnTransformer — fitted transformer (for inference)
    feature_names : list[str]         — final feature names post-encoding
    """

    # 1. Load & clean
    df = pd.read_csv(data_path)
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
    df = df.drop(columns=["employee_id"])

    # 2. Engineer target label
    df["at_risk"] = (df["burnout_score"] >= BURNOUT_THRESHOLD).astype(int)

    # 3. Split features / target
    X = df.drop(columns=DROP_COLS)
    y = df["at_risk"]

    numerical_cols = [
        c for c in X.columns
        if c not in NOMINAL_COLS and c not in ORDINAL_COLS
    ]

    # 4. Build preprocessing pipeline
    ordinal_transformer = Pipeline(steps=[
        ("encoder", OrdinalEncoder(
            categories=[ORDINAL_COLS[col] for col in ORDINAL_COLS],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    nominal_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    numerical_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numerical", numerical_transformer, numerical_cols),
        ("ordinal",   ordinal_transformer,   list(ORDINAL_COLS.keys())),
        ("nominal",   nominal_transformer,   NOMINAL_COLS),
    ], remainder="drop")

    # 5. Train/test split — fit preprocessor ONLY on training data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)

    # 6. Recover final feature names
    ohe_feature_names = (
        preprocessor
        .named_transformers_["nominal"]
        .named_steps["encoder"]
        .get_feature_names_out(NOMINAL_COLS)
        .tolist()
    )
    feature_names = numerical_cols + list(ORDINAL_COLS.keys()) + ohe_feature_names

    if verbose:
        at_risk_count = y.sum()
        print(f"Dataset loaded:          {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"At-risk (1):             {at_risk_count} ({at_risk_count/len(df)*100:.1f}%)")
        print(f"Not at risk (0):         {len(df)-at_risk_count} ({(len(df)-at_risk_count)/len(df)*100:.1f}%)")
        print(f"Train shape:             {X_train.shape}")
        print(f"Test  shape:             {X_test.shape}")
        print(f"Total encoded features:  {len(feature_names)}")

    return X_train, X_test, y_train, y_test, preprocessor, feature_names


# ─────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    load_and_preprocess(verbose=True)