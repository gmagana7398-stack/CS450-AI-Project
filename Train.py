"""
train.py
--------
Trains two classifiers to predict burnout risk:
  1. Logistic Regression  — interpretable baseline, extends our prior GD work
  2. Decision Tree        — captures non-linear feature interactions

Both are evaluated with Stratified 5-Fold cross-validation on the training
set, then the better model is evaluated on the held-out test set.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
import joblib

from Preprocess import load_and_preprocess

# ─────────────────────────────────────────────
# 1. LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────

print("=" * 55)
print("  STEP 1 — Loading & preprocessing data")
print("=" * 55)

X_train, X_test, y_train, y_test, preprocessor, feature_names = load_and_preprocess()

# ─────────────────────────────────────────────
# 2. DEFINE MODELS
# ─────────────────────────────────────────────
# Logistic Regression:
#   - solver="saga" uses stochastic average gradient descent — the closest
#     sklearn equivalent to the gradient descent loop we wrote for GPA/SAT.
#     It minimizes binary cross-entropy loss via iterative weight updates,
#     just like our prior GD code but with convergence optimizations.
#   - C is the inverse of regularization strength (smaller C = stronger
#     regularization). C=1.0 is the default neutral starting point.
#   - class_weight="balanced" automatically upweights the minority class
#     (at-risk, ~21%) during training to counteract class imbalance.
#
# Decision Tree:
#   - max_depth=6 limits tree depth to prevent overfitting on 1200 samples.
#   - min_samples_leaf=15 stops splitting when a node has fewer than 15
#     samples — avoids extremely narrow leaves that memorize training noise.
#   - class_weight="balanced" mirrors the logistic regression setting so
#     both models are evaluated on a fair footing.

models = {
    "Logistic Regression": LogisticRegression(
        solver="saga",
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=42,
    ),
}

# ─────────────────────────────────────────────
# 3. STRATIFIED 5-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────
# We evaluate on F1 (balances precision/recall for imbalanced classes)
# and ROC-AUC (measures discrimination ability across all thresholds).
# Stratified folds preserve the ~79/21 class split in every fold.

print("\n" + "=" * 55)
print("  STEP 2 — Stratified 5-Fold cross-validation")
print("=" * 55)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    scores = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=["f1", "roc_auc"],
        return_train_score=True,
    )
    cv_results[name] = scores

    print(f"\n  {name}")
    print(f"    CV F1       — mean: {scores['test_f1'].mean():.4f}  "
          f"std: ±{scores['test_f1'].std():.4f}")
    print(f"    CV ROC-AUC  — mean: {scores['test_roc_auc'].mean():.4f}  "
          f"std: ±{scores['test_roc_auc'].std():.4f}")
    print(f"    Train F1    — mean: {scores['train_f1'].mean():.4f}  "
          f"(gap vs CV F1: {scores['train_f1'].mean() - scores['test_f1'].mean():.4f})")

# ─────────────────────────────────────────────
# 4. SELECT BEST MODEL
# ─────────────────────────────────────────────
# Primary metric: CV F1 (most meaningful for imbalanced classification)
# Tiebreaker: CV ROC-AUC

best_name = max(
    cv_results,
    key=lambda n: (
        cv_results[n]["test_f1"].mean(),
        cv_results[n]["test_roc_auc"].mean(),
    )
)
best_model = models[best_name]

print(f"\n{'=' * 55}")
print(f"  Best model (by CV F1): {best_name}")
print(f"{'=' * 55}")

# ─────────────────────────────────────────────
# 5. FINAL FIT & TEST SET EVALUATION
# ─────────────────────────────────────────────
# Fit the winning model on the FULL training set, then evaluate once
# on the held-out test set. This is the only time test data is touched.

best_model.fit(X_train, y_train)
y_pred      = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

test_f1      = f1_score(y_test, y_pred)
test_roc_auc = roc_auc_score(y_test, y_pred_prob)
cm           = confusion_matrix(y_test, y_pred)

print(f"\n  Test set results ({best_name})")
print(f"    F1:       {test_f1:.4f}")
print(f"    ROC-AUC:  {test_roc_auc:.4f}")

print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
print(f"                 Pred: Not at risk   Pred: At Risk")
print(f"  Actual: Not at risk       {cm[0,0]:>4}            {cm[0,1]:>4}")
print(f"  Actual: At risk           {cm[1,0]:>4}            {cm[1,1]:>4}")

print(f"\n  Classification report:")
print(classification_report(y_test, y_pred, target_names=["Not at risk", "At risk"]))

# ─────────────────────────────────────────────
# 6. LOGISTIC REGRESSION — TOP FEATURE WEIGHTS
# ─────────────────────────────────────────────
# If logistic regression won (or even if it didn't), print the top
# drivers so we can sanity-check the model makes domain sense before
# wiring up the LLM explainer in a later step.

lr_model = models["Logistic Regression"]
if not hasattr(lr_model, "coef_"):
    lr_model.fit(X_train, y_train)   # fit if it wasn't selected as best

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": lr_model.coef_[0],
}).sort_values("coefficient", key=abs, ascending=False)

print("\n  Top 10 logistic regression coefficients (by magnitude):")
print(f"  {'Feature':<40} {'Coefficient':>12}")
print(f"  {'-'*40} {'-'*12}")
for _, row in coef_df.head(10).iterrows():
    direction = "↑ risk" if row["coefficient"] > 0 else "↓ risk"
    print(f"  {row['feature']:<40} {row['coefficient']:>+.4f}  {direction}")

# ─────────────────────────────────────────────
# 7. SAVE ARTEFACTS
# ─────────────────────────────────────────────

joblib.dump(best_model,   "best_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
np.save("feature_names.npy", np.array(feature_names))

print(f"\n  Saved: best_model.pkl, preprocessor.pkl, feature_names.npy")
print(f"  Training complete.")