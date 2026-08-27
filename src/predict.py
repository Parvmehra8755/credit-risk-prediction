"""Shared scoring logic.

This is the single implementation of "score one applicant". Both the Streamlit UI
(`app.py`) and the command-line demo (`demo_tool.py`) call into here, so the
decision logic lives in exactly one place.
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from config import (
    DECISION_THRESHOLD,
    METADATA_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    SCALER_PATH,
)

# Human-readable labels for the model's feature names, used by both UIs.
FEATURE_LABELS = {
    "age": "Age",
    "income": "Annual income",
    "credit_score": "Credit score",
    "loan_amount": "Loan amount",
    "loan_term": "Loan term",
    "employment_years": "Years employed",
    "dependents": "Dependents",
    "dti": "Debt-to-income ratio",
}


class ModelNotTrainedError(RuntimeError):
    """Raised when the model artifacts are missing."""


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the fitted model and scaler once per process."""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise ModelNotTrainedError(
            "Trained model artifacts not found. Run `python run_pipeline.py` first."
        )
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    """Return the training metadata written by src/train_model.py."""
    if not METADATA_PATH.exists():
        raise ModelNotTrainedError(
            "Model metadata not found. Run `python run_pipeline.py` first."
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def validate(income: float, credit_score: float, loan_amount: float) -> str | None:
    """Return an error message if the inputs are unusable, else None."""
    if income <= 0:
        return "Income must be greater than zero to compute a debt-to-income ratio."
    if not 300 <= credit_score <= 850:
        return f"Credit score {credit_score:g} is outside the valid range 300-850."
    if loan_amount <= 0:
        return "Loan amount must be greater than zero."
    return None


def score_applicant(
    age: int,
    income: float,
    credit_score: float,
    loan_amount: float,
    loan_term: int,
    existing_debt: float,
    employment_years: int,
    dependents: int,
) -> dict:
    """Score one applicant.

    Returns a dict with `status` of "success" (decision, probability, dti and
    the per-feature log-odds contributions) or "error" (error_message).
    """
    problem = validate(income, credit_score, loan_amount)
    if problem:
        return {"status": "error", "error_message": problem}

    try:
        model, scaler = load_artifacts()
    except ModelNotTrainedError as exc:
        return {"status": "error", "error_message": str(exc)}

    dti = existing_debt / income

    values = {
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "employment_years": employment_years,
        "dependents": dependents,
        "dti": dti,
    }
    # A DataFrame (not a bare array) preserves the feature names sklearn was
    # fitted with, and the column order must match MODEL_FEATURES exactly.
    row = pd.DataFrame([[values[f] for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
    row_scaled = scaler.transform(row)

    probability = float(model.predict_proba(row_scaled)[0][1])
    approved = probability >= DECISION_THRESHOLD

    # Per-feature contribution to the log-odds: coefficient x scaled value.
    # Summed with the intercept these reproduce the logit exactly, which is
    # what makes the explanation faithful rather than decorative.
    contributions = {
        feature: float(coef * scaled)
        for feature, coef, scaled in zip(MODEL_FEATURES, model.coef_[0], row_scaled[0])
    }
    ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "status": "success",
        "decision": "Approved" if approved else "Not Approved",
        "approved": approved,
        "approval_probability": round(probability, 4),
        "dti_ratio": round(dti, 4),
        "decision_threshold": DECISION_THRESHOLD,
        "confidence": "high" if abs(probability - 0.5) > 0.3 else "moderate",
        "contributions": contributions,
        "intercept": float(model.intercept_[0]),
        "top_positive_factors": [
            {"feature": f, "contribution": round(v, 4)} for f, v in ranked[:3] if v > 0
        ],
        "top_negative_factors": [
            {"feature": f, "contribution": round(v, 4)} for f, v in ranked[::-1][:3] if v < 0
        ],
    }


def score_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame of applicants, vectorised.

    Requires the eight raw input columns; `dti` is computed here if absent.
    Returns the input frame with `dti`, `approval_probability` and `decision`
    appended.
    """
    model, scaler = load_artifacts()

    out = df.copy()
    if "dti" not in out.columns:
        out["dti"] = out["existing_debt"] / out["income"]

    missing = [c for c in MODEL_FEATURES if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    scaled = scaler.transform(out[MODEL_FEATURES])
    probabilities = model.predict_proba(scaled)[:, 1]

    out["approval_probability"] = probabilities.round(4)
    out["decision"] = [
        "Approved" if p >= DECISION_THRESHOLD else "Not Approved" for p in probabilities
    ]
    return out


def explain_in_words(result: dict, income: float, credit_score: float) -> str:
    """Build a plain-English explanation of a scoring result.

    Deterministic and derived straight from the model's own coefficients, so
    the wording can never drift from the arithmetic behind the decision.
    """
    if result["status"] != "success":
        return result["error_message"]

    probability = result["approval_probability"]
    verdict = "approved" if result["approved"] else "not approved"
    parts = [
        f"This application would be **{verdict}**, with an approval probability of "
        f"{probability:.1%} ({result['confidence']} confidence)."
    ]

    positives = result["top_positive_factors"]
    negatives = result["top_negative_factors"]

    if positives:
        names = ", ".join(FEATURE_LABELS[f["feature"]].lower() for f in positives)
        parts.append(f"Working in the applicant's favour: {names}.")
    if negatives:
        names = ", ".join(FEATURE_LABELS[f["feature"]].lower() for f in negatives)
        parts.append(f"Counting against them: {names}.")

    parts.append(
        f"Their credit score of {credit_score:g} and debt-to-income ratio of "
        f"{result['dti_ratio']:.3f} are the two figures a reviewer would look at first."
    )
    return " ".join(parts)
