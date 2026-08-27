"""Data Layer -- synthetic applicant dataset generator (deck slide 6).

Simulates ~2,000 loan applicants. No real or public data is used: the target
is derived from a weighted latent score passed through a sigmoid, so we know
the ground-truth relationship the model is supposed to recover.

Run:
    python src/generate_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DATASET_PATH,
    N_APPLICANTS,
    RANDOM_SEED,
    TARGET,
)

# --- Weights of the latent scoring rule (slide 6) --------------------------
# Chosen so the approval rate lands inside the 55/45 - 70/30 band the deck asks
# for, rather than an extreme skew.
W_CREDIT_SCORE = 0.02       # per point above 650
W_INCOME = 0.00003          # per rupee above 50k
W_LOAN_AMOUNT = -0.000004   # per rupee above 200k
W_DTI = -3.0
W_EMPLOYMENT = 0.05
W_DEPENDENTS = -0.10
INTERCEPT = 0.60
NOISE_SD = 0.50


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_applicants(n: int = N_APPLICANTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Simulate `n` applicants and label them with the latent-score rule."""
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 66, size=n)
    income = np.clip(rng.normal(50_000, 15_000, size=n), 15_000, None)
    credit_score = np.clip(rng.normal(650, 80, size=n), 300, 850)
    loan_amount = np.clip(rng.normal(200_000, 80_000, size=n), 20_000, None)
    loan_term = rng.choice([12, 24, 36, 48, 60], size=n)
    existing_debt = np.clip(rng.normal(10_000, 8_000, size=n), 0, None)
    employment_years = rng.integers(0, 31, size=n)
    dependents = rng.integers(0, 5, size=n)

    # Engineered feature: debt-to-income ratio.
    dti = existing_debt / income

    score = (
        INTERCEPT
        + W_CREDIT_SCORE * (credit_score - 650)
        + W_INCOME * (income - 50_000)
        + W_LOAN_AMOUNT * (loan_amount - 200_000)
        + W_DTI * dti
        + W_EMPLOYMENT * employment_years
        + W_DEPENDENTS * dependents
        + rng.normal(0, NOISE_SD, size=n)
    )
    prob = sigmoid(score)
    loan_approved = (prob > 0.5).astype(int)

    # `age` and `loan_term` are deliberately left out of the scoring rule: they
    # act as controls, and their coefficients should come out near zero.
    return pd.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "credit_score": credit_score.round(1),
            "loan_amount": loan_amount.round(2),
            "loan_term": loan_term,
            "existing_debt": existing_debt.round(2),
            "employment_years": employment_years,
            "dependents": dependents,
            "dti": dti.round(4),
            TARGET: loan_approved,
        }
    )


def main() -> None:
    df = generate_applicants()
    df.to_csv(DATASET_PATH, index=False)

    approved = int(df[TARGET].sum())
    rate = approved / len(df)
    print(f"Wrote {len(df):,} synthetic applicants -> {DATASET_PATH}")
    print(f"Class balance: {approved:,} approved / {len(df) - approved:,} not approved "
          f"({rate:.1%} / {1 - rate:.1%})")
    if not 0.50 <= rate <= 0.75:
        print("WARNING: approval rate is outside the 55/45-70/30 band the design targets.")
    print("\nFirst rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
