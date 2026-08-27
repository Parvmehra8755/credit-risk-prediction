"""Exercises the scoring logic from the command line.

Calls exactly the function the Streamlit UI calls (`src/predict.py`), so it is
the fastest way to confirm the Model Layer is wired up without opening a browser.

    python demo_tool.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from predict import (  # noqa: E402
    ModelNotTrainedError,
    load_metadata,
    score_applicant,
)

APPLICANTS = [
    (
        "Rahul -- the deck's example (slide 13)",
        dict(age=35, income=60_000, credit_score=700, loan_amount=150_000,
             loan_term=36, existing_debt=5_000, employment_years=8, dependents=1),
    ),
    (
        "Strong applicant -- high score, low debt",
        dict(age=45, income=90_000, credit_score=790, loan_amount=120_000,
             loan_term=24, existing_debt=2_000, employment_years=20, dependents=0),
    ),
    (
        "Weak applicant -- low score, heavy debt",
        dict(age=26, income=28_000, credit_score=520, loan_amount=350_000,
             loan_term=60, existing_debt=18_000, employment_years=1, dependents=3),
    ),
    (
        "Borderline applicant -- average on every field",
        dict(age=40, income=50_000, credit_score=650, loan_amount=200_000,
             loan_term=36, existing_debt=10_000, employment_years=15, dependents=2),
    ),
    (
        "Invalid input -- credit score out of range",
        dict(age=30, income=50_000, credit_score=900, loan_amount=100_000,
             loan_term=36, existing_debt=5_000, employment_years=5, dependents=1),
    ),
]


def main() -> None:
    try:
        meta = load_metadata()
    except ModelNotTrainedError as exc:
        print(exc)
        return

    print("=" * 78)
    print("MODEL SUMMARY")
    print("=" * 78)
    print(f"  {meta['model']}")
    print(f"  Trained on {meta['n_train']:,} applicants, tested on {meta['n_test']:,}")
    print(f"  Accuracy {meta['metrics']['accuracy']:.4f} | "
          f"ROC-AUC {meta['metrics']['roc_auc']:.4f}")

    for label, fields in APPLICANTS:
        print("\n" + "=" * 78)
        print(label)
        print("-" * 78)
        result = score_applicant(**fields)

        if result["status"] == "error":
            print(f"  ERROR: {result['error_message']}")
            continue

        print(f"  Decision    : {result['decision']}")
        print(f"  Probability : {result['approval_probability']:.4f} "
              f"({result['confidence']} confidence)")
        print(f"  DTI ratio   : {result['dti_ratio']:.4f}")
        if result["top_positive_factors"]:
            print("  Pushing toward approval :")
            for f in result["top_positive_factors"]:
                print(f"      {f['feature']:<18} {f['contribution']:+.3f}")
        if result["top_negative_factors"]:
            print("  Pushing toward rejection:")
            for f in result["top_negative_factors"]:
                print(f"      {f['feature']:<18} {f['contribution']:+.3f}")

    print("\n" + "=" * 78)
    print("All decisions above come from a model trained on synthetic data -- "
          "a demonstration, not real lending advice.")


if __name__ == "__main__":
    main()
