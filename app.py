"""Credit Risk Loan Approval -- Streamlit UI.

Fill in an applicant, get a decision plus the reasoning behind it. All
scoring goes through `src/predict.py`.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DATASET_PATH, FIGURES_DIR, TARGET  # noqa: E402
from predict import (  # noqa: E402
    FEATURE_LABELS,
    ModelNotTrainedError,
    explain_in_words,
    load_metadata,
    score_applicant,
    score_batch,
)

st.set_page_config(
    page_title="Credit Risk Loan Approval",
    page_icon="🏦",
    layout="wide",
)


# --------------------------------------------------------------------------
# Guard: the model must be trained before anything here works
# --------------------------------------------------------------------------
try:
    METADATA = load_metadata()
except ModelNotTrainedError as exc:
    st.error(str(exc))
    st.code("python run_pipeline.py", language="bash")
    st.stop()


st.title("🏦 Credit Risk Loan Approval")
st.caption(
    "Logistic regression on synthetic applicant data — a demonstration of an "
    "interpretable credit model, not a real lending decision."
)

tab_score, tab_batch, tab_model, tab_data = st.tabs(
    ["Score an applicant", "Batch scoring", "Model performance", "Data & EDA"]
)


# --------------------------------------------------------------------------
# Tab 1 -- score a single applicant
# --------------------------------------------------------------------------
with tab_score:
    st.subheader("Applicant details")

    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 21, 65, 35, step=1)
            income = st.number_input("Annual income", 1_000, 500_000, 60_000, step=1_000)
            credit_score = st.number_input("Credit score", 300, 850, 700, step=5)
        with c2:
            loan_amount = st.number_input("Loan amount", 1_000, 1_000_000, 150_000, step=5_000)
            loan_term = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60], index=2)
            existing_debt = st.number_input("Existing debt", 0, 500_000, 5_000, step=1_000)
        with c3:
            employment_years = st.number_input("Years employed", 0, 30, 8, step=1)
            dependents = st.number_input("Dependents", 0, 4, 1, step=1)
            st.metric("Debt-to-income ratio", f"{existing_debt / max(income, 1):.3f}")

        submitted = st.form_submit_button("Assess application", type="primary")

    if submitted:
        result = score_applicant(
            age=age,
            income=income,
            credit_score=credit_score,
            loan_amount=loan_amount,
            loan_term=loan_term,
            existing_debt=existing_debt,
            employment_years=employment_years,
            dependents=dependents,
        )

        if result["status"] == "error":
            st.error(result["error_message"])
        else:
            st.divider()
            probability = result["approval_probability"]

            left, right = st.columns([1, 2])
            with left:
                if result["approved"]:
                    st.success(f"### ✅ {result['decision']}")
                else:
                    st.error(f"### ❌ {result['decision']}")
                st.metric("Approval probability", f"{probability:.1%}")
                st.metric("Debt-to-income ratio", f"{result['dti_ratio']:.3f}")
                st.caption(
                    f"Decision threshold {result['decision_threshold']:.0%} · "
                    f"{result['confidence']} confidence"
                )
            with right:
                st.progress(probability)
                st.markdown(explain_in_words(result, income, credit_score))

            st.divider()
            st.subheader("Why — contribution of each factor")
            st.caption(
                "Each bar is the feature's contribution to the log-odds of approval "
                "(coefficient × standardised value). Positive pushes toward approval, "
                "negative toward rejection. Together with the intercept "
                f"({result['intercept']:+.3f}) these sum to the model's logit."
            )

            contributions = (
                pd.DataFrame(
                    {
                        "Factor": [FEATURE_LABELS[f] for f in result["contributions"]],
                        "Contribution": list(result["contributions"].values()),
                    }
                )
                .set_index("Factor")
                .sort_values("Contribution")
            )
            st.bar_chart(contributions, horizontal=True, color="#028090")

            with st.expander("Raw model output"):
                st.json(result)


# --------------------------------------------------------------------------
# Tab 2 -- batch scoring
# --------------------------------------------------------------------------
with tab_batch:
    st.subheader("Score many applicants at once")
    st.caption(
        "Upload a CSV with columns: age, income, credit_score, loan_amount, "
        "loan_term, existing_debt, employment_years, dependents. "
        "`dti` is computed for you."
    )

    uploaded = st.file_uploader("CSV file", type="csv")

    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            scored = score_batch(incoming)
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # malformed CSV, bad dtypes, etc.
            st.error(f"Could not score this file: {exc}")
        else:
            approved = int((scored["decision"] == "Approved").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Applicants scored", f"{len(scored):,}")
            m2.metric("Approved", f"{approved:,}")
            m3.metric("Approval rate", f"{approved / len(scored):.1%}")

            st.dataframe(scored, width="stretch")
            st.download_button(
                "Download scored CSV",
                scored.to_csv(index=False).encode("utf-8"),
                file_name="scored_applicants.csv",
                mime="text/csv",
            )
    elif DATASET_PATH.exists():
        st.info(
            "No file yet. You can try it with the project's own dataset — "
            "`data/credit_risk_synthetic.csv`."
        )


# --------------------------------------------------------------------------
# Tab 3 -- model performance
# --------------------------------------------------------------------------
with tab_model:
    st.subheader("Held-out test set performance")
    metrics = METADATA["metrics"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    m2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    m3.metric("Recall (not approved)", f"{metrics['recall_not_approved']:.4f}")
    m4.metric("Precision (not approved)", f"{metrics['precision_not_approved']:.4f}")
    st.caption(
        f"{METADATA['model']} · trained on {METADATA['n_train']:,} applicants, "
        f"tested on {METADATA['n_test']:,} · seed {METADATA['random_seed']}"
    )

    c1, c2 = st.columns(2)
    with c1:
        cm = FIGURES_DIR / "confusion_matrix.png"
        if cm.exists():
            st.image(str(cm), width="stretch")
    with c2:
        roc = FIGURES_DIR / "roc_curve.png"
        if roc.exists():
            st.image(str(roc), width="stretch")

    st.divider()
    st.subheader("Coefficients")
    st.caption(
        "Features are standardised, so each coefficient is the change in log-odds "
        "of approval per one standard-deviation increase. `age` and `loan_term` were "
        "deliberately left out of the data-generating rule — their near-zero "
        "coefficients confirm the model is not inventing signal."
    )

    coefficients = (
        pd.DataFrame(
            {
                "Feature": [FEATURE_LABELS.get(f, f) for f in METADATA["coefficients"]],
                "Coefficient": list(METADATA["coefficients"].values()),
            }
        )
        .set_index("Feature")
        .sort_values("Coefficient")
    )
    st.bar_chart(coefficients, horizontal=True, color="#028090")

    report = PROJECT_ROOT / "outputs" / "evaluation_report.md"
    if report.exists():
        with st.expander("Full evaluation report"):
            st.markdown(report.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Tab 4 -- data and EDA
# --------------------------------------------------------------------------
with tab_data:
    st.subheader("Synthetic dataset")

    if not DATASET_PATH.exists():
        st.warning("Dataset not found. Run `python run_pipeline.py`.")
    else:
        data = pd.read_csv(DATASET_PATH)
        approved = int(data[TARGET].sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("Applicants", f"{len(data):,}")
        m2.metric("Approved", f"{approved:,} ({approved / len(data):.1%})")
        m3.metric("Not approved", f"{len(data) - approved:,} "
                                  f"({1 - approved / len(data):.1%})")

        st.dataframe(data.head(100), width="stretch")
        st.caption("First 100 rows.")

        st.divider()
        st.subheader("Exploratory analysis")
        for name, caption in [
            ("target_correlations.png", "Correlation of each predictor with the target"),
            ("class_balance.png", "Class balance"),
            ("key_predictors_by_class.png", "Key predictors split by outcome"),
            ("feature_distributions.png", "Feature distributions"),
            ("correlation_matrix.png", "Correlation matrix"),
        ]:
            path = FIGURES_DIR / name
            if path.exists():
                st.image(str(path), caption=caption, width="stretch")


st.divider()
st.caption(
    "Trained on fully synthetic data generated from a known scoring rule. "
    "Results do not generalise to real applicants and this is not lending advice."
)
