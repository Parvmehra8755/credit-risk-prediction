"""Model Layer -- preprocessing, training, evaluation, interpretation
(deck slides 7-9).

  * 80/20 train/test split, stratified on `loan_approved`
  * StandardScaler fitted on the training set only
  * LogisticRegression(max_iter=1000)
  * Accuracy, ROC-AUC, confusion matrix, precision/recall per class
  * Coefficient interpretation, checked against the sign the synthetic rule implies

Artifacts: models/logistic_model.joblib, models/scaler.joblib,
models/model_metadata.json, outputs/evaluation_report.md, outputs/figures/*.png

Run:
    python src/train_model.py
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATASET_PATH,
    DECISION_THRESHOLD,
    EVAL_REPORT_PATH,
    FIGURES_DIR,
    METADATA_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    RANDOM_SEED,
    SCALER_PATH,
    TARGET,
    TEST_SIZE,
)

# Sign each coefficient should carry if the model recovered the generating rule.
# `age` and `loan_term` are not in the rule, so no expectation is asserted.
EXPECTED_SIGNS = {
    "credit_score": "+",
    "income": "+",
    "employment_years": "+",
    "loan_amount": "-",
    "dependents": "-",
    "dti": "-",
}


def load_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found -- run `python src/generate_data.py` first."
        )
    return pd.read_csv(DATASET_PATH)


def split_and_scale(df: pd.DataFrame):
    """80/20 stratified split, then scale using training-set statistics only."""
    X = df[MODEL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit on train only
    X_test_scaled = scaler.transform(X_test)        # test is only transformed

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def train(X_train_scaled: np.ndarray, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train_scaled, y_train)
    return model


def plot_confusion(y_test, y_pred) -> np.ndarray:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(
        cm, display_labels=["Not approved", "Approved"]
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion matrix -- test set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    return cm


def plot_roc(y_test, y_proba, auc: float) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, color="#028090", linewidth=2.5,
            label=f"Logistic regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#B85042", linewidth=1.2,
            label="Random (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve -- test set", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=140)
    plt.close(fig)


def plot_coefficients(coef_df: pd.DataFrame) -> None:
    ordered = coef_df.sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#B85042" if v < 0 else "#028090" for v in ordered["coefficient"]]
    ax.barh(ordered["feature"], ordered["coefficient"], color=colors)
    ax.axvline(0, color="#36454F", linewidth=1)
    ax.set_title("Logistic regression coefficients (standardised features)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Coefficient -- log-odds change per 1 SD increase")
    span = max(abs(ordered["coefficient"].min()), abs(ordered["coefficient"].max()))
    for i, v in enumerate(ordered["coefficient"]):
        ax.text(v + (0.05 if v >= 0 else -0.05) * span, i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlim(-span * 1.35, span * 1.35)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "coefficients.png", dpi=140)
    plt.close(fig)


def build_coefficient_table(model: LogisticRegression) -> pd.DataFrame:
    coefs = model.coef_[0]
    df = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
            "abs_coefficient": np.abs(coefs),
        }
    )
    df["direction"] = np.where(df["coefficient"] >= 0, "toward approval", "toward rejection")
    df["expected_sign"] = df["feature"].map(EXPECTED_SIGNS).fillna("n/a (control)")
    actual = np.where(df["coefficient"] >= 0, "+", "-")
    df["sign_matches"] = [
        "n/a" if exp == "n/a (control)" else ("yes" if exp == act else "NO")
        for exp, act in zip(df["expected_sign"], actual)
    ]
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def write_report(metrics: dict, cm: np.ndarray, coef_df: pd.DataFrame,
                 report_text: str, intercept: float) -> None:
    tn, fp, fn, tp = cm.ravel()
    lines = [
        "# Model Evaluation Report",
        "",
        "Logistic regression on a synthetic credit-risk dataset. "
        "All metrics below are computed on the held-out 20% test set.",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| ROC-AUC | {metrics['roc_auc']:.4f} |",
        f"| Precision (Not approved) | {metrics['precision_not_approved']:.4f} |",
        f"| Recall (Not approved) | {metrics['recall_not_approved']:.4f} |",
        f"| Precision (Approved) | {metrics['precision_approved']:.4f} |",
        f"| Recall (Approved) | {metrics['recall_approved']:.4f} |",
        "",
        "## Confusion matrix",
        "",
        "| | Predicted: Not approved | Predicted: Approved |",
        "| --- | ---: | ---: |",
        f"| **Actual: Not approved** | {tn} (TN) | {fp} (FP -- false approval) |",
        f"| **Actual: Approved** | {fn} (FN -- false rejection) | {tp} (TP) |",
        "",
        f"False approvals ({fp}) are the costlier error for a lender: the model let through "
        f"an applicant the generating rule would have rejected. False rejections ({fn}) cost "
        "business rather than money lost.",
        "",
        "## Per-class report",
        "",
        "```",
        report_text.rstrip(),
        "```",
        "",
        "## Coefficient interpretation",
        "",
        "Features are standardised, so each coefficient is the change in log-odds of approval "
        "per one standard-deviation increase in that feature. Ranked by absolute magnitude:",
        "",
        "| Feature | Coefficient | Odds ratio | Pushes | Expected sign | Matches? |",
        "| --- | ---: | ---: | --- | :---: | :---: |",
    ]
    for _, r in coef_df.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['coefficient']:+.4f} | {r['odds_ratio']:.4f} | "
            f"{r['direction']} | {r['expected_sign']} | {r['sign_matches']} |"
        )

    mismatches = coef_df[coef_df["sign_matches"] == "NO"]
    lines += [
        "",
        f"Intercept: `{intercept:+.4f}`",
        "",
    ]
    if mismatches.empty:
        lines.append(
            "Every coefficient with a directional expectation came out with the sign the "
            "synthetic scoring rule implies: `credit_score`, `income` and `employment_years` "
            "push toward approval, while `dti`, `loan_amount` and `dependents` push toward "
            "rejection. The model recovered the data-generating process."
        )
    else:
        names = ", ".join(f"`{f}`" for f in mismatches["feature"])
        lines.append(
            f"**Sign mismatch on {names}.** The coefficient runs opposite to the direction "
            "the generating rule implies. Worth reporting as a modelling insight -- most "
            "often it reflects correlation between predictors or a weak effect swamped by "
            "noise -- rather than quietly correcting it."
        )
    lines += [
        "",
        "`age` and `loan_term` are controls: they were left out of the synthetic scoring "
        "rule, so near-zero coefficients on them are the correct result and a useful sanity "
        "check that the model is not inventing signal.",
        "",
        "## Figures",
        "",
        "- `figures/confusion_matrix.png`",
        "- `figures/roc_curve.png`",
        "- `figures/coefficients.png`",
        "",
    ]
    EVAL_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_data()
    (X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled, scaler) = split_and_scale(df)

    print(f"Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
    print(f"Approval rate -- train {y_train.mean():.1%}, test {y_test.mean():.1%} "
          "(stratified, so these match)\n")

    model = train(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision_not_approved": float(precision_score(y_test, y_pred, pos_label=0)),
        "recall_not_approved": float(recall_score(y_test, y_pred, pos_label=0)),
        "precision_approved": float(precision_score(y_test, y_pred, pos_label=1)),
        "recall_approved": float(recall_score(y_test, y_pred, pos_label=1)),
        "train_accuracy": float(accuracy_score(y_train, model.predict(X_train_scaled))),
    }

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"(train accuracy {metrics['train_accuracy']:.4f} -- gap to test indicates fit)\n")

    report_text = classification_report(
        y_test, y_pred, target_names=["Not approved (0)", "Approved (1)"], digits=4
    )
    print(report_text)

    cm = plot_confusion(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion matrix: TN={tn}  FP={fp} (false approvals)  "
          f"FN={fn} (false rejections)  TP={tp}\n")

    plot_roc(y_test, y_proba, metrics["roc_auc"])

    coef_df = build_coefficient_table(model)
    plot_coefficients(coef_df)

    print("Coefficients (standardised features, strongest first):")
    print(coef_df[["feature", "coefficient", "odds_ratio", "direction",
                   "expected_sign", "sign_matches"]].to_string(index=False))
    print(f"\nIntercept: {model.intercept_[0]:+.4f}")

    mismatches = coef_df[coef_df["sign_matches"] == "NO"]
    if mismatches.empty:
        print("All directional expectations hold -- the model recovered the generating rule.")
    else:
        print("Sign mismatch on: " + ", ".join(mismatches["feature"]) +
              " -- report as a modelling insight, see outputs/evaluation_report.md")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "model": "LogisticRegression(max_iter=1000)",
                "features": MODEL_FEATURES,
                "target": TARGET,
                "decision_threshold": DECISION_THRESHOLD,
                "random_seed": RANDOM_SEED,
                "test_size": TEST_SIZE,
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "metrics": metrics,
                "intercept": float(model.intercept_[0]),
                "coefficients": {
                    f: float(c) for f, c in zip(MODEL_FEATURES, model.coef_[0])
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(metrics, cm, coef_df, report_text, float(model.intercept_[0]))

    print(f"\nSaved model    -> {MODEL_PATH}")
    print(f"Saved scaler   -> {SCALER_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")
    print(f"Saved report   -> {EVAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
