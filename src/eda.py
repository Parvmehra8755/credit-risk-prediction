"""Data Layer -- exploratory data analysis (deck slide 7).

Produces the three checks the design asks for:
  1. Histograms per feature
  2. Correlation of each predictor with `loan_approved`
  3. Class-balance check (target band: 55/45 to 70/30)

Figures are written to outputs/figures/ and a summary to outputs/eda_summary.md.

Run:
    python src/eda.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DATASET_PATH, FIGURES_DIR, MODEL_FEATURES, OUTPUTS_DIR, RAW_FEATURES, TARGET

sns.set_theme(style="whitegrid")

EDA_SUMMARY_PATH = OUTPUTS_DIR / "eda_summary.md"
NUMERIC_COLS = RAW_FEATURES + ["dti"]


def load_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found -- run `python src/generate_data.py` first."
        )
    return pd.read_csv(DATASET_PATH)


def plot_distributions(df: pd.DataFrame) -> None:
    """Histogram per feature, in one grid."""
    n = len(NUMERIC_COLS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    for ax, col in zip(axes.ravel(), NUMERIC_COLS):
        sns.histplot(df[col], bins=30, kde=True, ax=ax, color="#028090")
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Feature distributions -- synthetic applicants", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_distributions.png", dpi=140)
    plt.close(fig)


def plot_target_correlations(df: pd.DataFrame) -> pd.Series:
    """Correlation of each predictor with the target, sorted by strength."""
    corr = df[NUMERIC_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#B85042" if v < 0 else "#028090" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="#36454F", linewidth=1)
    ax.set_title(f"Correlation with {TARGET}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Pearson correlation")
    for i, v in enumerate(corr.values):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlim(corr.min() - 0.12, corr.max() + 0.12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_correlations.png", dpi=140)
    plt.close(fig)
    return corr


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df[NUMERIC_COLS + [TARGET]].corr(),
        annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        square=True, cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title("Correlation matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=140)
    plt.close(fig)


def plot_class_balance(df: pd.DataFrame) -> tuple[int, int, float]:
    counts = df[TARGET].value_counts().sort_index()
    not_approved, approved = int(counts.get(0, 0)), int(counts.get(1, 0))
    rate = approved / len(df)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Not approved (0)", "Approved (1)"], [not_approved, approved],
                  color=["#B85042", "#028090"], width=0.55)
    for bar, count in zip(bars, [not_approved, approved]):
        ax.text(bar.get_x() + bar.get_width() / 2, count + len(df) * 0.01,
                f"{count:,}\n({count / len(df):.1%})", ha="center", fontweight="bold")
    ax.set_ylim(0, max(not_approved, approved) * 1.20)
    ax.set_title("Class balance", fontsize=14, fontweight="bold")
    ax.set_ylabel("Applicants")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=140)
    plt.close(fig)
    return approved, not_approved, rate


def plot_feature_by_class(df: pd.DataFrame) -> None:
    """Boxplots of the strongest drivers, split by approval outcome."""
    key = ["credit_score", "dti", "income", "employment_years"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, col in zip(axes, key):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax,
                    hue=TARGET, legend=False, palette=["#B85042", "#028090"])
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_xlabel("loan_approved")
    fig.suptitle("Key predictors by approval outcome", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "key_predictors_by_class.png", dpi=140)
    plt.close(fig)


def main() -> None:
    df = load_data()

    print(f"Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}\n")
    print("Descriptive statistics:")
    print(df[NUMERIC_COLS].describe().T.to_string())

    plot_distributions(df)
    corr = plot_target_correlations(df)
    plot_correlation_matrix(df)
    approved, not_approved, rate = plot_class_balance(df)
    plot_feature_by_class(df)

    print("\nCorrelation with loan_approved (strongest first):")
    for name, value in corr.reindex(corr.abs().sort_values(ascending=False).index).items():
        print(f"  {name:<18} {value:+.3f}")

    balanced = 0.50 <= rate <= 0.75
    print(f"\nClass balance: {approved:,} approved ({rate:.1%}) / "
          f"{not_approved:,} not approved ({1 - rate:.1%})")
    print("Balance check:", "PASS -- not extremely skewed" if balanced else "WARNING -- skewed")

    lines = [
        "# EDA Summary",
        "",
        f"- Rows: **{df.shape[0]:,}**, columns: **{df.shape[1]}**",
        f"- Missing values: **{int(df.isna().sum().sum())}**, duplicate rows: "
        f"**{int(df.duplicated().sum())}**",
        f"- Class balance: **{approved:,} approved ({rate:.1%})** vs "
        f"**{not_approved:,} not approved ({1 - rate:.1%})** -- "
        f"{'within' if balanced else 'OUTSIDE'} the 55/45-70/30 design band",
        "",
        "## Correlation with `loan_approved`",
        "",
        "| Feature | Correlation |",
        "| --- | ---: |",
    ]
    for name, value in corr.reindex(corr.abs().sort_values(ascending=False).index).items():
        lines.append(f"| `{name}` | {value:+.3f} |")
    lines += [
        "",
        "`age` and `loan_term` are excluded from the synthetic scoring rule by design, so "
        "their correlations should sit near zero -- they act as controls.",
        "",
        "## Figures",
        "",
        "- `figures/feature_distributions.png`",
        "- `figures/target_correlations.png`",
        "- `figures/correlation_matrix.png`",
        "- `figures/class_balance.png`",
        "- `figures/key_predictors_by_class.png`",
        "",
        f"Model feature set ({len(MODEL_FEATURES)}): "
        + ", ".join(f"`{c}`" for c in MODEL_FEATURES),
        "",
    ]
    EDA_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFigures -> {FIGURES_DIR}")
    print(f"Summary -> {EDA_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
