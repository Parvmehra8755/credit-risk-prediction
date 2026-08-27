"""Shared configuration: paths, feature lists and the synthetic-data constants.

Everything downstream (EDA, training, scoring) imports from here so the
column order used at training time is exactly the column order used at
inference time.
"""

from pathlib import Path

# --- Reproducibility -------------------------------------------------------
RANDOM_SEED = 42
N_APPLICANTS = 2000
TEST_SIZE = 0.20

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

DATASET_PATH = DATA_DIR / "credit_risk_synthetic.csv"
MODEL_PATH = MODELS_DIR / "logistic_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
EVAL_REPORT_PATH = OUTPUTS_DIR / "evaluation_report.md"

for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Columns ---------------------------------------------------------------
TARGET = "loan_approved"

# The eight raw fields an applicant is described by. These are exactly the
# eight inputs the UI collects.
RAW_FEATURES = [
    "age",
    "income",
    "credit_score",
    "loan_amount",
    "loan_term",
    "existing_debt",
    "employment_years",
    "dependents",
]

# The eight features the model is actually fitted on. `existing_debt` is
# dropped in favour of the engineered `dti` ratio it feeds into -- keeping both
# would put two near-collinear columns in an interpretable model and muddy the
# coefficients we want to read off.
MODEL_FEATURES = [
    "age",
    "income",
    "credit_score",
    "loan_amount",
    "loan_term",
    "employment_years",
    "dependents",
    "dti",
]

DECISION_THRESHOLD = 0.5
