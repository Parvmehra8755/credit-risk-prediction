# Credit Risk Loan Approval

An end-to-end credit-risk project: **synthetic data → logistic regression → Streamlit UI**.
Fill in a loan applicant's details and get an approval decision along with the reasoning
behind it, straight from an interpretable model.

Built to the specification in `credit_risk_solution_design.pptx`, with one deliberate
deviation: the deck specified a conversational Google ADK agent as the interface, and this
project uses a Streamlit web UI instead.

| Layer | What it does | Files |
| --- | --- | --- |
| **Data** | Simulates ~2,000 applicants, runs EDA, preprocesses | `src/generate_data.py`, `src/eda.py` |
| **Model** | 80/20 split, scaling, logistic regression, evaluation | `src/train_model.py` |
| **Scoring** | Single decision path used by the UI and the CLI | `src/predict.py` |
| **UI** | Streamlit app: scoring, batch, metrics, EDA | `app.py` |

## Results

Held-out test set (400 applicants):

| Metric | Value |
| --- | ---: |
| Accuracy | **0.8975** |
| ROC-AUC | **0.9787** |
| Precision (Not approved) | 0.8667 |
| Recall (Not approved) | 0.8827 |
| Precision (Approved) | 0.9191 |
| Recall (Approved) | 0.9076 |

Every coefficient carries the sign the synthetic scoring rule implies — `credit_score`,
`income` and `employment_years` push toward approval; `dti`, `loan_amount` and `dependents`
push toward rejection. Full breakdown in [`outputs/evaluation_report.md`](outputs/evaluation_report.md).

## Quick start

Requires Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

Run the whole data + model pipeline:

```bash
python run_pipeline.py
```

Launch the web UI:

```bash
streamlit run app.py
```

It opens at **http://localhost:8501**. No API key, no account, no network access required.

If `streamlit` isn't on your PATH, use `python -m streamlit run app.py`.

## The UI

Four tabs:

**Score an applicant** — a form for the eight applicant fields, with the debt-to-income ratio
updating live. Submitting gives the decision, the approval probability, a plain-English
explanation, and a bar chart breaking down each feature's contribution to the log-odds. Those
contributions are `coefficient × standardised value`, so together with the intercept they sum
to the model's logit exactly — the explanation is the arithmetic, not a narrative layered on top.

**Batch scoring** — upload a CSV and get every row scored, with a summary approval rate and a
download button. Required columns: `age`, `income`, `credit_score`, `loan_amount`,
`loan_term`, `existing_debt`, `employment_years`, `dependents` — `dti` is computed for you.
`data/credit_risk_synthetic.csv` works as a test file.

**Model performance** — test-set metrics, the confusion matrix, the ROC curve, the coefficient
chart, and the full evaluation report.

**Data & EDA** — the synthetic dataset and all five EDA figures.

Check the scoring logic from the command line instead:

```bash
python demo_tool.py
```

## Project structure

```
credit-risk-prediction/
├── app.py                   # Streamlit UI — the main interface
├── src/
│   ├── config.py            # paths, feature lists, seed — single source of truth
│   ├── generate_data.py     # Data Layer: synthetic applicants (slide 6)
│   ├── eda.py               # Data Layer: distributions, correlations, balance (slide 7)
│   ├── train_model.py       # Model Layer: split, scale, fit, evaluate, interpret (slides 7–9)
│   └── predict.py           # shared scoring logic — used by the UI and the CLI demo
├── data/credit_risk_synthetic.csv
├── models/                  # logistic_model.joblib, scaler.joblib, model_metadata.json
├── outputs/
│   ├── eda_summary.md
│   ├── evaluation_report.md
│   └── figures/             # 8 PNGs
├── docs/WRITEUP.md          # problem, method, results, limitations
├── run_pipeline.py          # runs the three pipeline scripts in order
├── demo_tool.py             # exercises the scoring logic from the CLI
├── requirements.txt
└── CLAUDE.md                # architecture notes for Claude Code
```

`src/` is deliberately not a package — entry points add it to `sys.path`, and modules inside
it import each other flat (`from config import ...`).

## Design notes

**The target is generated, not observed.** Each applicant gets a latent score

```
score = 0.60
      + 0.02   × (credit_score − 650)
      + 0.00003 × (income − 50,000)
      − 0.000004 × (loan_amount − 200,000)
      − 3.0    × dti
      + 0.05   × employment_years
      − 0.10   × dependents
      + Normal(0, 0.5)
```

pushed through a sigmoid; `loan_approved = 1` when the probability exceeds 0.5. That yields a
**59.5 / 40.5** class balance, inside the 55/45–70/30 band the design calls for.

**Eight model features, not nine.** The UI collects the eight raw fields an applicant is
described by, but `existing_debt` is dropped from the model in favour of the `dti` ratio it
feeds. Keeping both would put two near-collinear columns into a model whose whole selling
point is readable coefficients.

**`age` and `loan_term` are controls.** They are deliberately absent from the scoring rule, so
their near-zero coefficients (−0.084 each) are a sanity check that the model isn't inventing
signal.

**Scaling is fitted on training data only.** `StandardScaler` sees `X_train` and only
transforms `X_test`, so no test-set statistics leak into training.

**The explanation is the arithmetic.** Each factor's contribution is `coefficient × standardised
value`; summed with the intercept these reproduce the model's logit exactly. Nothing is
approximated or narrated after the fact.

**Reproducible.** Seed 42 throughout; rerunning the pipeline reproduces these numbers exactly.
The figures quoted above are hardcoded here and in `docs/WRITEUP.md` — changing the seed,
sample size, generating weights or model hyperparameters will make them stale.

## Verification

There is no test suite. `demo_tool.py` is the regression check — it runs five fixed profiles
through the same `score_applicant` the UI calls, covering the approve path, the reject path
and the input-validation guard:

```bash
python demo_tool.py
```

Scoring all 2,000 generated applicants back through the trained model agrees with the ground
truth labels **91.4%** of the time.

## Limitations

Synthetic data, a rule-based target that is easier to learn than real default behaviour, no
macroeconomic factors, no fairness audit across demographic groups, and a single train/test
split with no cross-validation. See [`docs/WRITEUP.md`](docs/WRITEUP.md) for the full
discussion and future scope.
