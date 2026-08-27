# Credit Risk Loan Approval — Write-up

## 1. Problem statement

Predict whether a loan applicant should be **approved** or **not approved**, using a fully
synthetic applicant dataset and an interpretable model, then expose that model through a web
interface so that entering an applicant's details produces both a decision and an explanation
of it.

Three constraints shaped the design:

- **No real or public data.** The dataset is simulated end to end.
- **The model must stay interpretable.** Logistic regression only — no ensembles, no black boxes.
- **No external services.** Everything runs locally; no API keys, no accounts, no network calls.

The deck originally specified a conversational Google ADK agent as the interface and explicitly
ruled out Streamlit and Flask. That constraint was lifted during implementation and the project
now ships a Streamlit UI instead — a deliberate, recorded deviation from the original design.

Target variable: `loan_approved`, binary, `1 = Approved` / `0 = Not Approved`.

## 2. Method

### 2.1 Data layer

2,000 applicants are simulated with NumPy (seed 42, so the dataset is reproducible):

| Feature | Type | Simulated as |
| --- | --- | --- |
| `age` | int | 21–65, uniform |
| `income` | float | Normal(50k, 15k), clipped at 15k |
| `credit_score` | float | Normal(650, 80), clipped to 300–850 |
| `loan_amount` | float | Normal(200k, 80k), clipped at 20k |
| `loan_term` | int | 12 / 24 / 36 / 48 / 60 months |
| `existing_debt` | float | Normal(10k, 8k), clipped at 0 |
| `employment_years` | int | 0–30, uniform |
| `dependents` | int | 0–4, uniform |
| `dti` | float | engineered: `existing_debt / income` |

The label is not sampled independently — it is produced from a weighted latent score:

```
score = 0.60
      + 0.02     × (credit_score − 650)
      + 0.00003  × (income − 50,000)
      − 0.000004 × (loan_amount − 200,000)
      − 3.0      × dti
      + 0.05     × employment_years
      − 0.10     × dependents
      + Normal(0, 0.5)

prob = sigmoid(score)
loan_approved = 1 if prob > 0.5
```

The intercept of 0.60 was chosen to land the approval rate inside the 55/45–70/30 band the
design specifies rather than an extreme skew. The realised balance is **1,190 approved (59.5%)
vs 810 not approved (40.5%)**.

Because we wrote the generating rule, we know exactly which relationships the model *should*
recover — which turns coefficient interpretation from guesswork into a verifiable check.

### 2.2 EDA

`src/eda.py` produces histograms for every feature, a full correlation matrix, correlations
against the target, boxplots of the key drivers split by outcome, and the class-balance check.
No missing values and no duplicate rows. Correlations with `loan_approved`:

| Feature | Correlation |
| --- | ---: |
| `credit_score` | +0.649 |
| `dti` | −0.285 |
| `income` | +0.267 |
| `employment_years` | +0.177 |
| `existing_debt` | −0.160 |
| `loan_amount` | −0.112 |
| `dependents` | −0.072 |
| `age` | −0.042 |
| `loan_term` | −0.023 |

The ordering matches the generating rule: `credit_score` dominates, `dti` is the strongest
negative, and `age` and `loan_term` — which never entered the rule — sit near zero.

### 2.3 Preprocessing

- **Split:** 80/20, stratified on `loan_approved`, so train and test both carry a 59.5%
  approval rate (1,600 / 400 rows).
- **Scaling:** `StandardScaler` is fitted on the training set only and merely applied to the
  test set. Logistic regression is scale-sensitive and fitting the scaler on the full dataset
  would leak test-set statistics into training.
- **Feature set:** eight columns — `age`, `income`, `credit_score`, `loan_amount`, `loan_term`,
  `employment_years`, `dependents`, `dti`. `existing_debt` is intentionally excluded: `dti`
  already carries it, and two near-collinear columns would destabilise the very coefficients
  this project exists to read. The UI still collects `existing_debt` and computes `dti` from it,
  so nothing is lost at the interface.

### 2.4 Model

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
```

Logistic regression, not a tree ensemble, because every coefficient is directly readable. In a
credit decision the ability to tell a reviewer *why* an applicant was rejected is not a
nice-to-have — it is the product.

### 2.5 Interface layer

All scoring goes through a single function, `score_applicant` in `src/predict.py`, which takes
the eight raw applicant fields, computes `dti`, applies the persisted scaler, and calls the
persisted model. Keeping it in one module means the web UI and the command-line demo cannot
drift apart.

Beyond the decision and probability, it returns the per-feature contributions to the log-odds
(`coefficient × scaled value`). Summed with the intercept these reproduce the model's logit
exactly, so the explanation shown to the user *is* the arithmetic of the decision rather than a
narrative laid over it. `explain_in_words` turns the same numbers into a plain-English sentence.

The Streamlit app (`app.py`) presents four tabs:

- **Score an applicant** — a form for the eight fields, returning the decision, probability,
  written explanation and a contribution bar chart.
- **Batch scoring** — CSV upload, vectorised scoring via `score_batch`, and a download of the
  results.
- **Model performance** — test metrics, confusion matrix, ROC curve, coefficient chart and the
  full evaluation report.
- **Data & EDA** — the dataset and every EDA figure.

The app refuses to start with a clear message if the model artifacts are missing, pointing the
user at `run_pipeline.py`.

## 3. Results

Held-out test set, 400 applicants:

| Metric | Value |
| --- | ---: |
| Accuracy | **0.8975** |
| ROC-AUC | **0.9787** |
| Precision (Not approved) | 0.8667 |
| Recall (Not approved) | 0.8827 |
| Precision (Approved) | 0.9191 |
| Recall (Approved) | 0.9076 |

Training accuracy is 0.9181 against test accuracy of 0.8975 — a gap of about 2 points, which
indicates a well-fitted rather than overfitted model.

### Confusion matrix

| | Predicted: Not approved | Predicted: Approved |
| --- | ---: | ---: |
| **Actual: Not approved** | 143 (TN) | 19 (FP) |
| **Actual: Approved** | 22 (FN) | 216 (TP) |

The 19 **false approvals** are the expensive error — the model let through applicants the
generating rule would have rejected. The 22 **false rejections** cost business rather than money
lost. Recall on the "not approved" class is 0.8827, meaning the model catches roughly 88% of the
applicants it should decline; a lender wanting to be more conservative would raise the decision
threshold above 0.5 and trade approval recall for that.

### Coefficient interpretation

Features are standardised, so each coefficient is the change in log-odds of approval per one
standard-deviation increase:

| Feature | Coefficient | Odds ratio | Pushes | Expected | Matches? |
| --- | ---: | ---: | --- | :---: | :---: |
| `credit_score` | +5.0078 | 149.57 | toward approval | + | yes |
| `dti` | −1.8340 | 0.16 | toward rejection | − | yes |
| `income` | +1.3914 | 4.02 | toward approval | + | yes |
| `employment_years` | +1.3909 | 4.02 | toward approval | + | yes |
| `loan_amount` | −1.0105 | 0.36 | toward rejection | − | yes |
| `dependents` | −0.6053 | 0.55 | toward rejection | − | yes |
| `loan_term` | −0.0843 | 0.92 | — | control | n/a |
| `age` | −0.0841 | 0.92 | — | control | n/a |

Intercept: +1.4758.

**Every directional expectation holds.** `credit_score` dominates by a wide margin — a one-SD
increase (about 80 points) multiplies the odds of approval by roughly 150. `dti` is the
strongest brake. And the two control features, which were never part of the generating rule,
came back at −0.084 apiece: effectively zero, confirming the model is not manufacturing signal
from noise.

The ROC-AUC of 0.979 is high by real-world credit-scoring standards, and that is a property of
the data, not a triumph of the model — see the first limitation below.

## 4. Limitations

- **Synthetic data.** Results will not generalise to real applicants. The numbers here measure
  how well logistic regression recovers a rule we wrote, not how well it predicts default.
- **The target is rule-based**, and the rule is itself a sigmoid of a linear combination — which
  is precisely the functional form logistic regression assumes. That structural match is the
  main reason ROC-AUC reaches 0.979; real default behaviour is far noisier and non-linear, and a
  realistic AUC would be closer to 0.70–0.80.
- **No macroeconomic or time-series factors.** Interest rates, unemployment, and seasonality all
  move real default rates and are entirely absent.
- **No fairness or bias audit.** The dataset carries no protected attributes, so none could be
  audited — but that also means the pipeline has never been tested for disparate impact, which a
  production credit model would require by law in most jurisdictions.
- **A single train/test split**, no cross-validation, so the reported metrics carry sampling
  variance that is not quantified.
- **`existing_debt` clipped at zero** creates a spike of applicants with exactly zero debt and
  therefore zero DTI, which is a slightly unrealistic mass point in the distribution.

## 5. Future scope

- Benchmark against a tree-based model (Random Forest / XGBoost) to measure what interpretability
  actually costs in accuracy on this problem.
- Add WoE / Information Value binning to produce a true points-based scorecard, the format credit
  risk teams actually deploy.
- Cross-validation and hyperparameter tuning (regularisation strength `C`, penalty type) with
  confidence intervals on the metrics.
- SHAP explanations alongside the coefficients, so per-applicant attributions have a principled
  basis rather than the raw `coefficient × value` decomposition used here.
- Threshold tuning driven by an explicit cost matrix — false approvals and false rejections do not
  cost a lender the same amount, and 0.5 is only the right cutoff when they do.
- Swap in a real, anonymised dataset once available, and re-run the fairness audit that synthetic
  data made impossible.

## 6. Reproducing

```bash
pip install -r requirements.txt
python run_pipeline.py     # data → EDA → train, ~10 seconds
streamlit run app.py       # web UI at http://localhost:8501
python demo_tool.py        # or score from the command line
```

Seed 42 is fixed throughout, so every number in this write-up reproduces exactly.
