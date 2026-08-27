# EDA Summary

- Rows: **2,000**, columns: **10**
- Missing values: **0**, duplicate rows: **0**
- Class balance: **1,190 approved (59.5%)** vs **810 not approved (40.5%)** -- within the 55/45-70/30 design band

## Correlation with `loan_approved`

| Feature | Correlation |
| --- | ---: |
| `credit_score` | +0.649 |
| `dti` | -0.285 |
| `income` | +0.267 |
| `employment_years` | +0.177 |
| `existing_debt` | -0.160 |
| `loan_amount` | -0.112 |
| `dependents` | -0.072 |
| `age` | -0.042 |
| `loan_term` | -0.023 |

`age` and `loan_term` are excluded from the synthetic scoring rule by design, so their correlations should sit near zero -- they act as controls.

## Figures

- `figures/feature_distributions.png`
- `figures/target_correlations.png`
- `figures/correlation_matrix.png`
- `figures/class_balance.png`
- `figures/key_predictors_by_class.png`

Model feature set (8): `age`, `income`, `credit_score`, `loan_amount`, `loan_term`, `employment_years`, `dependents`, `dti`
