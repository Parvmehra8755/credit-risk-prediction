# Model Evaluation Report

Logistic regression on a synthetic credit-risk dataset. All metrics below are computed on the held-out 20% test set.

## Headline metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8975 |
| ROC-AUC | 0.9787 |
| Precision (Not approved) | 0.8667 |
| Recall (Not approved) | 0.8827 |
| Precision (Approved) | 0.9191 |
| Recall (Approved) | 0.9076 |

## Confusion matrix

| | Predicted: Not approved | Predicted: Approved |
| --- | ---: | ---: |
| **Actual: Not approved** | 143 (TN) | 19 (FP -- false approval) |
| **Actual: Approved** | 22 (FN -- false rejection) | 216 (TP) |

False approvals (19) are the costlier error for a lender: the model let through an applicant the generating rule would have rejected. False rejections (22) cost business rather than money lost.

## Per-class report

```
                  precision    recall  f1-score   support

Not approved (0)     0.8667    0.8827    0.8746       162
    Approved (1)     0.9191    0.9076    0.9133       238

        accuracy                         0.8975       400
       macro avg     0.8929    0.8951    0.8940       400
    weighted avg     0.8979    0.8975    0.8976       400
```

## Coefficient interpretation

Features are standardised, so each coefficient is the change in log-odds of approval per one standard-deviation increase in that feature. Ranked by absolute magnitude:

| Feature | Coefficient | Odds ratio | Pushes | Expected sign | Matches? |
| --- | ---: | ---: | --- | :---: | :---: |
| `credit_score` | +5.0078 | 149.5724 | toward approval | + | yes |
| `dti` | -1.8340 | 0.1598 | toward rejection | - | yes |
| `income` | +1.3914 | 4.0205 | toward approval | + | yes |
| `employment_years` | +1.3909 | 4.0184 | toward approval | + | yes |
| `loan_amount` | -1.0105 | 0.3640 | toward rejection | - | yes |
| `dependents` | -0.6053 | 0.5459 | toward rejection | - | yes |
| `loan_term` | -0.0843 | 0.9192 | toward rejection | n/a (control) | n/a |
| `age` | -0.0841 | 0.9193 | toward rejection | n/a (control) | n/a |

Intercept: `+1.4758`

Every coefficient with a directional expectation came out with the sign the synthetic scoring rule implies: `credit_score`, `income` and `employment_years` push toward approval, while `dti`, `loan_amount` and `dependents` push toward rejection. The model recovered the data-generating process.

`age` and `loan_term` are controls: they were left out of the synthetic scoring rule, so near-zero coefficients on them are the correct result and a useful sanity check that the model is not inventing signal.

## Figures

- `figures/confusion_matrix.png`
- `figures/roc_curve.png`
- `figures/coefficients.png`
