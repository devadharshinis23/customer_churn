# Telecom Customer Churn Prediction: Recall-Focused XGBoost vs Logistic Regression

An end-to-end churn prediction pipeline that identifies customers at high risk of leaving. The focus is on **catching churners (recall / PR-AUC) instead of maximising accuracy**, with a leakage-safe preprocessing pipeline and a fair model comparison at a fixed recall target.

## Problem

Predict whether a telecom customer will churn (1) or stay (0). Only 26.5% of customers churn, so accuracy is misleading: a model that predicts "nobody churns" scores 73.5% accuracy and catches zero churners. The goal here is to catch about 92% of churners and to understand the cost of doing so.

## Dataset

[Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers, 21 columns (demographics, services, contract, billing). Downloaded automatically in the notebook with `kagglehub`.

## Results

Test set of 1,409 customers (374 churners). Both models have their decision threshold tuned to a **92% recall target**, chosen on out-of-fold *training* predictions, so the test set stays unseen.

| Model | Threshold | Recall | Precision | F1 | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.311 | 0.925 | 0.432 | 0.589 | 0.636 |
| XGBoost (GridSearchCV) | 0.262 | 0.925 | 0.424 | 0.581 | 0.665 |

Confusion matrices at the tuned thresholds:

| Model | Churners caught | Churners missed | False alarms |
|---|---|---|---|
| Logistic Regression | 346 | 28 | 454 |
| XGBoost | 346 | 28 | 471 |

XGBoost cross-validated PR-AUC was 0.667 (test 0.665), so it does not overfit.

### Key findings

- **Both models catch about 92.5% of churners.** The threshold approach reaches the recall target on unseen data.
- **XGBoost has the better PR-AUC (0.665 vs 0.636), but at equal recall it has no precision advantage.** Logistic Regression is slightly higher (43.2% vs 42.4%). On this dataset the simpler, more interpretable model is a reasonable production choice.
- **High recall has a cost.** Precision is about 42-43%, so most flagged customers will not churn. XGBoost flags 817 of 1,409 test customers (58%). That is about 1.6x the 26.5% base churn rate, so the model ranks risk better than random but is not sharp.
- **Accuracy is not optimised.** At these thresholds it is about 65%, below the 73.5% "predict nobody churns" baseline, which is expected when recall is prioritised.

## Pipeline

```
[1] Data ingestion            Kaggle Telco Customer Churn (kagglehub)
[2] Data cleaning             Remove duplicates, fix data types (row-level only)
[3] Feature engineering       Charge trend, tenure groups, number of services
[4] Target definition         Churn = 1 (left), 0 (retained)
[5] Train / test split        Stratified 80/20
[6] Preprocessing pipeline    Median/mode imputation, 1st-99th percentile clipping,
                              scaling, one-hot encoding (fitted on training data only)
[7] Class imbalance           Balance check -> class weights / scale_pos_weight
[8] Baseline model            Logistic Regression
[9] XGBoost model             GridSearchCV (5-fold, 160 fits), refit on PR-AUC
[10] Threshold tuning         Per-model threshold targeting 92% recall (out-of-fold, train only)
[11] Evaluation               Recall, precision, F1, PR-AUC, confusion matrices, PR curves
[12] Output                   Risk score and high-risk flag per test-set customer
```

### Leakage control

Everything that learns from data (imputation, percentile clipping, scaling, encoding, class weighting) lives inside a scikit-learn `Pipeline` / `ColumnTransformer`. It is fitted on the training set only and refitted on each training fold during cross-validation. Thresholds are selected from out-of-fold training predictions, never from the test set. A small custom transformer (`QuantileClipper`) learns its clipping bounds in `fit()`, since scikit-learn has no built-in percentile clipper.

## Best XGBoost parameters

`n_estimators=200`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, with `scale_pos_weight` set from the class ratio.

## How to run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib kagglehub joblib imbalanced-learn
jupyter notebook churn_prediction.ipynb
```

Or open the notebook in Google Colab and run all cells. The dataset downloads automatically.

Outputs written by the notebook: `churn_scores.csv` (all test customers with risk scores), `high_risk_customers.csv` and `churn_model.joblib` (trained pipeline plus threshold).

## Tech stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Kaggle Hub, Joblib, Google Colab. (`imbalanced-learn` is only needed if you set `USE_SMOTE = True`.)

## Limitations

- **Recency is not available.** The dataset has no purchase dates, so RFM features could not be built.
- **Risk scores are not calibrated probabilities.** Class weighting inflates them, so treat them as a ranking.
- **No cost analysis.** The 92% recall target is fixed; the retention cost and customer value that would justify a different threshold are not modelled.
- **Modest lift.** Precision at 92% recall is around 42-43%.
- **Single dataset and a single random split.** The small precision differences between models may not be statistically meaningful.

## Possible next steps

- Feature importance / SHAP to explain churn drivers
- Cost-based threshold selection
- Repeated cross-validation to test whether model differences are significant
- A dataset with transaction history to add true RFM features

## Repository structure

```
├── churn_prediction.ipynb
├── README.md
```

## Author

Devadharshini S | linkedin.com/in/devadharshinis23 | devadharshinisenthil1@gmail.com
