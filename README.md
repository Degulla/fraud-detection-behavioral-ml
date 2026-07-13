# Behavioral Machine Learning for Fraud Detection

## Project Overview

This project develops a machine learning framework for detecting fraudulent financial transactions in a highly imbalanced dataset containing approximately 1.85 million transactions.

The project focuses on improving fraud detection through past-only behavioral and temporal feature engineering, chronological model evaluation, cost-sensitive analysis, and explainable machine learning.

## Problem Statement

Fraud detection is a rare-event classification problem where fraudulent transactions represent approximately 0.1% of the dataset.

Traditional accuracy-based evaluation can be misleading because a model predicting most transactions as legitimate may still achieve very high accuracy.

The objective of this project is to improve fraud detection recall while reducing false-positive alerts and missed fraudulent transactions.

## Dataset

- Total transactions: 1,858,649
- Fraudulent transactions: 1,874
- Fraud rate: approximately 0.1%
- Original features include transaction type, channel, amount, timestamp, payer and payee information, and transaction locations.

## Machine Learning Models

The following models were explored:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- Artificial Neural Network
- XGBoost

## Class Imbalance Handling

The dataset contains extreme class imbalance.

The project explores:

- Class weighting
- SMOTE
- Scale positive weight for XGBoost
- Decision threshold optimization
- Cost-sensitive fraud analysis

Model performance is evaluated using Precision, Recall, F1-score, ROC-AUC, and PR-AUC.

## Behavioral Feature Engineering

Past-only behavioral and temporal features were engineered to model transaction behavior while reducing future data leakage.

Features include:

- Payer previous transaction count
- Payer historical average transaction amount
- Amount deviation ratio
- New payee indicator
- Payer history indicator
- Minutes since previous transaction
- High transaction velocity indicator
- Location mismatch
- Night transaction indicator
- Weekend transaction indicator

## Chronological Evaluation

Transactions were evaluated using a chronological train-test split to better simulate a real-world fraud detection scenario.

The model is trained on earlier transactions and evaluated on later transactions.

## Final Results

| Metric | Baseline XGBoost | Behavioral XGBoost |
|---|---:|---:|
| PR-AUC | 0.5534 | 0.8292 |
| ROC-AUC | 0.9670 | 0.9991 |
| Fraud Recall | 83% | 93% |
| Fraud F1-score | 7% | 49% |
| False Positives | 10,269 | 836 |
| Missed Fraud | 77 | 33 |

The behavioral model improved PR-AUC from 0.553 to 0.829 and reduced false-positive alerts from 10,269 to 836.

Missed fraud cases were reduced from 77 to 33.

## Explainable AI

SHAP was used to interpret the final XGBoost model and analyze the features influencing fraud predictions.

The explainability analysis helps identify behavioral patterns contributing to suspicious transaction detection.

## Key Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
- Matplotlib
- Jupyter Notebook

## Project Structure

    Fraud_Detection_Final.ipynb
    final_model_results.csv
    final_shap_summary.png
    fraud_detection_xgb_final.pkl
    fraud_model_feature_columns.pkl
    README.md

## Conclusion

Past-only behavioral and temporal transaction features significantly improved fraud detection under chronological evaluation.

The final Behavioral XGBoost model achieved a PR-AUC of 0.829 and 93% fraud recall while substantially reducing false-positive alerts compared with the baseline model.
