"""
Train credit card default prediction models (v1 and v2).
Uses the UCI Default of Credit Card Clients Dataset.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import json

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]
TARGET_COLUMN = 'default.payment.next.month'


def load_data():
    """Load UCI Credit Card dataset."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'UCI_Credit_Card.csv')

    print(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    return df


def train_and_save_models():
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Model v1: Logistic Regression (simple, interpretable baseline)
    pipeline_v1 = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
        ))
    ])
    pipeline_v1.fit(X_train, y_train)
    y_pred_v1 = pipeline_v1.predict(X_test)
    f1_v1 = f1_score(y_test, y_pred_v1)
    print("\n=== Model v1 (LogisticRegression) ===")
    print(classification_report(y_test, y_pred_v1))

    # Model v2: Gradient Boosting (improved performance)
    pipeline_v2 = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=3, random_state=RANDOM_STATE
        ))
    ])
    pipeline_v2.fit(X_train, y_train)
    y_pred_v2 = pipeline_v2.predict(X_test)
    f1_v2 = f1_score(y_test, y_pred_v2)
    print("\n=== Model v2 (GradientBoosting) ===")
    print(classification_report(y_test, y_pred_v2))

    # Save models
    models_dir = os.path.dirname(__file__)
    joblib.dump(pipeline_v1, os.path.join(models_dir, 'model_v1.pkl'))
    joblib.dump(pipeline_v2, os.path.join(models_dir, 'model_v2.pkl'))

    # Save metadata
    metadata = {
        "v1": {
            "algorithm": "LogisticRegression",
            "f1_score": round(f1_v1, 4),
            "features": FEATURE_COLUMNS
        },
        "v2": {
            "algorithm": "GradientBoostingClassifier",
            "f1_score": round(f1_v2, 4),
            "features": FEATURE_COLUMNS
        }
    }
    with open(os.path.join(models_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n model_v1.pkl (F1={f1_v1:.4f}), model_v2.pkl (F1={f1_v2:.4f})")
    return metadata


if __name__ == '__main__':
    train_and_save_models()
