"""
Model handler — loads and manages v1 / v2 models for inference.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '../models')

FEATURE_COLUMNS = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

_models = {}
_metadata = {}


def _load_metadata():
    meta_path = os.path.join(MODELS_DIR, '../models/metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def get_model(version: str = 'v1'):
    """Return cached model, loading from disk on first access."""
    if version not in _models:
        path = os.path.join(MODELS_DIR, f'model_{version}.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        logger.info(f"Loading model {version} from {path}")
        _models[version] = joblib.load(path)
        if not _metadata:
            _metadata.update(_load_metadata())
    return _models[version]


def get_metadata(version: str = None):
    if not _metadata:
        _metadata.update(_load_metadata())
    if version:
        return _metadata.get(version, {})
    return _metadata


def preprocess_input(data: dict) -> np.ndarray:
    """
    Convert JSON input dict to feature array in the correct column order.
    Missing features default to 0.
    """
    row = {col: float(data.get(col, 0)) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict(data: dict, version: str = 'v1') -> dict:
    """Run inference and return prediction + probability."""
    model = get_model(version)
    X = preprocess_input(data)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    return {
        'prediction': prediction,
        'probability': round(probability, 4),
        'model_version': version,
        'risk_label': 'HIGH' if prediction == 1 else 'LOW'
    }
