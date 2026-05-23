"""
Tests for the Flask API endpoints.
Run with: pytest tests/test_api.py -v
"""
import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'app'))

from app.api import app

SAMPLE_PAYLOAD = {
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 15000, "BILL_AMT2": 14000, "BILL_AMT3": 13000,
    "BILL_AMT4": 12000, "BILL_AMT5": 11000, "BILL_AMT6": 10000,
    "PAY_AMT1": 1500, "PAY_AMT2": 1400, "PAY_AMT3": 1300,
    "PAY_AMT4": 1200, "PAY_AMT5": 1100, "PAY_AMT6": 1000
}

HIGH_RISK_PAYLOAD = {**SAMPLE_PAYLOAD, "PAY_0": 8, "PAY_2": 7, "LIMIT_BAL": 20000}


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get('/health')
    assert resp.status_code == 200

def test_health_body(client):
    resp = client.get('/health')
    data = resp.get_json()
    assert data['status'] == 'healthy'
    assert 'timestamp' in data


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_v1(client):
    resp = client.post('/predict?version=v1',
                       data=json.dumps(SAMPLE_PAYLOAD),
                       content_type='application/json')
    assert resp.status_code == 200
    body = resp.get_json()
    assert 'prediction' in body
    assert body['prediction'] in (0, 1)
    assert 0.0 <= body['probability'] <= 1.0
    assert body['model_version'] == 'v1'
    assert body['risk_label'] in ('LOW', 'HIGH')

def test_predict_v2(client):
    resp = client.post('/predict?version=v2',
                       data=json.dumps(SAMPLE_PAYLOAD),
                       content_type='application/json')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['model_version'] == 'v2'

def test_predict_default_ab_routing(client):
    resp = client.post('/predict',
                       data=json.dumps(SAMPLE_PAYLOAD),
                       content_type='application/json')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['model_version'] in ('v1', 'v2')

def test_predict_invalid_version(client):
    resp = client.post('/predict?version=v99',
                       data=json.dumps(SAMPLE_PAYLOAD),
                       content_type='application/json')
    assert resp.status_code == 400

def test_predict_empty_body(client):
    resp = client.post('/predict',
                       data='',
                       content_type='application/json')
    assert resp.status_code in (400, 500)

def test_predict_partial_features(client):
    """Missing features should default to 0, not crash."""
    partial = {"LIMIT_BAL": 30000, "AGE": 40}
    resp = client.post('/predict?version=v1',
                       data=json.dumps(partial),
                       content_type='application/json')
    assert resp.status_code == 200


# ── /models ───────────────────────────────────────────────────────────────────

def test_models_endpoint(client):
    resp = client.get('/models')
    assert resp.status_code == 200
    body = resp.get_json()
    assert 'v1' in body['available_versions']
    assert 'v2' in body['available_versions']


# ── /features ─────────────────────────────────────────────────────────────────

def test_features_endpoint(client):
    resp = client.get('/features')
    assert resp.status_code == 200
    body = resp.get_json()
    assert 'LIMIT_BAL' in body['features']
    assert len(body['features']) == 23
