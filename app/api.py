"""
Flask web service for credit card default prediction.
Endpoints:
  POST /predict          — predict default for a single client
  POST /predict?version= — specify model version (v1 or v2)
  GET  /health           — service health check
  GET  /models           — list available models and their metadata
"""
import os
import json
import logging
import random
import time
from datetime import datetime
from flask import Flask, request, jsonify

from model_handler import predict, get_metadata, FEATURE_COLUMNS

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ── A/B routing weights (modifiable at runtime via env var) ──────────────────
AB_SPLIT = float(os.environ.get('AB_SPLIT_V2', '0.5'))   # fraction sent to v2


def _log_request(endpoint: str, payload: dict, result: dict, duration_ms: float):
    """Emit a structured JSON log line (ELK-ready)."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoint": endpoint,
        "model_version": result.get("model_version"),
        "prediction": result.get("prediction"),
        "probability": result.get("probability"),
        "duration_ms": round(duration_ms, 2),
        "features_received": list(payload.keys())
    }
    logger.info(json.dumps(log_entry))


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Health check — used by Docker / load-balancers."""
    return jsonify({
        'status': 'healthy',
        'service': 'credit-card-default-predictor',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 200


@app.route('/models', methods=['GET'])
def models_info():
    """Return metadata for all available models."""
    return jsonify({
        'available_versions': ['v1', 'v2'],
        'metadata': get_metadata(),
        'ab_split_v2': AB_SPLIT
    }), 200


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Predict credit card default.

    Query params:
      version — 'v1', 'v2', or 'ab' (random A/B routing). Default: 'ab'.

    Request body (JSON):
      {
        "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
        "PAY_0": 0,  "PAY_2": 0,  "PAY_3": 0,  "PAY_4": 0,  "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 15000, ... "BILL_AMT6": 10000,
        "PAY_AMT1": 1500,  ... "PAY_AMT6": 1000
      }

    Response:
      {
        "prediction":    0 or 1,
        "probability":   0.0–1.0,
        "model_version": "v1" | "v2",
        "risk_label":    "LOW" | "HIGH"
      }
    """
    t0 = time.time()
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({'error': 'Request body must be valid JSON'}), 400

        # Determine model version
        version_param = request.args.get('version', 'ab').lower()
        if version_param == 'ab':
            version = 'v2' if random.random() < AB_SPLIT else 'v1'
        elif version_param in ('v1', 'v2'):
            version = version_param
        else:
            return jsonify({'error': "version must be 'v1', 'v2', or 'ab'"}), 400

        result = predict(data, version=version)
        duration_ms = (time.time() - t0) * 1000
        _log_request('/predict', data, result, duration_ms)
        return jsonify(result), 200

    except FileNotFoundError as e:
        logger.error(str(e))
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        logger.exception("Unexpected error in /predict")
        return jsonify({'error': str(e)}), 500


@app.route('/features', methods=['GET'])
def features():
    """Return the list of expected input features."""
    return jsonify({'features': FEATURE_COLUMNS}), 200


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    logger.info(f"Starting service on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
