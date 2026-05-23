# Architecture Documentation

## Monolith vs Microservices Decision

### Choice: Monolithic architecture

For this project, a **monolithic** deployment was chosen.

**Reasons:**

1. **Single ML model per request** — there is no need to split training, inference,
   and feature engineering into independent services; they share code and state
   trivially.
2. **Team size** — a single ML engineer can own, deploy, and debug one container
   more effectively than a distributed mesh.
3. **Latency** — in-process calls between the API router and the model handler
   have zero network overhead; microservices would add 1–5 ms per hop.
4. **Operational complexity** — no service mesh (Istio/Linkerd), no inter-service
   authentication, no distributed tracing required.
5. **A/B routing** — implemented inside the Flask process using a single
   environment variable (`AB_SPLIT_V2`), which is simpler and equally effective
   at this scale.

### When to migrate to microservices

If the system were to scale to 10+ models, different SLAs per endpoint, or
independent team ownership of each component, the following split would make
sense:

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  API Gateway │────▶│  Feature Store  │────▶│  Model Registry  │
│  (FastAPI)   │     │  (Redis/Feast)  │     │  (MLflow)        │
└──────────────┘     └─────────────────┘     └──────────────────┘
        │
        ▼
┌──────────────────────┐      ┌────────────────────────┐
│  Inference Service v1 │      │  Inference Service v2   │
│  (Flask + model_v1)   │      │  (Flask + model_v2)     │
└──────────────────────┘      └────────────────────────┘
```

---

## RabbitMQ — Asynchronous Processing Concept

In a high-throughput production scenario, synchronous HTTP inference is
inadequate when:

- Batch scoring jobs arrive for thousands of clients overnight.
- Downstream systems (fraud triggers, CRM) consume predictions asynchronously.
- The model is slow (> 100 ms) and clients cannot wait.

**Proposed message flow:**

```
Client ──POST /enqueue──▶ RabbitMQ queue "predict_requests"
                                │
                    ┌───────────▼───────────┐
                    │  Worker (consumer)    │
                    │  - loads model        │
                    │  - runs inference     │
                    │  - writes result      │
                    └───────────┬───────────┘
                                │
                    RabbitMQ queue "predict_results"
                                │
                    Client polls GET /result/{job_id}
```

For this project, synchronous inference is sufficient; the RabbitMQ
architecture is documented here as the production upgrade path.

---

## Logging and Monitoring

### Current implementation

Every `/predict` call emits a structured JSON log line to stdout:

```json
{
  "timestamp":      "2024-11-15T09:23:11Z",
  "endpoint":       "/predict",
  "model_version":  "v2",
  "prediction":     1,
  "probability":    0.71,
  "duration_ms":    12.4,
  "features_received": ["LIMIT_BAL", "AGE", ...]
}
```

### Production ELK stack

```
Flask stdout
     │
     ▼
Filebeat (sidecar container)
     │
     ▼
Logstash (parse JSON, enrich)
     │
     ▼
Elasticsearch (index)
     │
     ▼
Kibana dashboard
  - requests/sec per model version
  - prediction distribution (% HIGH risk)
  - latency percentiles (p50, p95, p99)
  - error rate
```

---

## DVC and MLflow — MLOps Concepts

### DVC (Data Version Control)

DVC tracks large files (datasets, model binaries) outside Git by storing
them in remote storage (S3, GCS, Azure Blob) and keeping only small `.dvc`
pointer files in the repository.

**In this project DVC would manage:**
- `data/UCI_Credit_Card.csv` — raw dataset
- `models/model_v1.pkl`, `models/model_v2.pkl` — trained model artifacts

**Example workflow:**
```bash
dvc add data/UCI_Credit_Card.csv   # creates data/UCI_Credit_Card.csv.dvc
dvc push                            # uploads to remote storage
dvc pull                            # team member downloads exact same file
```

### MLflow

MLflow tracks experiment parameters, metrics, and artifacts across training runs.

**In this project MLflow would record for each training run:**
- Parameters: `n_estimators`, `learning_rate`, `max_depth`, `random_state`
- Metrics: `f1_score`, `precision`, `recall`, `roc_auc`
- Artifacts: trained `.pkl` file, `classification_report.txt`

**Example:**
```python
import mlflow
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("f1_score", 0.62)
    mlflow.sklearn.log_model(pipeline_v2, "model_v2")
```

This enables reproducible model comparison and one-click rollback to any
previous version.

---

## Business Metrics

| Metric | Formula | Owner |
|--------|---------|-------|
| **Expected Loss Reduction** | `(TP_new − TP_old) × avg_exposure × historical_loss_rate` | Risk department |
| **Approval Rate at Same Risk** | `TN / (TN + FP)` | Product / Revenue |

These metrics translate model quality (F1, Recall) into financial language
that stakeholders understand, bridging ML evaluation and business decision-making.

---

## ONNX-ML (Overview)

The trained scikit-learn pipeline can be exported to ONNX format for
runtime-agnostic, optimised inference:

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 23]))]
onnx_model = convert_sklearn(pipeline_v1, initial_types=initial_type)
with open("models/model_v1.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

**Benefits:**
- ~2–10× faster inference via ONNX Runtime vs pure Python
- Language-agnostic: can be served from C++, Java, or .NET
- Smaller model files for edge deployment

---

## uWSGI + NGINX

For production traffic, Flask's built-in development server is replaced
by **uWSGI** (application server) behind **NGINX** (reverse proxy):

```
Internet ──▶ NGINX (80/443)
               │  SSL termination, rate limiting, static files
               ▼
          uWSGI (4 worker processes × 2 threads)
               │  Python WSGI protocol, process management
               ▼
          Flask application
               │
               ▼
          Model inference
```

This stack handles thousands of concurrent requests safely, whereas Flask's
dev server is single-threaded and unsuitable for production.
