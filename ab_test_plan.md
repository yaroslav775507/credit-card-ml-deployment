# A/B Test Plan — Credit Card Default Prediction

## 1. Objective

Compare **Model v1** (Logistic Regression) against **Model v2** (Gradient Boosting)
to determine whether v2 delivers a statistically significant improvement in business
outcomes without increasing risk exposure.

---

## 2. Hypothesis

- **H₀ (null):** There is no significant difference in default-detection F1-score between v1 and v2.
- **H₁ (alternative):** Model v2 achieves a higher F1-score for the default class (class = 1).

---

## 3. Traffic Allocation

| Group     | Model | Traffic Share |
|-----------|-------|--------------|
| Control   | v1 — LogisticRegression  | 50% |
| Treatment | v2 — GradientBoosting    | 50% |

**Routing mechanism:** Each incoming `/predict` request without an explicit `version`
parameter is randomly assigned to v1 or v2 with equal probability (controlled by
the `AB_SPLIT_V2=0.5` environment variable).

```python
version = 'v2' if random.random() < AB_SPLIT else 'v1'
```

Assignment is **stateless per request** (no sticky sessions), which is appropriate
for a stateless scoring API where individual predictions are independent.

---

## 4. Test Duration

| Parameter         | Value                                   |
|-------------------|-----------------------------------------|
| Minimum duration  | **4 weeks** (captures weekly patterns)  |
| Minimum samples   | **1 000 predictions per group**         |
| Early stopping    | Not applied — avoids inflated Type I error |

---

## 5. Metrics

### 5.1 Primary Technical Metric

**F1-score (default class = 1)**

$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Chosen because the dataset is imbalanced (~22% defaults), and we care equally
about catching real defaults (Recall) and not flagging good customers (Precision).

### 5.2 Secondary Technical Metric

**Recall (Sensitivity) for class = 1**

$$\text{Recall} = \frac{TP}{TP + FN}$$

*Rationale:* In credit risk, missing a real defaulter (FN) costs the bank more
than a false alarm (FP). Recall directly measures how many defaulters we catch.

### 5.3 Business Metrics

| Metric | Formula | Rationale |
|--------|---------|-----------|
| **Expected Loss Reduction (ELR)** | `ELR = (TP_v2 − TP_v1) × avg_credit_limit × loss_rate` | Quantifies the additional financial exposure avoided by the new model |
| **Approval Rate at Same Risk** | `Approval% = TN / (TN + FP)` | Measures how many creditworthy customers are correctly approved; a better model improves customer experience without increasing risk |

---

## 6. Statistical Analysis

### 6.1 Test Statistic

Use a **two-proportion z-test** to compare F1-scores between groups:

$$z = \frac{\hat{p}_2 - \hat{p}_1}{\sqrt{\hat{p}(1 - \hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}$$

where $\hat{p}$ is the pooled proportion of correct predictions and $n_1, n_2$
are group sizes.

### 6.2 Confidence Intervals

95% CI for the difference in F1-scores (Wilson interval):

$$\Delta F1 \pm 1.96 \cdot \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

### 6.3 Significance Threshold

| Parameter | Value |
|-----------|-------|
| Significance level α | **0.05** |
| Minimum detectable effect | **+0.03 F1 points** |
| Statistical power (1-β) | **0.80** |

### 6.4 Success Criterion

Model v2 is promoted to 100% traffic if **all** of the following hold:

1. `F1_v2 − F1_v1 ≥ 0.03` (practical significance)
2. `p-value < 0.05` (statistical significance)
3. `Recall_v2 ≥ Recall_v1` (no regression in loss avoidance)
4. `Approval Rate_v2 ≥ Approval Rate_v1 − 0.01` (no degradation for good customers)

---

## 7. Practical Implementation

### API usage during test

```bash
# Force v1 (control)
curl -X POST http://localhost:5000/predict?version=v1 \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL": 50000, "AGE": 35, ...}'

# Force v2 (treatment)
curl -X POST http://localhost:5000/predict?version=v2 \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL": 50000, "AGE": 35, ...}'

# A/B routing (default — 50/50 random)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL": 50000, "AGE": 35, ...}'
```

### Log schema for A/B analysis

Every prediction emits a structured JSON log line:

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

These logs are collected by the ELK stack (Elasticsearch + Logstash + Kibana)
and aggregated into a dashboard showing per-version metrics in near real-time.

---

## 8. Roll-back Plan

If at any point during the test period:

- Recall_v2 drops **> 5 percentage points** below Recall_v1, **or**
- Error rate for v2 exceeds **1%** of requests

→ Immediately set `AB_SPLIT_V2=0.0` (route 100% to v1) without redeployment.

---

## 9. Architectural Influence

The A/B routing is implemented at the **application layer** inside the Flask
service, keeping the architecture monolithic and avoiding extra infrastructure.
For a production system at scale, routing could be moved to:

- **NGINX** (weighted upstream configuration) — zero application-level overhead
- **API Gateway** (AWS / GCP) — full traffic management without touching code
- **Feature flag service** (LaunchDarkly, Unleash) — dynamic percentage control

The current design satisfies the course requirements while documenting the
production upgrade path.
