# FactoryGuard AI - IoT Predictive Maintenance Engine

**Production-ready time-series ML system** for predicting catastrophic machine failures **24 hours in advance** using sensor telemetry (temperature, vibration, pressure).

---

## 🎯 Key Results

| Metric | Target | Achieved |
|--------|--------|----------|
| **PR-AUC** | > 0.90 | **0.9334** |
| **Precision** | > 0.70 | **0.74** |
| **Recall** | > 0.90 | **0.93** |
| **Inference Latency** | < 50 ms | **5.4 ms** |
| **False Alarm Rate** | < 1% | **0.32%** |
| **Missed Failures** | < 10% | **7.0%** |

**Model:** LightGBM with class weighting  
**Dataset:** 1,080,000 sensor readings (500 machines × 90 days)  
**Failure Rate:** 0.94% (highly imbalanced)

---

## 🏗️ Architecture

```
Data Pipeline          ML Training             Deployment
─────────────────      ─────────────────       ─────────────────
sensor_data.csv  →     Feature Eng.      →     Flask REST API
                       (rolling windows)       (< 10 ms latency)
↓                      ↓                       ↓
Data Validation  →     Temporal Split    →     Real-time Stream
                       (no leakage)            (history buffer)
↓                      ↓                       ↓
Missing Values   →     LightGBM + GridCV →     SHAP Explainer
(forward fill)         (PR-AUC optimized)      (human alerts)
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Generate Training Data
```powershell
python generate_data.py
```
**Output:** `data/sensor_data.csv` (1.08M rows, 0.94% failure rate)

### 3. Train All Models
```powershell
python train.py
```
**Duration:** ~6 minutes  
**Outputs:**
- `models/factoryguard_latest.pkl` (versioned model + scaler)
- `outputs/confusion_matrix_*.png` (3 models)
- `outputs/pr_curve_*.png`
- `outputs/shap_global_importance.png`
- `outputs/shap_local_explanation.png`
- `outputs/failure_explanation.txt` (human-readable alert)
- `outputs/model_comparison.csv`

**Add `--tune` for hyperparameter optimization** (slower but +2% PR-AUC):
```powershell
python train.py --tune
```

### 4. Launch Web Application
```powershell
python serve.py
```
**🌐 Open in browser: http://localhost:5000/**

**Features:**
- Beautiful gradient UI with real-time predictions
- Color-coded risk levels (GREEN/YELLOW/RED)
- Mobile-responsive design
- <10ms inference latency displayed
- No page reloads (fetch API)

See [UI_GUIDE.md](UI_GUIDE.md) for detailed web interface documentation.

### 5. Test the System

**Test Web UI:**
```powershell
python test_ui.py
```

**Test REST API:**
```powershell
python test_api.py
```

**Expected Output:**
```
[1] GET /health → 200
[2] POST /predict (normal)  → p_fail=0.0003 | LOW
[3] POST /predict (risky)   → p_fail=0.8198 | CRITICAL
[4] POST /predict (invalid) → 400 Missing fields
[5] GET /model-info → 30 features
[6] Latency: Avg 5.4 ms | PASS
```

### 6. Real-Time Streaming Demo
```powershell
python predict_realtime.py
```
Simulates 50 hourly sensor readings with gradual degradation → CRITICAL alerts.

---

## 📊 Model Performance

### Confusion Matrix (Test Set = 215,000 samples)

|                  | Predicted Normal | Predicted Failure |
|------------------|------------------|-------------------|
| **Actual Normal**   | 212,189 (TN)     | 687 (FP)          |
| **Actual Failure**  | 149 (FN)         | 1,975 (TP)        |

- **False Positive Rate:** 0.32% (687 false alarms out of 212,876 normal readings)
- **False Negative Rate:** 7.0% (149 missed failures out of 2,124 actual failures)

**Tradeoff:** The model is tuned to favor **high recall** (catch 93% of failures) at the cost of occasional false alarms. In production, false alarms trigger preventive inspections (low cost), while missed failures cause catastrophic downtime (high cost).

---

## 🔬 Feature Importance (Top 10 by SHAP)

| Feature | SHAP Importance |
|---------|----------------|
| `pressure_rolling_mean_12h` | 0.3090 |
| `temperature_rolling_std_12h` | 0.2298 |
| `vibration_rolling_std_12h` | 0.2124 |
| `vibration_rolling_mean_12h` | 0.2030 |
| `temperature_rolling_std_6h` | 0.1934 |
| `pressure_rolling_std_12h` | 0.1917 |
| `vibration_rolling_mean_1h` | 0.1726 |
| `vibration_rolling_std_6h` | 0.1476 |
| `vibration_ema_12h` | 0.1458 |
| `temperature_rolling_mean_12h` | 0.1352 |

**Key Insight:** Pressure drops and vibration/temperature volatility (std dev) are the strongest failure predictors.

---

## 🧠 Explainability Example

**Local explanation for highest-risk prediction:**

```
FAILURE PREDICTION EXPLANATION:
The model predicted high failure risk because:
  • vibration_rolling_mean_1h = 8.91 (increased failure risk by 2.14)
  • temperature_lag_1 = 6.76 (increased failure risk by 1.74)
  • vibration_lag_1 = 8.39 (increased failure risk by 1.57)
  • temperature_rolling_mean_1h = 9.83 (increased failure risk by 1.56)
  • pressure_diff_1 = -4.09 (increased failure risk by 1.38)
```

This is saved to `outputs/failure_explanation.txt` for maintenance engineers.

---

## 📡 API Endpoints

### **GET /** — Web UI Interface
Open in browser: **http://localhost:5000/**

Beautiful web interface with:
- Real-time sensor input forms
- Instant failure predictions
- Color-coded risk levels (LOW/MEDIUM/HIGH/CRITICAL)
- Responsive design for all devices

### **POST /predict** — Prediction API

**Request:**
```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
  "temperature": 89.0,
  "vibration": 8.5,
  "pressure": 3.2
}
```

**Response:**
```json
{
  "failure_probability": 0.8198,
  "risk_level": "CRITICAL",
  "prediction": 1,
  "response_time_ms": 4.2
}
```

**Risk Levels:**
- `LOW`: p < 0.25 (normal operation)
- `MEDIUM`: 0.25 ≤ p < 0.50
- `HIGH`: 0.50 ≤ p < 0.80 (schedule inspection)
- `CRITICAL`: p ≥ 0.80 (immediate shutdown)

---

## 📂 Project Structure

```
Safe Guard AI/
├── generate_data.py          # Synthetic sensor data generator
├── train.py                  # End-to-end ML pipeline (Steps 1-8)
├── serve.py                  # Flask API server
├── predict_realtime.py       # Streaming inference demo
├── test_api.py               # API test suite
├── requirements.txt          # Python dependencies
│
├── src/
│   ├── data_loader.py        # Step 1: Load & validate CSV
│   ├── feature_engineering.py# Step 2: Rolling window features
│   ├── splitting.py          # Step 3: Time-aware train/test split
│   ├── models.py             # Steps 4-5: LR, RF, LightGBM, XGBoost
│   ├── evaluation.py         # Step 6: PR-AUC, confusion matrix, plots
│   └── explainability.py     # Step 7: SHAP global + local explanations
│
├── api/
│   └── app.py                # Step 9: Flask /predict endpoint
│
├── data/
│   └── sensor_data.csv       # Generated training data
│
├── models/
│   └── factoryguard_latest.pkl  # Versioned model + scaler
│
└── outputs/
    ├── confusion_matrix_*.png
    ├── pr_curve_*.png
    ├── shap_global_importance.png
    ├── shap_local_explanation.png
    ├── failure_explanation.txt
    ├── model_comparison.csv
    └── training.log
```

---

## 🛠️ ML Pipeline Details

### Step 1: Data Loading & Validation
- Sort by `machine_id` + `timestamp` (prevent future leakage)
- Forward/backward fill missing values (2% NaN rate)
- Sanity-bound clipping (e.g., vibration ∈ [0, 50] mm/s)

### Step 2: Feature Engineering (30 features)
**Per sensor (temp, vibration, pressure):**
- Rolling mean: 1h, 6h, 12h
- Rolling std: 1h, 6h, 12h
- Exponential moving average (span=12h)
- Lag features: t-1, t-2
- First derivative (rate of change)

**Critical:** All windows are **backward-looking only**.

### Step 3: Temporal Split
- **Train:** 80% (earliest data)
- **Test:** 20% (most recent data)
- **NO random shuffling** to prevent leakage

### Step 4-5: Model Training
1. **Logistic Regression** (baseline)
2. **Random Forest** (baseline)
3. **LightGBM** (production) with `scale_pos_weight=106.3`

**Optimization metric:** PR-AUC (NOT accuracy)

### Step 6: Evaluation
- Precision-Recall curve (not ROC, since data is imbalanced)
- Confusion matrix with FPR/FNR analysis
- Model comparison table

### Step 7: SHAP Explainability
- Global feature importance (beeswarm plot)
- Local waterfall plot for highest-risk sample
- Human-readable text explanation

### Step 8: Model Versioning
- Saved via `joblib` with timestamp versioning
- Bundles: model + scaler + feature names + metadata

### Step 9: Deployment
- Flask REST API with <10ms inference time
- Thread-safe, production-ready
- Health checks + model introspection

---

## 🔒 Production Considerations

### Current Limitations (Dev Server)
- **Werkzeug dev server** (single-threaded)
- **Stateless API** (no rolling history — features use current reading only)

### Production Recommendations
1. **WSGI Server:** Deploy with `gunicorn` or `uwsgi`
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 serve:app
   ```

2. **Time-Series Store:** Integrate Redis/InfluxDB for rolling feature computation
   - Store last 24h of readings per machine
   - Compute accurate rolling means/stds on each prediction

3. **Monitoring:**
   - Log all predictions to database
   - Alert on model drift (feature distribution shifts)
   - A/B test new model versions

4. **Edge Deployment:** Export to ONNX for edge inference (no Python runtime)

---

## 🧪 Testing

Run the full test suite:
```powershell
python test_api.py
```

**Tests:**
1. Health check
2. Normal reading → LOW risk
3. Degraded reading → CRITICAL risk
4. Invalid input → 400 error
5. Model metadata
6. Latency benchmark (20 requests)

**Pass Criteria:** All 6 tests pass, latency < 50 ms

---

## 📈 Model Comparison

| Model | PR-AUC | Precision | Recall | F1 |
|-------|--------|-----------|--------|-----|
| **Random Forest** | 0.9338 | **0.9584** | 0.8795 | **0.9173** |
| **LightGBM** | **0.9334** | 0.7419 | **0.9298** | 0.8253 |
| Logistic Regression | 0.9026 | 0.5207 | 0.9360 | 0.6691 |

**Winner:** LightGBM (best recall for failure detection, acceptable precision)

---

## 📝 License

Copyright (c) 2026 — Educational/Research Use

---

## 🤝 Contributing

This is a production-ready ML engineering template. Key features:
- ✅ **No data leakage** (temporal split, backward-only features)
- ✅ **Class imbalance handling** (scale_pos_weight, PR-AUC)
- ✅ **Production code quality** (logging, error handling, versioning)
- ✅ **Explainability** (SHAP for trust)
- ✅ **Low latency** (5ms inference)
- ✅ **Modular design** (easy to swap models/features)

---

## 📚 References

- **Dataset:** Synthetic (generator simulates realistic degradation patterns)
- **Metrics:** PR-AUC preferred over ROC-AUC for imbalanced data
- **SHAP:** Lundberg & Lee (2017) — Unified approach to interpreting model predictions
- **LightGBM:** Ke et al. (2017) — Gradient boosting decision tree

---

**Built with:** Python 3.12, LightGBM, SHAP, Flask, Scikit-learn, Pandas, NumPy
"# IoT_Predictive_Engine" 
