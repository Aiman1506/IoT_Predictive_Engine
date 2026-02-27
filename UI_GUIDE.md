# FactoryGuard AI - Web UI Quick Start Guide

## 🌐 Web Interface Now Live!

The Flask API now includes a **beautiful web UI** for testing the predictive maintenance model directly in your browser.

---

## 🚀 Launch the Application

### 1. Start the Server
```powershell
python serve.py
```

**Server will start at:**
- 🌐 **http://localhost:5000**
- 🌐 **http://127.0.0.1:5000**

### 2. Open in Browser
Navigate to: **http://localhost:5000/**

---

## 🎨 Web UI Features

### **Clean, Modern Interface**
- 🎯 Gradient purple background
- 📱 Mobile-responsive design
- ⚡ Real-time predictions (no page reload)
- 🎭 Animated results with color-coded risk levels

### **Input Fields**
- **Temperature** (°C) — Default: 55.0
- **Vibration** (mm/s) — Default: 2.0  
- **Pressure** (bar) — Default: 6.0

### **Risk Level Indicators**

| Color | Risk Level | Probability | Action |
|-------|-----------|-------------|--------|
| 🟢 Green | **LOW** | < 25% | Normal operation |
| 🟡 Yellow | **MEDIUM** | 25% - 50% | Monitor closely |
| 🔴 Red | **HIGH** | 50% - 80% | Schedule maintenance |
| 🔴 Dark Red | **CRITICAL** | ≥ 80% | Immediate shutdown |

---

## 🧪 Test Scenarios

### **Normal Operation**
```
Temperature: 55.0 °C
Vibration: 2.0 mm/s
Pressure: 6.0 bar
Expected: LOW risk (< 1%)
```

### **Critical Failure**
```
Temperature: 89.0 °C
Vibration: 8.5 mm/s
Pressure: 3.2 bar
Expected: CRITICAL risk (~82%)
```

### **Gradual Degradation**
```
Temperature: 70.0 °C
Vibration: 4.5 mm/s
Pressure: 4.8 bar
Expected: HIGH risk (~60-75%)
```

---

## 📡 API Endpoints

### 1. **GET /** — Web UI
Serves the interactive frontend interface.

### 2. **POST /predict** — Prediction API
```json
// Request
{
  "temperature": 55.0,
  "vibration": 2.0,
  "pressure": 6.0
}

// Response
{
  "failure_probability": 0.0003,
  "risk_level": "LOW",
  "prediction": 0,
  "response_time_ms": 5.4
}
```

### 3. **GET /health** — Server Status
```json
{
  "status": "healthy",
  "model_name": "lgbm",
  "model_version": "20260226_153114"
}
```

### 4. **GET /model-info** — Model Metadata
```json
{
  "model_name": "lgbm",
  "version": "20260226_153114",
  "description": "FactoryGuard AI - IoT Predictive Maintenance",
  "n_features": 30,
  "feature_names": ["temperature_rolling_mean_1h", ...]
}
```

---

## 🧪 Automated Testing

### Test the Web UI
```powershell
python test_ui.py
```

**Expected Output:**
```
✓ Title present
✓ Form present
✓ Temperature input
✓ Vibration input
✓ Pressure input
✓ Predict button
✓ Fetch API

✅ Frontend UI: PASS
✅ Prediction API: PASS
✅ High-risk detection: PASS

🎉 All Tests Passed!
```

### Test the REST API
```powershell
python test_api.py
```

---

## 🎯 How It Works

### **Frontend Flow**
1. User enters sensor values
2. JavaScript `fetch()` calls `/predict` endpoint
3. Flask processes request → LightGBM inference
4. JSON response returned (<10ms)
5. UI updates with animated result card

### **Visual Feedback**
- **Loading spinner** during prediction
- **Color-coded result card** (green/yellow/red)
- **Risk badge** (LOW/MEDIUM/HIGH/CRITICAL)
- **Latency display** in footer
- **Smooth animations** for professional UX

---

## 📂 Project Files

```
Safe Guard AI/
├── serve.py                    # Server entry point
├── test_ui.py                  # Web UI test suite
├── test_api.py                 # API test suite
│
├── api/
│   ├── app.py                  # Flask application
│   └── templates/
│       └── index.html          # Web UI (HTML + CSS + JS)
│
└── models/
    └── factoryguard_latest.pkl # Trained model
```

---

## 🔧 Tech Stack

### **Backend**
- Flask 3.0
- LightGBM (model)
- Joblib (serialization)
- NumPy (preprocessing)

### **Frontend**
- Pure HTML5
- CSS3 (gradients, animations, flexbox)
- Vanilla JavaScript (fetch API)
- No external dependencies!

### **Styling**
- **Gradient background**: Purple (#667eea → #764ba2)
- **Card design**: White with shadow/border-radius
- **Responsive**: Works on mobile/tablet/desktop
- **Animations**: Fade-in, slide-in, spin

---

## 🚀 Next Steps

### **Production Deployment**

1. **Use Production WSGI Server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 serve:app
   ```

2. **Add HTTPS with Nginx**
   ```nginx
   server {
       listen 443 ssl;
       server_name factoryguard.ai;
       
       location / {
           proxy_pass http://localhost:5000;
       }
   }
   ```

3. **Docker Containerization**
   ```dockerfile
   FROM python:3.12
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "serve:app"]
   ```

4. **Real-Time Integration**
   - WebSocket for live sensor streaming
   - Redis for rolling history buffer
   - Grafana for dashboard visualization

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Backend Response** | 5.4 ms |
| **Frontend Render** | < 100 ms |
| **Total User Experience** | < 150 ms |
| **HTML Size** | 13.6 KB (gzips to ~4 KB) |
| **Zero Dependencies** | Pure vanilla stack |

---

## 🎉 Success!

Your **production-ready IoT Predictive Maintenance system** now has:

✅ **Trained ML model** (93.3% PR-AUC)  
✅ **REST API** (<10ms latency)  
✅ **Beautiful Web UI** (mobile-responsive)  
✅ **Real-time predictions** (fetch API)  
✅ **Comprehensive tests** (UI + API)  
✅ **SHAP explainability** (model interpretability)  
✅ **Complete documentation** (README + guides)

---

**🌐 Access the UI:** http://localhost:5000/  
**📖 Full Docs:** [README.md](README.md)  
**🧪 Run Tests:** `python test_ui.py`
