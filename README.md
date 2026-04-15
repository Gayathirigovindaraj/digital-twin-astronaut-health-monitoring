# 🚀 Digital Twin Using AI for Astronaut Health Monitoring

> A software-only Digital Twin system for real-time astronaut health monitoring using Isolation Forest anomaly detection and Random Forest classification — no IoT hardware required.

---

## 📌 Project Overview

This project implements a **Digital Twin** — a dynamic computational model that replicates the physiological state of a specific astronaut in real time. Instead of relying on physical sensors, the system:

- **Simulates** physiologically accurate vital sign data using statistical models
- **Detects anomalies** using Isolation Forest (unsupervised ML)
- **Classifies health status** using Random Forest (supervised ML)
- **Visualizes** everything through an interactive web dashboard

Built as a Micro Project Report for:
- **Course:** E1CSC 25607 — Virtual Reality
- **Program:** M. Tech. in Artificial Intelligence and Data Science
- **Institution:** Alliance School of Advanced Computing, Alliance University

---

## 🧑‍💻 Author

| Field | Details |
|---|---|
| **Name** | G. Gayathri |
| **Roll No.** | 2511022120036 |
| **Class** | M Tech AI&DS — Semester II |
| **Faculty** | Dr. Shekar R |
| **Duration** | January 2026 – May 2026 |

---

## 🏗️ System Architecture

```
┌─────────────────┐     Vitals     ┌─────────────────┐     Score     ┌─────────────────┐     Alerts     ┌──────────────────────┐
│  Data Simulator │ ─────────────► │    AI Engine     │ ────────────► │  Alert Engine   │ ─────────────► │ Digital Twin         │
│   (NumPy)       │                │  (scikit-learn)  │               │ (Python Rules)  │                │ Dashboard            │
│                 │                │                  │               │                 │                │ (Flask / HTML5)      │
└─────────────────┘                └─────────────────┘               └─────────────────┘                └──────────────────────┘
      INPUT                             INFERENCE                          RESPONSE                            INTERFACE
```

### AI Classification Pipeline (Dual-Path)

```
                        ┌──────────────────────────┐     Anomaly Score (0–100%)  ┐
                        │  Isolation Forest         │                              │
Vital Sign   ──────────►│  Anomaly Detection        │                              ├──► Composite
Feature Vector          └──────────────────────────┘                              │    Health Score
                        ┌──────────────────────────┐     Class Probs              │    (0–100)
                        │  Random Forest            │  [Stable/Monitor/Critical]  │
                        │  Classification           │                              │
                        └──────────────────────────┘ ─────────────────────────── ┘
```

---

## ✨ Features

- ✅ **Software-only** — no physical IoT sensors or hardware needed
- ✅ **Real-time** anomaly detection using Isolation Forest
- ✅ **3-class health classification** (Stable / Monitor / Critical) using Random Forest
- ✅ **Composite Health Score** (0–100) per astronaut
- ✅ **Rule-based clinical alert engine** with colour-coded alerts
- ✅ **2-hour physiological trajectory prediction**
- ✅ **Interactive web dashboard** for 3 simultaneous crew members
- ✅ **RESTful Flask API** — `/api/vitals`, `/api/astronauts`, `/api/history`
- ✅ **Offline mode** — dashboard runs in JavaScript simulation mode without backend

---

## 📊 Model Performance

| Metric | Result | Target |
|---|---|---|
| RF Classification Accuracy (3-class) | **91.2%** | > 85% |
| Isolation Forest Anomaly Detection Rate | **88.7%** | > 80% |
| False Positive Rate | **3.1%** | < 5% |
| End-to-end Inference Latency | **< 8 ms** | < 100 ms |
| API Response Time (Flask) | **< 45 ms** | < 200 ms |
| Rule-based Alert Precision | **96.4%** | — |
| Health Score Correlation (r) | **0.94** | > 0.90 |

---

## 🧬 Monitored Vital Parameters

| Vital Sign | Normal Range | Noise (σ) |
|---|---|---|
| Heart Rate (bpm) | 65 – 80 | 3 |
| SpO2 (%) | 96 – 99 | 1 |
| Systolic BP (mmHg) | 110 – 130 | 4 |
| Core Temperature (°C) | 36.5 – 37.2 | 0.1 |
| Respiratory Rate (/min) | 12 – 18 | 1 |
| Cortisol Stress Index | 1.5 – 3.0 | 0.2 |

---

## 🛠️ Tech Stack

| Category | Tool / Library |
|---|---|
| Language | Python 3.11 |
| Web Framework | Flask 3.0 + Flask-CORS |
| AI / ML | scikit-learn 1.5 |
| Numerical Computing | NumPy 1.26 |
| Frontend | HTML5 / CSS3 / JavaScript |
| Charting | Chart.js 4.4 |
| Typography | Space Mono / DM Sans (Google Fonts) |
| IDE | PyCharm Community |
| Version Control | Git |
| Package Management | pip |

---

## 📁 Project Structure

```
digital-twin-astronaut-health-ai/
│
├── app.py                  # Flask application entry point
├── requirements.txt        # Python dependencies
│
├── models/
│   ├── isolation_forest.py # Anomaly detection model
│   ├── random_forest.py    # Health classification model
│   └── scaler.py           # StandardScaler feature preprocessing
│
├── simulation/
│   ├── data_simulator.py   # Synthetic vital sign generator
│   ├── scenarios.py        # Mission scenario delta configs
│   └── baselines.py        # Per-astronaut baseline profiles
│
├── engine/
│   ├── alert_engine.py     # Rule-based clinical alert system
│   ├── health_score.py     # Composite health score calculator
│   └── predictor.py        # 2-hour trajectory prediction
│
├── api/
│   ├── routes.py           # Flask REST API routes
│   └── serializers.py      # JSON response builders
│
├── dashboard/
│   ├── index.html          # Digital Twin web dashboard
│   ├── style.css           # Dark space-themed UI styles
│   └── dashboard.js        # Real-time chart + DOM updates
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/digital-twin-astronaut-health-ai.git
cd digital-twin-astronaut-health-ai
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask API server
```bash
python app.py
```

### 5. Open the dashboard
```
http://localhost:5000
```

---

## 📦 requirements.txt

```
flask==3.0.0
flask-cors==4.0.0
numpy==1.26.0
scikit-learn==1.5.0
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/astronauts` | List all crew members and their baseline profiles |
| `GET` | `/api/vitals?astronaut_id=1&scenario=EVA` | Get real-time AI-processed vital reading |
| `GET` | `/api/history?astronaut_id=1&limit=20` | Get last N vital readings with health scores |

### Sample API Response — `/api/vitals`

```json
{
  "astronaut": "Cmdr. Patel",
  "scenario": "EVA",
  "vitals": {
    "heart_rate": 112,
    "spo2": 94,
    "systolic_bp": 148,
    "core_temp": 37.8,
    "resp_rate": 24,
    "cortisol_index": 6.8
  },
  "health_status": "Critical",
  "health_score": 28,
  "anomaly_probability": 0.73,
  "class_probabilities": {
    "stable": 0.05,
    "monitor": 0.18,
    "critical": 0.77
  },
  "alerts": [
    "⚠️ Tachycardia detected — HR 112 bpm. Rest and ECG review recommended.",
    "🔴 Hypertension — Systolic BP 148 mmHg. Antihypertensive protocol initiated."
  ]
}
```

---

## 🎯 Mission Scenarios Simulated

| Scenario | Description |
|---|---|
| `NORMAL_REST` | Baseline resting state — all vitals within normal ranges |
| `EVA` | Extravehicular Activity — elevated HR, BP, respiration, cortisol |
| `HIGH_COGNITIVE_STRESS` | High mental load — elevated cortisol and HR |
| `MILD_HYPOXIA` | Reduced cabin oxygen — SpO2 drop, elevated respiration |
| `CARDIAC_ANOMALY` | Simulated cardiac event — severe HR, BP, cortisol spikes |

---

## 👨‍🚀 Crew Members

| Astronaut | Role | Baseline Profile |
|---|---|---|
| Cmdr. Patel | Mission Commander | Standard adult male baseline |
| Dr. Okafor | Mission Scientist | Standard adult female baseline |
| Lt. Reyes | Pilot | Athlete-adjusted baseline |

---

## 📈 Health Score Interpretation

| Score Range | Status | Dashboard Colour | Action |
|---|---|---|---|
| 75 – 100 | ✅ Stable | Green | Continue monitoring |
| 45 – 74 | ⚠️ Monitor | Amber | Reduce workload; observe |
| 0 – 44 | 🔴 Critical | Red | Immediate medical intervention |

---

## ⚠️ Limitations

- Synthetic data cannot fully replicate real spaceflight biometric distributions
- Models are static — no online learning for physiological drift adaptation
- No temporal modelling between successive readings (LSTM recommended for future)
- Flask dev server is not production-grade (use Gunicorn/uWSGI for deployment)
- Crew configuration is fixed at startup — no dynamic profile management

---

## 🔭 Future Work

- [ ] LSTM recurrent neural network for temporal trend modelling
- [ ] Federated learning for privacy-preserving multi-crew model training
- [ ] Drug interaction and radiation dose modelling in simulation engine
- [ ] PostgreSQL persistent astronaut profile database
- [ ] WebSocket real-time communication (replace polling)
- [ ] Validation against NASA Human Research Program datasets
- [ ] Cloud deployment on AWS / Google Cloud

---

## 📄 License

This project was developed as an academic Micro Project Report for Alliance University.
For academic and educational use only.

---

## 🙏 Acknowledgements

- **Faculty Guide:** Dr. Shekar R
- **Institution:** Alliance School of Advanced Computing, Alliance University
- **Program:** M. Tech. in Artificial Intelligence and Data Science
- **Course:** E1CSC 25607 — Virtual Reality

---

> *"The Digital Twin doesn't replace the astronaut's body — it understands it."*
