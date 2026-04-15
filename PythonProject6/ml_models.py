"""
ml_models.py
AI Health Engine for Digital Twin
- Isolation Forest: anomaly detection (unsupervised)
- Random Forest: health status classification (supervised)
- Linear Regression: 2-hour prediction
"""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class HealthAIEngine:
    """
    Core AI engine that powers the Digital Twin.
    Trains on synthetic 'normal' astronaut health data.
    """

    FEATURE_NAMES = ["hr", "spo2", "bp", "temp", "rr", "stress"]

    # Status labels
    STATUS_LABELS = {0: "Stable", 1: "Monitor", 2: "Critical"}

    def __init__(self):
        self.iso_forest = None
        self.rf_classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False

    # ------------------------------------------------------------------
    # DATA GENERATION
    # ------------------------------------------------------------------
    def _generate_training_data(self):
        """
        Synthesize training data for three health states:
        0 = Stable, 1 = Monitor, 2 = Critical
        Replaces real IoT sensor data with realistic distributions.
        """
        rng = np.random.RandomState(42)
        N = 1000

        def sample(hr, spo2, bp, temp, rr, stress, n=N, noise=1.0):
            return np.column_stack([
                rng.normal(hr,    3*noise,   n),
                rng.normal(spo2,  1*noise,   n),
                rng.normal(bp,    5*noise,   n),
                rng.normal(temp,  0.2*noise, n),
                rng.normal(rr,    2*noise,   n),
                rng.normal(stress,0.5*noise, n),
            ])

        # Stable (normal astronaut vitals in microgravity)
        stable   = sample(hr=70,  spo2=97, bp=118, temp=37.0, rr=15, stress=2.0)
        labels_s = np.zeros(N, dtype=int)

        # Monitor (mildly elevated — EVA fatigue, mild stress)
        monitor  = sample(hr=95,  spo2=94, bp=130, temp=37.5, rr=20, stress=5.0, noise=1.5)
        labels_m = np.ones(N, dtype=int)

        # Critical (dangerous — cardiac event, hypoxia, severe stress)
        critical = sample(hr=115, spo2=88, bp=150, temp=38.2, rr=26, stress=8.0, noise=2.0)
        labels_c = np.full(N, 2, dtype=int)

        X = np.vstack([stable, monitor, critical])
        y = np.concatenate([labels_s, labels_m, labels_c])

        # Clip to realistic physiological ranges
        X[:, 0] = np.clip(X[:, 0], 30,  200)   # HR
        X[:, 1] = np.clip(X[:, 1], 70,  100)   # SpO2
        X[:, 2] = np.clip(X[:, 2], 70,  200)   # BP
        X[:, 3] = np.clip(X[:, 3], 35,  42)    # Temp
        X[:, 4] = np.clip(X[:, 4], 8,   40)    # RR
        X[:, 5] = np.clip(X[:, 5], 0,   10)    # Stress

        return X, y

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    def train(self):
        print("[AI Engine] Generating synthetic training data...")
        X, y = self._generate_training_data()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # 1. Isolation Forest — trained only on stable data
        stable_X = X_scaled[y == 0]
        self.iso_forest = IsolationForest(
            n_estimators=150,
            contamination=0.05,
            random_state=42
        )
        self.iso_forest.fit(stable_X)
        print("[AI Engine] Isolation Forest trained on stable vitals.")

        # 2. Random Forest Classifier — multi-class (0,1,2)
        self.rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )
        self.rf_classifier.fit(X_scaled, y)
        print("[AI Engine] Random Forest Classifier trained.")

        self.is_trained = True
        print("[AI Engine] Training complete. Digital Twin AI ready.")

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------
    def _vitals_to_vector(self, vitals: dict) -> np.ndarray:
        return np.array([[
            vitals["hr"],
            vitals["spo2"],
            vitals["bp"],
            vitals["temp"],
            vitals["rr"],
            vitals["stress"],
        ]])

    def _anomaly_probability(self, x_scaled: np.ndarray) -> float:
        """
        Convert Isolation Forest score to anomaly probability (0-100%).
        Scores closer to -1 are more anomalous.
        """
        score = self.iso_forest.score_samples(x_scaled)[0]
        # score range roughly [-0.7, 0.1]; map to [0, 100]
        prob = np.clip((score + 0.7) / 0.8, 0, 1)
        prob = 1 - prob  # invert: higher = more anomalous
        return round(float(prob * 100), 1)

    def _rule_based_alerts(self, vitals: dict):
        """
        Deterministic alert rules — acts as the clinical knowledge layer.
        Complements ML predictions with interpretable logic.
        """
        alerts = []
        recommendations = []

        hr = vitals["hr"]
        spo2 = vitals["spo2"]
        bp = vitals["bp"]
        temp = vitals["temp"]
        rr = vitals["rr"]
        stress = vitals["stress"]

        if hr > 110:
            alerts.append({"level": "danger", "message": "Heart rate critically high — tachycardia"})
            recommendations.append("Immediate rest and cardiac monitoring required")
        elif hr > 95:
            alerts.append({"level": "warning", "message": "Heart rate elevated above mission threshold"})
            recommendations.append("Reduce physical activity for 30 minutes")
        elif hr < 45:
            alerts.append({"level": "danger", "message": "Bradycardia detected — HR dangerously low"})
            recommendations.append("Medical officer review required immediately")

        if spo2 < 90:
            alerts.append({"level": "danger", "message": "SpO2 critically low — hypoxia risk"})
            recommendations.append("Supplemental oxygen and cabin pressure check")
        elif spo2 < 94:
            alerts.append({"level": "warning", "message": "SpO2 below optimal — monitor closely"})
            recommendations.append("Check O2 supply and reduce exertion")

        if bp > 140:
            alerts.append({"level": "danger", "message": "Systolic BP dangerously high"})
            recommendations.append("Anti-hypertensive protocol initiated")
        elif bp > 128:
            alerts.append({"level": "warning", "message": "Blood pressure elevated"})
            recommendations.append("Hydrate and avoid strenuous tasks")

        if temp > 38.0:
            alerts.append({"level": "danger", "message": "Core temperature high — fever indicated"})
            recommendations.append("Antipyretic medication and medical evaluation")
        elif temp > 37.6:
            alerts.append({"level": "warning", "message": "Mild temperature elevation detected"})
            recommendations.append("Increase fluid intake, monitor every 15 min")

        if rr > 22:
            alerts.append({"level": "warning", "message": "Respiratory rate elevated"})
            recommendations.append("Controlled breathing exercises recommended")

        if stress > 7:
            alerts.append({"level": "danger", "message": "Extreme cognitive stress — cortisol critical"})
            recommendations.append("Mandatory rest cycle and psychological check-in")
        elif stress > 4.5:
            alerts.append({"level": "warning", "message": "Stress index elevated"})
            recommendations.append("Schedule 20-min mindfulness session")

        if not alerts:
            alerts.append({"level": "ok", "message": "All vitals within mission-safe parameters"})
            recommendations.append("Continue current mission activities")

        return alerts, recommendations

    def _health_score(self, vitals: dict, status_class: int) -> int:
        """
        Composite health score 0-100 based on vital deviations and ML class.
        """
        score = 100

        # Deduct based on vital deviations from ideal
        deductions = {
            "hr":     (abs(vitals["hr"]    - 70)  / 70)  * 20,
            "spo2":   (max(0, 98 - vitals["spo2"]) / 8)  * 20,
            "bp":     (max(0, vitals["bp"] - 120)  / 80) * 20,
            "temp":   (max(0, vitals["temp"] - 37) / 2)  * 15,
            "rr":     (max(0, vitals["rr"]  - 16)  / 20) * 10,
            "stress": (vitals["stress"] / 10)             * 15,
        }
        for d in deductions.values():
            score -= d

        # ML penalty
        if status_class == 1:
            score -= 10
        elif status_class == 2:
            score -= 25

        return max(0, min(100, int(score)))

    def analyze(self, vitals: dict) -> dict:
        """Main inference method called by the API."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        x = self._vitals_to_vector(vitals)
        x_scaled = self.scaler.transform(x)

        # Isolation Forest — anomaly probability
        anomaly_prob = self._anomaly_probability(x_scaled)

        # Random Forest — status class + probabilities
        status_class = int(self.rf_classifier.predict(x_scaled)[0])
        class_probs  = self.rf_classifier.predict_proba(x_scaled)[0]

        # Confidence-weighted prediction for 2 hours
        if class_probs[0] > 0.6:
            prediction_2h = "Stable"
        elif class_probs[2] > 0.3:
            prediction_2h = "Intervention likely needed"
        else:
            prediction_2h = "Close monitoring advised"

        # Rule-based alerts and recommendations
        alerts, recommendations = self._rule_based_alerts(vitals)

        # Health score
        health_score = self._health_score(vitals, status_class)

        return {
            "health_score":    health_score,
            "anomaly_prob":    anomaly_prob,
            "status":          self.STATUS_LABELS[status_class],
            "class_probs":     {
                "stable":   round(float(class_probs[0]) * 100, 1),
                "monitor":  round(float(class_probs[1]) * 100, 1),
                "critical": round(float(class_probs[2]) * 100, 1),
            },
            "prediction_2h":   prediction_2h,
            "alerts":          alerts,
            "recommendations": recommendations,
        }
