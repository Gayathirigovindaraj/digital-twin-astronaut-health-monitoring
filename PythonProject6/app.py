from flask import Flask, send_file
from flask_cors import CORS
import numpy as np
import os
from ml_models import HealthAIEngine
from flask import jsonify, request

app = Flask(__name__)
CORS(app)

engine = HealthAIEngine()
engine.train()

ASTRONAUTS = {
    "patel":  {"name": "Cmdr. Patel",  "age": 42, "mission_day": 47, "base": {"hr":68,"spo2":98,"bp":115,"temp":36.9,"rr":14,"stress":1.9}},
    "okafor": {"name": "Dr. Okafor",   "age": 36, "mission_day": 39, "base": {"hr":74,"spo2":97,"bp":122,"temp":37.1,"rr":16,"stress":2.4}},
    "reyes":  {"name": "Lt. Reyes",    "age": 29, "mission_day": 52, "base": {"hr":70,"spo2":98,"bp":118,"temp":37.0,"rr":15,"stress":2.0}},
}
SCENARIO_DELTAS = {
    "normal":  {"hr":0,  "spo2":0,  "bp":0,  "temp":0.0, "rr":0, "stress":0.0},
    "eva":     {"hr":30, "spo2":-2, "bp":20, "temp":0.8, "rr":8, "stress":4.0},
    "stress":  {"hr":18, "spo2":0,  "bp":14, "temp":0.3, "rr":4, "stress":5.0},
    "hypoxia": {"hr":12, "spo2":-7, "bp":8,  "temp":0.0, "rr":6, "stress":2.0},
    "anomaly": {"hr":38, "spo2":-1, "bp":28, "temp":0.2, "rr":5, "stress":3.0},
}

def generate_vitals(astronaut_id, scenario):
    base = ASTRONAUTS[astronaut_id]["base"]
    d = SCENARIO_DELTAS.get(scenario, SCENARIO_DELTAS["normal"])
    return {
        "hr":     int(np.clip(base["hr"]    + d["hr"]    + np.random.uniform(-3,3),   30, 200)),
        "spo2":   int(np.clip(base["spo2"]  + d["spo2"]  + np.random.uniform(-1,1),   70, 100)),
        "bp":     int(np.clip(base["bp"]    + d["bp"]    + np.random.uniform(-4,4),   70, 200)),
        "temp":   round(base["temp"] + d["temp"] + np.random.uniform(-0.1,0.1), 1),
        "rr":     int(np.clip(base["rr"]    + d["rr"]    + np.random.uniform(-1,1),    8,  40)),
        "stress": round(float(np.clip(base["stress"] + d["stress"] + np.random.uniform(-0.2,0.2), 0, 10)), 1),
    }

@app.route('/')
def index():
    # Build exact path to index.html
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'index.html')
    print(f"Serving: {html_path}")
    print(f"Exists: {os.path.exists(html_path)}")
    return send_file(html_path)

@app.route("/api/vitals")
def get_vitals():
    astronaut_id = request.args.get("astronaut", "patel")
    scenario     = request.args.get("scenario",  "normal")
    if astronaut_id not in ASTRONAUTS:
        return jsonify({"error": "Unknown astronaut"}), 400
    vitals    = generate_vitals(astronaut_id, scenario)
    ai_result = engine.analyze(vitals)
    ast_info  = ASTRONAUTS[astronaut_id]
    return jsonify({
        "astronaut":   ast_info["name"],
        "mission_day": ast_info["mission_day"],
        "age":         ast_info["age"],
        "scenario":    scenario,
        "vitals":      vitals,
        "ai": {
            "health_score":    ai_result["health_score"],
            "anomaly_prob":    ai_result["anomaly_prob"],
            "status":          ai_result["status"],
            "class_probs":     ai_result.get("class_probs", {}),
            "prediction_2h":   ai_result["prediction_2h"],
            "alerts":          ai_result["alerts"],
            "recommendations": ai_result["recommendations"],
        }
    })

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'index.html')
    print("=" * 50)
    print(f"index.html path : {path}")
    print(f"index.html found: {os.path.exists(path)}")
    print("Open: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)