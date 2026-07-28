import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

tfidf = None
model = None

# Changer le nom ici pour utiliser un autre modèle :
#   "mnb_model.joblib"    → Multinomial Naive Bayes
#   "rf_model.joblib"     → Random Forest
#   "voting_model.joblib" → Soft Voting Classifier (par défaut)
MODEL_FILENAME = "rf_model.joblib"

def load_model():
    global tfidf, model
    tfidf_path = os.path.join(MODELS_DIR, "tfidf.joblib")
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)

    if os.path.exists(tfidf_path) and os.path.exists(model_path):
        tfidf = joblib.load(tfidf_path)
        model = joblib.load(model_path)
        print(f"Vectorizer et modèle ({MODEL_FILENAME}) chargés avec succès.")
    else:
        print("Fichiers modèle non trouvés dans", MODELS_DIR)

LABEL_MAP = {
    0: {"name": "Neutre", "emoji": "😐", "color": "#64748b"},
    1: {"name": "Négatif", "emoji": "🙁", "color": "#ef4444"},
    2: {"name": "Positif", "emoji": "😊", "color": "#22c55e"}
}

@app.route("/api/predict", methods=["POST"])
def predict():
    if tfidf is None or model is None:
        load_model()
        if tfidf is None or model is None:
            return jsonify({"error": "Modèle non entraîné/exporté."}), 500

    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Veuillez fournir un texte en fon."}), 400

    try:
        X = tfidf.transform([text])
        pred_label = int(model.predict(X)[0])

        probs = {}
        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            confidence = float(max(probabilities))
            probs = {
                "Neutre": float(probabilities[0]),
                "Négatif": float(probabilities[1]),
                "Positif": float(probabilities[2]),
            }

        label_info = LABEL_MAP.get(pred_label, LABEL_MAP[0])

        return jsonify({
            "text": text,
            "label_id": pred_label,
            "sentiment": label_info["name"],
            "emoji": label_info["emoji"],
            "color": label_info["color"],
            "confidence": confidence,
            "probabilities": probs
        })
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

@app.route("/")
def home():
    html_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>API XoNet OK</h1>"

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
