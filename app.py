import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

tfidf = None
model = None
sentiment_pipeline = None

# Changer le nom ici pour utiliser un autre modèle :
#   "mnb_model.joblib"    → Multinomial Naive Bayes
#   "rf_model.joblib"     → Random Forest
#   "voting_model.joblib" → Soft Voting Classifier (par défaut)
MODEL_FILENAME = os.getenv("XONET_MODEL_FILENAME", "voting_model.joblib")

# Default to the tuned v3 pipeline. Override with a custom filename or an empty
# env var to force legacy mode,
# or set a custom pipeline filename.
# PowerShell:
#   $env:XONET_PIPELINE_FILENAME="sentiment_pipeline_v3_tuned.joblib"
#   $env:XONET_PIPELINE_FILENAME=""
PIPELINE_FILENAME = os.getenv(
    "XONET_PIPELINE_FILENAME", "sentiment_pipeline_v3_tuned.joblib"
).strip()

def load_model():
    global tfidf, model, sentiment_pipeline

    if PIPELINE_FILENAME:
        pipeline_path = os.path.join(MODELS_DIR, PIPELINE_FILENAME)
        if os.path.exists(pipeline_path):
            sentiment_pipeline = joblib.load(pipeline_path)
            model = sentiment_pipeline
            tfidf = None
            print(f"Pipeline ({PIPELINE_FILENAME}) loaded successfully.")
            return
        print("Requested pipeline not found in", MODELS_DIR)

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
    model_missing = (
        sentiment_pipeline is None
        if PIPELINE_FILENAME
        else tfidf is None or model is None
    )
    if model_missing:
        load_model()
        model_missing = (
            sentiment_pipeline is None
            if PIPELINE_FILENAME
            else tfidf is None or model is None
        )
        if model_missing:
            return jsonify({"error": "Modele non entraine/exporte."}), 500

    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Veuillez fournir un texte en fon."}), 400

    try:
        if sentiment_pipeline is not None:
            proba_input = [text]
            probability_model = sentiment_pipeline
            pred_label = int(sentiment_pipeline.predict(proba_input)[0])
        else:
            X = tfidf.transform([text])
            proba_input = X
            probability_model = model
            pred_label = int(model.predict(X)[0])

        probs = {}
        confidence = None
        if hasattr(probability_model, "predict_proba"):
            probabilities = probability_model.predict_proba(proba_input)[0]
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
