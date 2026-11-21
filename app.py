import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Changer le nom ici pour utiliser un autre modèle legacy :
#   "mnb_model.joblib"    → Multinomial Naive Bayes
#   "rf_model.joblib"     → Random Forest
#   "voting_model.joblib" → Soft Voting Classifier (par défaut)
MODEL_FILENAME = os.getenv("XONET_MODEL_FILENAME", "voting_model.joblib")

# Default to the tuned v3 pipeline. Override with a custom filename or an empty
# env var to force legacy mode (Soft Voting).
# PowerShell:
#   $env:XONET_PIPELINE_FILENAME="sentiment_pipeline_v3_tuned.joblib"
#   $env:XONET_PIPELINE_FILENAME=""
PIPELINE_ENV = os.getenv("XONET_PIPELINE_FILENAME")
PIPELINE_FILENAME = (
    "sentiment_pipeline_v3_tuned.joblib"
    if PIPELINE_ENV is None
    else PIPELINE_ENV.strip()
)

MODEL_SPECS = {
    "v3_tuned": {
        "type": "pipeline",
        "filename": "sentiment_pipeline_v3_tuned.joblib",
        "label": "SVM Hybride v3 (Char+Word)",
        "accuracy": "75.4%",
        "f1_macro": "70.5%",
    },
    "soft_voting": {
        "type": "legacy",
        "vectorizer": "tfidf.joblib",
        "model": MODEL_FILENAME,
        "decision_weights": None,
        "label": "Soft Voting (MNB + Random Forest)",
        "accuracy": "71.9%",
        "f1_macro": "65.2%",
    },
    "v2": {
        "type": "pipeline",
        "filename": "sentiment_pipeline_v2.joblib",
        "label": "LinearSVC Calibré v2",
        "accuracy": "72.3%",
        "f1_macro": "66.6%",
    },
}

LOADED_MODELS = {}

LABEL_MAP = {
    0: {"name": "Neutre", "emoji": "😐", "color": "#eab308"},
    1: {"name": "Négatif", "emoji": "🙁", "color": "#dc2626"},
    2: {"name": "Positif", "emoji": "😊", "color": "#16a34a"},
}


def _spec_exists(spec):
    if spec["type"] == "pipeline":
        return os.path.exists(os.path.join(MODELS_DIR, spec["filename"]))
    return (
        os.path.exists(os.path.join(MODELS_DIR, spec["vectorizer"]))
        and os.path.exists(os.path.join(MODELS_DIR, spec["model"]))
    )


def get_available_models():
    models = []
    for key, spec in MODEL_SPECS.items():
        if _spec_exists(spec):
            models.append({
                "key": key,
                "label": spec["label"],
                "accuracy": spec["accuracy"],
                "f1_macro": spec["f1_macro"],
            })
    return models


def get_default_model_key():
    if PIPELINE_ENV is not None and not PIPELINE_FILENAME:
        return "soft_voting"
    for key, spec in MODEL_SPECS.items():
        if spec.get("filename") == PIPELINE_FILENAME and _spec_exists(spec):
            return key
    for key in ("v3_tuned", "v2", "soft_voting"):
        if _spec_exists(MODEL_SPECS[key]):
            return key
    return "v3_tuned"


def _normalize_probs(probs, weights=None):
    probs = np.asarray(probs, dtype=float)
    if weights is not None and probs.shape[-1] == len(weights):
        weighted = probs * weights
        total = weighted.sum(axis=-1, keepdims=True)
        return np.divide(
            weighted,
            total,
            out=np.array(probs, dtype=float, copy=True),
            where=total > 0,
        )
    return probs


def load_model_entry(model_key):
    if model_key in LOADED_MODELS:
        return LOADED_MODELS[model_key]

    spec = MODEL_SPECS.get(model_key)
    if spec is None or not _spec_exists(spec):
        return None

    if spec["type"] == "pipeline":
        pipeline = joblib.load(os.path.join(MODELS_DIR, spec["filename"]))
        entry = {
            "key": model_key,
            "type": "pipeline",
            "label": spec["label"],
            "accuracy": spec["accuracy"],
            "f1_macro": spec["f1_macro"],
            "pipeline": pipeline,
        }
    else:
        tfidf = joblib.load(os.path.join(MODELS_DIR, spec["vectorizer"]))
        model = joblib.load(os.path.join(MODELS_DIR, spec["model"]))
        entry = {
            "key": model_key,
            "type": "legacy",
            "label": spec["label"],
            "accuracy": spec["accuracy"],
            "f1_macro": spec["f1_macro"],
            "tfidf": tfidf,
            "model": model,
            "decision_weights": spec.get("decision_weights"),
        }

    LOADED_MODELS[model_key] = entry
    print(f"Model '{model_key}' ({spec['label']}) loaded successfully.")
    return entry


def load_model():
    default_key = get_default_model_key()
    load_model_entry(default_key)
    # Preload Soft Voting too if available so conference switching is instant
    if default_key != "soft_voting" and _spec_exists(MODEL_SPECS["soft_voting"]):
        load_model_entry("soft_voting")


@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify({
        "default_model": get_default_model_key(),
        "models": get_available_models(),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({"error": "Veuillez fournir un texte en fon."}), 400

    requested_key = data.get("model_key")
    if requested_key and requested_key not in MODEL_SPECS:
        return jsonify({"error": f"Modèle inconnu : {requested_key}"}), 400

    selected_key = requested_key or get_default_model_key()
    entry = load_model_entry(selected_key)
    if entry is None:
        return jsonify({"error": "Modèle non entraîné ou introuvable."}), 500

    try:
        if entry["type"] == "pipeline":
            pipeline = entry["pipeline"]
            if hasattr(pipeline, "predict_proba"):
                probabilities = np.asarray(pipeline.predict_proba([text])[0], dtype=float)
                classes = getattr(pipeline, "classes_", np.array([0, 1, 2]))
                pred_label = int(classes[int(np.argmax(probabilities))])
            else:
                probabilities = None
                pred_label = int(pipeline.predict([text])[0])
        else:
            X = entry["tfidf"].transform([text])
            estimator = entry["model"]
            if hasattr(estimator, "predict_proba"):
                raw_probs = estimator.predict_proba(X)[0]
                probabilities = _normalize_probs(raw_probs, entry.get("decision_weights"))
                classes = getattr(estimator, "classes_", np.array([0, 1, 2]))
                pred_label = int(classes[int(np.argmax(probabilities))])
            else:
                probabilities = None
                pred_label = int(estimator.predict(X)[0])

        probs = {}
        confidence = None
        if probabilities is not None:
            confidence = float(np.max(probabilities))
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
            "probabilities": probs,
            "model_key": entry["key"],
            "model_label": entry["label"],
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
