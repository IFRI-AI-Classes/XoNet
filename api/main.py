"""
XoNet API — Backend FastAPI
Endpoints :
  POST /translate  — lookup dans corpus_final.csv (fon→fr ou fr→fon)
  POST /sentiment  — classification HuggingFace sur texte français
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
CORPUS_CSV = BASE_DIR / "data" / "corpus_final.csv"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="XoNet API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restreindre en production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Corpus (chargé une seule fois) ───────────────────────────────────────────
@lru_cache(maxsize=1)
def load_corpus() -> tuple[dict[str, str], dict[str, str]]:
    """
    Retourne deux dictionnaires :
      fon_to_fr : { texte_fongbé : traduction_française }
      fr_to_fon : { texte_français : texte_fongbé }
    """
    fon_to_fr: dict[str, str] = {}
    fr_to_fon: dict[str, str] = {}

    if not CORPUS_CSV.exists():
        return fon_to_fr, fr_to_fon

    with open(CORPUS_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            fon = row.get("fon", "").strip()
            fr  = row.get("fr",  "").strip()
            if fon and fr:
                fon_to_fr.setdefault(fon, fr)
                fr_to_fon.setdefault(fr, fon)

    return fon_to_fr, fr_to_fon


# ── Classifieur de sentiment (chargé à la demande) ───────────────────────────
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )
        model_name = "philschmid/pt-tblard-tf-allocine"
        tokenizer  = AutoTokenizer.from_pretrained(model_name)
        model      = AutoModelForSequenceClassification.from_pretrained(model_name)
        _classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return _classifier


# ── Schémas Pydantic ──────────────────────────────────────────────────────────
class TranslateRequest(BaseModel):
    text: str
    direction: Literal["fon-fr", "fr-fon"] = "fon-fr"

class TranslateResponse(BaseModel):
    translation: str
    found: bool          # True = correspondance exacte dans le corpus


class SentimentRequest(BaseModel):
    text: str            # phrase en français

class SentimentResponse(BaseModel):
    label: Literal["positif", "neutre", "negatif"]
    scores: dict[str, float]   # { "positif": 0.78, "neutre": 0.14, "negatif": 0.08 }
    raw_label: str             # label brut du modèle (POSITIVE / NEGATIVE)
    raw_score: float


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "project": "XoNet", "version": "0.1.0"}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Le champ 'text' est vide.")

    fon_to_fr, fr_to_fon = load_corpus()

    if req.direction == "fon-fr":
        result = fon_to_fr.get(text)
    else:
        result = fr_to_fon.get(text)

    if result:
        return TranslateResponse(translation=result, found=True)

    # Fallback : recherche insensible à la casse
    text_lower = text.lower()
    lookup = fon_to_fr if req.direction == "fon-fr" else fr_to_fon
    for key, val in lookup.items():
        if key.lower() == text_lower:
            return TranslateResponse(translation=val, found=True)

    return TranslateResponse(
        translation="Traduction non trouvée dans le corpus XoNet.",
        found=False,
    )


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(req: SentimentRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Le champ 'text' est vide.")

    clf    = get_classifier()
    result = clf(text, truncation=True)[0]

    raw_label = result["label"].upper()   # "POSITIVE" ou "NEGATIVE"
    raw_score = float(result["score"])
    SEUIL     = 0.70

    # Conversion en 3 classes (logique du notebook classification.ipynb)
    if raw_label == "POSITIVE" and raw_score > SEUIL:
        label = "positif"
        scores = {"positif": raw_score, "neutre": round(1 - raw_score, 4), "negatif": 0.0}
    elif raw_label == "NEGATIVE" and raw_score > SEUIL:
        label = "negatif"
        scores = {"negatif": raw_score, "neutre": round(1 - raw_score, 4), "positif": 0.0}
    else:
        label = "neutre"
        # Score neutre = 1 − |score_signé| pour rester cohérent avec l'affichage
        complement = round(1 - raw_score, 4)
        if raw_label == "POSITIVE":
            scores = {"neutre": raw_score, "positif": complement, "negatif": 0.0}
        else:
            scores = {"neutre": raw_score, "negatif": complement, "positif": 0.0}

    return SentimentResponse(
        label=label,
        scores=scores,
        raw_label=raw_label,
        raw_score=raw_score,
    )
