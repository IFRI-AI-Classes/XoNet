# =============================================================================
# Évaluation du traducteur XoNet (NLLB fine-tuné) — bidirectionnelle
#   Fongbé → Français  et  Français → Fongbé
# =============================================================================
# Ce script charge le modèle fine-tuné et mesure ses performances sur les
# 500 premières paires du test set (paires jamais vues à l'entraînement),
# dans les deux sens de traduction.
#
# Le split train/val/test est reproduit À L'IDENTIQUE de src/finetune_nllb.py :
#   - même fichier : data/cleaned_dataset_final.csv
#   - même random_state = 42
#   - mêmes ratios : 80% / 10% / 10%
#
# Métriques par sens : BLEU (sacrebleu), chrF (evaluate), Exact Match,
# longueur moyenne des prédictions vs références.
#
# Sorties :
#   - results/evaluation_results.json  → métriques globales par sens
#   - results/translations_sample.csv  → 10 exemples par sens
#
# Usage :
#   python src/evaluate_model.py                    # 500 paires, beam=5
#   python src/evaluate_model.py --samples 20       # test rapide (CPU)
#   python src/evaluate_model.py --num-beams 3      # génération plus rapide
#
# Prérequis :
#   pip install sacrebleu evaluate tqdm pandas torch transformers
# =============================================================================

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from sacrebleu.metrics import BLEU
import evaluate

# =============================================================================
# 1. CONFIGURATION & CHEMINS (toujours relatifs à src/)
# =============================================================================
# On ne modifie JAMAIS les caractères Fongbé spéciaux (ɔ, ɛ, ɖ, Ɖ,
# diacritiques tonals) : les textes sont traités tels quels.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Chemins surchargeables par variables d'environnement (utilisé sur Kaggle)
MODEL_PATH = os.environ.get(
    "XONET_MODEL_PATH",
    os.path.join(PROJECT_DIR, "models", "nllb-finetuned-fon-fr"),
)
DATA_PATH = os.environ.get(
    "XONET_DATA_PATH",
    os.path.join(PROJECT_DIR, "data", "cleaned_dataset_final.csv"),
)
RESULTS_DIR = os.environ.get(
    "XONET_RESULTS_DIR",
    os.path.join(PROJECT_DIR, "results"),
)

# Codes de langue NLLB (FLORES-200)
LANG_CODES = {
    "fon": "fon_Latn",   # Fongbé
    "fr": "fra_Latn",    # Français
}

DEFAULT_CONFIG = {
    "seed": 42,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "eval_samples": 500,      # nb de paires du test set (comme à l'entraînement)
    "max_source_length": 128,
    "max_target_length": 128,
    "num_beams": 5,
    "batch_size": 8,
    "num_examples_shown": 10,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Évaluation bidirectionnelle (Fon↔Fr) du traducteur XoNet"
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_CONFIG["eval_samples"],
                        help="Nombre de paires du test set à évaluer (défaut : 500)")
    parser.add_argument("--num-beams", type=int, default=DEFAULT_CONFIG["num_beams"],
                        help="Nombre de beams pour la génération (défaut : 5)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Taille des lots de génération (défaut : 8)")
    return parser.parse_args()


def detect_device() -> torch.device:
    """Détecter le dispositif disponible (CUDA, MPS, ou CPU)."""
    if torch.cuda.is_available():
        print(f"  -> GPU CUDA détecté : {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("  -> GPU Apple Silicon (MPS) détecté")
        return torch.device("mps")
    else:
        print("  -> CPU uniquement détecté (la génération peut être lente)")
        return torch.device("cpu")


# =============================================================================
# 2. CHARGEMENT DU MODÈLE
# =============================================================================

def charger_modele(device: torch.device):
    """Charge le modèle fine-tuné et son tokenizer."""
    if not os.path.isdir(MODEL_PATH):
        print(f"ERREUR : Le dossier du modèle '{MODEL_PATH}' n'existe pas.")
        print("Lance d'abord l'entraînement : python src/finetune_nllb.py")
        sys.exit(1)

    print(f"Chargement du modèle depuis : {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()
    # Accélère l'inférence : réactive le cache des past_key_values
    # (désactivé dans le config par l'entraînement).
    model.config.use_cache = True
    print(f"  -> Paramètres : {model.num_parameters():,}")
    print(f"  -> Device : {device.type}")
    return model, tokenizer


# =============================================================================
# 3. CHARGEMENT DU DATASET + REPRODUCTION DU SPLIT D'ENTRAÎNEMENT
# =============================================================================

def charger_test_set(n_samples: int) -> pd.DataFrame:
    """
    Reproduit exactement le split train/val/test de src/finetune_nllb.py
    (même random_state=42, mêmes ratios 80/10/10) puis renvoie les 500
    premières paires du test set (paires jamais vues à l'entraînement).
    """
    print(f"\nChargement du dataset : {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  -> {len(df)} paires chargées")
    print(f"  -> Colonnes : {list(df.columns)}")

    # Même shuffle que l'entraînement
    df = df.sample(frac=1, random_state=DEFAULT_CONFIG["seed"]).reset_index(drop=True)

    # Même découpage train / val / test
    n = len(df)
    n_train = int(n * DEFAULT_CONFIG["train_ratio"])
    n_val = int(n * DEFAULT_CONFIG["val_ratio"])
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

    # Même troncature que l'évaluation d'entraînement (500 paires)
    test_df = test_df.iloc[:DEFAULT_CONFIG["eval_samples"]]
    print(f"  -> Test set (jamais vu à l'entraînement) : {len(test_df)} paires")

    # Échantillonnage optionnel (test rapide)
    if n_samples < len(test_df):
        test_df = test_df.sample(n=n_samples, random_state=DEFAULT_CONFIG["seed"]).reset_index(drop=True)

    print(f"  -> Évaluation sur {len(test_df)} paires")
    return test_df


# =============================================================================
# 4. GÉNÉRATION DES TRADUCTIONS
# =============================================================================

def generer_traductions(textes_source, model, tokenizer, device,
                        src_lang: str, tgt_lang: str, config: dict):
    """
    Traduit une liste de textes par lots avec beam search.
    Les textes sont traités tels quels (aucune normalisation des caractères).
    Retourne la liste des traductions générées.
    """
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    predictions = []
    batch_size = config["batch_size"]

    for start in tqdm(range(0, len(textes_source), batch_size),
                      desc="Génération", unit="batch"):
        batch = textes_source[start:start + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["max_source_length"],
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=config["max_target_length"],
                num_beams=config["num_beams"],
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    return predictions


# =============================================================================
# 5. CALCUL DES MÉTRIQUES
# =============================================================================

def calculer_metriques(predictions, references) -> dict:
    """
    Calcule, pour un sens donné :
      - BLEU          (sacrebleu, 0-100)
      - chrF          (evaluate, 0-100)
      - Exact Match   (% de prédictions strictement identiques à la référence)
      - Longueur moyenne des prédictions vs des références (en mots)
    """
    refs = [[r] for r in references]   # sacrebleu / evaluate attendent une liste par prédiction

    metriques = {}

    # BLEU (sacrebleu)
    metriques["bleu"] = BLEU().corpus_score(predictions, refs).score

    # chrF (evaluate)
    chrf = evaluate.load("chrf")
    metriques["chrf"] = chrf.compute(predictions=predictions, references=refs)["score"]

    # Exact Match
    exact = sum(1 for p, r in zip(predictions, references) if p == r)
    metriques["exact_match_pct"] = 100.0 * exact / len(predictions)

    # Longueurs moyennes (en mots)
    metriques["avg_pred_len"] = float(np.mean([len(p.split()) for p in predictions]))
    metriques["avg_ref_len"] = float(np.mean([len(r.split()) for r in references]))

    return metriques


def afficher_metriques(metriques: dict, label: str):
    """Affiche les métriques d'un sens dans un tableau lisible."""
    print("\n" + "-" * 50)
    print(f"  MÉTRIQUES — {label}")
    print("-" * 50)
    print(f"  BLEU                     : {metriques['bleu']:8.2f}")
    print(f"  chrF                     : {metriques['chrf']:8.2f}")
    print(f"  Exact Match (%)          : {metriques['exact_match_pct']:8.2f}")
    print(f"  Longueur moyenne prédit  : {metriques['avg_pred_len']:8.2f} mots")
    print(f"  Longueur moyenne réf.    : {metriques['avg_ref_len']:8.2f} mots")


# =============================================================================
# 6. AFFICHAGE DE TRADUCTIONS CÔTE À CÔTE
# =============================================================================

def afficher_exemples(source, predictions, references, label: str,
                      config: dict) -> list[dict]:
    """
    Affiche 10 exemples côte à côte (source / prédiction / référence).
    Retourne les exemples affichés pour l'export CSV.
    """
    n = min(config["num_examples_shown"], len(source))
    indices = random.sample(range(len(source)), n)
    bleu_sentence = BLEU()

    print("\n" + "-" * 100)
    print(f"  {n} EXEMPLES — {label}")
    print("-" * 100)

    exemples = []
    for i in indices:
        s, p, r = source[i], predictions[i], references[i]
        score = bleu_sentence.sentence_score(p, [r]).score

        print(f"\n  [{i}] Source     : {s}")
        print(f"      Prédiction : {p}")
        print(f"      Référence  : {r}")
        print(f"      BLEU       : {score:.2f}")

        exemples.append({"source": s, "prediction": p, "reference": r})

    return exemples


# =============================================================================
# 7. FONCTION PRINCIPALE
# =============================================================================

def main():
    args = parse_args()
    config = {
        **DEFAULT_CONFIG,
        "num_beams": args.num_beams,
        "batch_size": args.batch_size,
        "eval_samples": args.samples,
    }

    print("=" * 60)
    print("  Évaluation du traducteur XoNet (Fongbé ↔ Français)")
    print("=" * 60)

    # Device + modèle
    device = detect_device()
    model, tokenizer = charger_modele(device)

    # Dataset (test set identique à l'entraînement)
    test_df = charger_test_set(args.samples)

    # Les deux sens à évaluer
    SENS = [
        {"label": "Fongbé → Français", "col_source": "fon", "col_cible": "fr",
         "src_lang": LANG_CODES["fon"], "tgt_lang": LANG_CODES["fr"], "sens": "fon_fr"},
        {"label": "Français → Fongbé", "col_source": "fr", "col_cible": "fon",
         "src_lang": LANG_CODES["fr"], "tgt_lang": LANG_CODES["fon"], "sens": "fr_fon"},
    ]

    metriques_globales = {}
    exemples_export = []
    summary = []

    for sens in SENS:
        print(f"\n{'=' * 60}")
        print(f"  SENS : {sens['label']}")
        print(f"{'=' * 60}")

        source_texts = test_df[sens["col_source"]].tolist()
        ref_texts = test_df[sens["col_cible"]].tolist()

        # Génération
        print("\nGénération des traductions...")
        predictions = generer_traductions(
            source_texts, model, tokenizer, device,
            sens["src_lang"], sens["tgt_lang"], config,
        )

        # Métriques
        metriques = calculer_metriques(predictions, ref_texts)
        metriques_globales[sens["sens"]] = metriques
        afficher_metriques(metriques, sens["label"])

        # Exemples côte à côte
        exemples = afficher_exemples(
            source_texts, predictions, ref_texts, sens["label"], config,
        )
        for ex in exemples:
            ex["sens"] = sens["sens"]
            exemples_export.append(ex)

        summary.append({"sens": sens["sens"],
                        "label": sens["label"],
                        "bleu": metriques["bleu"],
                        "chrf": metriques["chrf"]})

    # =============================================================================
    # EXPORT DES RÉSULTATS
    # =============================================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Métriques globales (JSON)
    rapport = {
        "model": os.path.relpath(MODEL_PATH, PROJECT_DIR),
        "data": os.path.relpath(DATA_PATH, PROJECT_DIR),
        "num_evaluated": len(test_df),
        "split": {"seed": DEFAULT_CONFIG["seed"],
                  "train": DEFAULT_CONFIG["train_ratio"],
                  "val": DEFAULT_CONFIG["val_ratio"],
                  "test": DEFAULT_CONFIG["test_ratio"]},
        "metrics": metriques_globales,
    }
    json_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    print(f"\nMétriques globales exportées dans : {json_path}")

    # 2. Échantillon de traductions (CSV)
    csv_path = os.path.join(RESULTS_DIR, "translations_sample.csv")
    pd.DataFrame(exemples_export, columns=["source", "prediction", "reference", "sens"]) \
      .to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Exemples exportés dans : {csv_path}")

    # Récapitulatif
    print("\n" + "=" * 60)
    print("  RÉCAPITULATIF")
    print("=" * 60)
    for s in summary:
        print(f"  {s['label']:20s} → BLEU : {s['bleu']:6.2f} | chrF : {s['chrf']:6.2f}")

    print("\nÉvaluation terminée.")


if __name__ == "__main__":
    main()
