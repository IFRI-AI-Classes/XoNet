# =============================================================================
# Script d'inférence : traduire avec le modèle fine-tuné
# =============================================================================
# Utilise le modèle fine-tuné NLLB pour traduire du Fongbe vers le
# Français ou du Français vers le Fongbe.
#
# Usage :
#   python src/translate.py "texte à traduire"
#   python src/translate.py "texte" --src fr --tgt fon
#
# Options :
#   --src   langue source : "fon" (fongbe) ou "fr" (français)
#   --tgt   langue cible  : "fon" ou "fr"
#   --model chemin du modèle (défaut : models/nllb-finetuned-fon-fr)
# =============================================================================

import os
import sys
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Correspondance langue courte → code NLLB (FLORES-200)
LANG_CODES = {
    "fon": "fon_Latn",
    "fr": "fra_Latn",
}


def charger_modele(model_path: str):
    """
    Charge le modèle fine-tuné et son tokenizer.
    Retourne (model, tokenizer).
    """
    print(f"Chargement du modèle depuis : {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Détection du device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    print(f"  -> Device : {device.type}")
    return model, tokenizer, device


def traduire(texte: str, tokenizer, model, device,
             source_lang: str = "fon_Latn",
             target_lang: str = "fra_Latn") -> str:
    """
    Traduit un texte du Fongbe vers le Français (ou vice versa).

    Arguments :
      - texte : phrase d'entrée
      - tokenizer, model, device : modèle chargé
      - source_lang / target_lang : codes FLORES-200 (ex: "fon_Latn")

    Retourne la traduction générée.
    """
    # Configurer les langues du tokenizer
    tokenizer.src_lang = source_lang
    tokenizer.tgt_lang = target_lang

    # Tokeniser l'entrée
    inputs = tokenizer(
        texte,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    # ID du token qui force la langue cible dans le decoder
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    # Génération avec beam search (meilleure qualité que greedy)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    # Décoder la sortie
    traduction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return traduction


def main():
    """Point d'entrée du script."""
    parser = argparse.ArgumentParser(
        description="Traducteur Fongbe ↔ Français (modèle NLLB fine-tuné)"
    )
    parser.add_argument("texte", type=str, help="Texte à traduire")
    parser.add_argument(
        "--src", type=str, choices=["fon", "fr"], default="fon",
        help="Langue source (défaut : fon)",
    )
    parser.add_argument(
        "--tgt", type=str, choices=["fon", "fr"], default="fr",
        help="Langue cible (défaut : fr)",
    )
    parser.add_argument(
        "--model", type=str, default="models/nllb-finetuned-fon-fr",
        help="Chemin du modèle fine-tuné",
    )
    args = parser.parse_args()

    # Vérifier que le modèle existe
    if not os.path.isdir(args.model):
        print(f"ERREUR : Le dossier du modèle '{args.model}' n'existe pas.")
        print("Lance d'abord l'entraînement : python src/finetune_nllb.py")
        sys.exit(1)

    # Charger le modèle
    model, tokenizer, device = charger_modele(args.model)

    # Codes de langue
    source_lang = LANG_CODES[args.src]
    target_lang = LANG_CODES[args.tgt]

    # Traduire
    print(f"\n{args.src} → {args.tgt} :")
    print(f"  Entrée  : {args.texte}")
    resultat = traduire(
        args.texte, tokenizer, model, device,
        source_lang=source_lang, target_lang=target_lang,
    )
    print(f"  Sortie  : {resultat}")


if __name__ == "__main__":
    main()
