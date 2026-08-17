# =============================================================================
# Fine-tuning de NLLB pour la traduction Fongbe  Français
# =============================================================================
# Ce script fine-tune le modèle NLLB (No Language Left Behind) de Meta
# sur le dataset nettoyé cleaned_dataset_final.csv pour produire un
# traducteur Fongbe  Français et Français  Fongbe.
#
# Utilisation :
#   python src/finetune_nllb.py
#
# Prérequis :
#   pip install torch transformers datasets evaluate sentencepiece sacrebleu pandas tqdm
# =============================================================================

import os
import json
import random
import numpy as np
import pandas as pd

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset

import evaluate
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# Tous les paramètres sont regroupés ici. Modifie cette section pour
# changer la taille du modèle, les chemins, ou les hyperparamètres.

MODEL_CONFIG = {
    # Taille du modèle NLLB :
    #   "facebook/nllb-200-distilled-600M"  → léger, pour CPU
    #   "facebook/nllb-200-distilled-1.3B"  → moyen, nécessite GPU
    #   "facebook/nllb-200-3.3B"            → lourd, nécessite GPU
    "model_name": "facebook/nllb-200-distilled-600M",

    # Chemins des fichiers (relatifs à la racine du projet)
    # POUR COLOMB : mets le output_dir sur ton Drive pour que les checkpoints
    # survivent aux déconnexions, ex : "/content/drive/MyDrive/XoNet/models/nllb-finetuned-fon-fr"
    "data_path": os.environ.get("XONET_DATA_PATH", "data/cleaned_dataset_final.csv"),
    "output_dir": os.environ.get("XONET_OUTPUT_DIR", "models/nllb-finetuned-fon-fr"),
    "cache_dir": "models/cache",
    "checkpoint_dir": os.environ.get("XONET_CHECKPOINT_DIR", "models/checkpoints"),

    # Limite du nombre de paires d'entraînement.
    # None = utiliser TOUTES les paires (71 000).
    # Sur CPU, mets 30000 pour aller plus vite.
    "max_train_samples": None,

    # Hyperparamètres d'entraînement
    "max_source_length": 128,
    "max_target_length": 128,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "num_epochs": 3,
    "warmup_ratio": 0.1,

    # Split des données
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,

    # Nombre d'exemples pour l'évaluation BLEU (pour accélérer)
    # Sur CPU, évaluer tout le test set est trop long. On échantillonne.
    "eval_samples": 500,

    # Seeds pour la reproductibilité
    "seed": 42,

    # Langues NLLB
    # Le code FLORES-200 du Fongbe est "fon_Latn", du français est "fra_Latn"
    "source_lang": "fon_Latn",
    "target_lang": "fra_Latn",
}


def set_seed(seed: int):
    """Fixer les seeds pour reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    """Détecter le dispositif disponible (CUDA, MPS, ou CPU)."""
    if torch.cuda.is_available():
        print(f"  -> GPU CUDA détecté : {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("  -> GPU Apple Silicon (MPS) détecté")
        return torch.device("mps")
    else:
        print("  -> CPU uniquement détecté")
        return torch.device("cpu")


# =============================================================================
# 2. CHARGEMENT ET PRÉPARATION DU DATASET
# =============================================================================

def load_and_split_dataset(config: dict) -> tuple:
    """
    Charge le CSV et le split en train/val/test.

    Retourne :
        (dataset_dict, raw_datasets) où dataset_dict contient les splits
        et raw_datasets contient les données brutes.
    """
    data_path = config["data_path"]
    print(f"\n[1/5] Chargement du dataset : {data_path}")

    df = pd.read_csv(data_path)
    print(f"  -> {len(df)} paires chargées")
    print(f"  -> Colonnes : {list(df.columns)}")

    # Mélanger les lignes
    df = df.sample(frac=1, random_state=config["seed"]).reset_index(drop=True)

    # Split train / val / test
    n = len(df)
    n_train = int(n * config["train_ratio"])
    n_val = int(n * config["val_ratio"])

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    # Limiter le nombre de paires d'entraînement (optionnel, pour CPU)
    max_train = config.get("max_train_samples")
    if max_train and len(train_df) > max_train:
        print(f"  -> Limitation du train set à {max_train} paires (pour CPU)")
        train_df = train_df.iloc[:max_train]

    # Limiter le nombre d'exemples d'évaluation (optionnel, pour CPU)
    eval_samples = config.get("eval_samples")
    if eval_samples and len(test_df) > eval_samples:
        test_df = test_df.iloc[:eval_samples]

    print(f"  -> Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")

    # Conversion en Dataset HuggingFace
    train_dataset = HFDataset.from_pandas(train_df)
    val_dataset = HFDataset.from_pandas(val_df)
    test_dataset = HFDataset.from_pandas(test_df)

    return train_dataset, val_dataset, test_dataset


# =============================================================================
# 3. TOKENISATION AVEC LE TOKENIZER NLLB
# =============================================================================
# NLLB utilise SentencePiece (BPE) avec un vocabulaire multilingue de 200+
# langues. Le tokenizer gère déjà le Fongbe (code : fon_Latn) et le
# français (code : fra_Latn).
#
# IMPORTANT : les tokenizers NLLB imposent d'ajouter le préfixe de langue
# cible ("__fra_Latn__") au début du texte cible (decoder_input_ids).

def tokenize_function(examples, tokenizer, config: dict):
    """
    Tokenize un batch d'exemples pour l'entraînement NLLB.

    Pour chaque paire (fon, fr) :
      - input_ids : texte source tokenizé
      - labels    : texte cible tokenizé (avec décalage pour le decoder)
      - attention_mask : masque d'attention pour l'encodeur

    NLLB nécessite le préfixe de langue cible dans le decoder.
    On suit le format attendu par NLLB pour le fine-tuning.
    """
    source_lang = config["source_lang"]
    target_lang = config["target_lang"]

    # Tokenisation des textes source (Fongbe)
    tokenizer.src_lang = source_lang
    model_inputs = tokenizer(
        examples["fon"],
        max_length=config["max_source_length"],
        padding=False,
        truncation=True,
    )

    # Tokenisation des textes cible (Français)
    tokenizer.tgt_lang = target_lang
    labels = tokenizer(
        text_target=examples["fr"],
        max_length=config["max_target_length"],
        padding=False,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def prepare_datasets(train_dataset, val_dataset, test_dataset, tokenizer, config: dict):
    """Tokeniser les trois splits."""
    print(f"\n[2/5] Tokenisation des données...")

    fn_kwargs = {"tokenizer": tokenizer, "config": config}

    train_tokenized = train_dataset.map(
        tokenize_function, fn_kwargs=fn_kwargs, batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenisation (train)",
    )
    val_tokenized = val_dataset.map(
        tokenize_function, fn_kwargs=fn_kwargs, batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenisation (val)",
    )
    test_tokenized = test_dataset.map(
        tokenize_function, fn_kwargs=fn_kwargs, batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenisation (test)",
    )

    print(f"  -> Train : {len(train_tokenized)} | Val : {len(val_tokenized)} | Test : {len(test_tokenized)}")

    return train_tokenized, val_tokenized, test_tokenized


# =============================================================================
# 4. MÉTRIQUE D'ÉVALUATION (BLEU)
# =============================================================================
# Le BLEU (Bilingual Evaluation Understudy) mesure la similarité entre
# les traductions générées et les traductions de référence.
# Score de 0 (mauvais) à 100 (parfait).

def compute_bleu_metric(tokenizer):
    """
    Retourne une fonction de métrique utilisable par le Seq2SeqTrainer.
    """
    bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Décoder les prédictions
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Décoder les labels (ignorer -100)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # SacreBLEU attend des listes de références (une liste par prédiction)
        decoded_labels = [[label] for label in decoded_labels]

        result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    return compute_metrics


# =============================================================================
# 5. FINE-TUNING AVEC Seq2SeqTrainer
# =============================================================================

def train_model(train_dataset, val_dataset, tokenizer, model, config: dict, device):
    """Lancer le fine-tuning du modèle NLLB."""
    print(f"\n[3/5] Configuration de l'entraînement...")

    # Data collator : s'occupe du padding dynamique des batches
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # Arguments d'entraînement
    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        weight_decay=config["weight_decay"],
        num_train_epochs=config["num_epochs"],
        warmup_steps=int(len(train_dataset) * config["num_epochs"] / (config["batch_size"] * config["gradient_accumulation_steps"]) * config["warmup_ratio"]),
        predict_with_generate=True,
        generation_max_length=config["max_target_length"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        fp16=(device.type == "cuda"),
        gradient_checkpointing=True,
        report_to="none",
        seed=config["seed"],
        remove_unused_columns=False,
        dataloader_pin_memory=(device.type != "cpu"),
    )

    # Métrique BLEU
    compute_metrics = compute_bleu_metric(tokenizer)

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n[4/5] Lancement de l'entraînement...")
    print(f"  -> Modèle : {config['model_name']}")
    print(f"  -> Époques : {config['num_epochs']}")
    print(f"  -> Batch size : {config['batch_size']}")
    print(f"  -> Learning rate : {config['learning_rate']}")
    print(f"  -> Device : {device.type}")
    print(f"  -> Nombre d'itérations : {len(train_dataset) // (config['batch_size'] * config['gradient_accumulation_steps']) * config['num_epochs']}")

    # Reprise automatique : si un checkpoint existe déjà, on reprend là où on s'est arrêté
    # Le Trainer sauvegarde ses checkpoints dans output_dir sous la forme checkpoint-XXX
    resume_from_checkpoint = False
    for base_dir in [config["output_dir"], config.get("checkpoint_dir")]:
        if base_dir and os.path.isdir(base_dir):
            checkpoints = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                # Prendre le checkpoint le plus récent
                latest = max(checkpoints, key=lambda d: int(d.split("-")[-1]))
                resume_path = os.path.join(base_dir, latest)
                print(f"\n  -> Checkpoint trouvé : {resume_path}")
                print(f"  -> Reprise de l'entraînement à partir de ce point...")
                resume_from_checkpoint = resume_path
                break

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    return trainer


# =============================================================================
# 6. ÉVALUATION SUR LE TEST SET
# =============================================================================

def evaluate_test(trainer, test_dataset, tokenizer, config: dict):
    """Évaluer le modèle fine-tuné sur le jeu de test."""
    print(f"\n[5/5] Évaluation sur le test set...")

    test_results = trainer.predict(test_dataset)
    metrics = test_results.metrics
    bleu_key = "test_bleu" if "test_bleu" in metrics else list(metrics.keys())[0]
    print(f"  -> BLEU sur le test : {metrics[bleu_key]:.2f}")

    # Afficher quelques traductions aléatoires pour inspection
    print("\n--- Exemples de traductions ---")
    preds = test_results.predictions[0] if isinstance(test_results.predictions, tuple) else test_results.predictions
    labels = test_results.label_ids

    # Échantillonner quelques indices
    n = len(preds)
    indices = random.sample(range(n), min(5, n))

    for i in indices:
        pred = tokenizer.decode(preds[i], skip_special_tokens=True)
        label = tokenizer.decode(
            [t for t in labels[i] if t != -100],
            skip_special_tokens=True,
        )
        print(f"  Prédit    : {pred}")
        print(f"  Référence : {label}")
        print()

    return test_results


# =============================================================================
# 7. SAUVEGARDE DU MODÈLE
# =============================================================================

def save_model(trainer, tokenizer, config: dict):
    """Sauvegarder le modèle fine-tuné et le tokenizer."""
    output_dir = config["output_dir"]
    print(f"\nSauvegarde du modèle dans : {output_dir}")

    # Sauvegarder le modèle
    trainer.save_model(output_dir)

    # Sauvegarder le tokenizer
    tokenizer.save_pretrained(output_dir)

    # Sauvegarder la configuration utilisée
    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("  -> Modèle sauvegardé avec succès !")
    print(f"  -> Tu peux charger le modèle avec :")
    print(f"     from transformers import AutoTokenizer, AutoModelForSeq2SeqLM")
    print(f"     tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    print(f"     model = AutoModelForSeq2SeqLM.from_pretrained('{output_dir}')")


# =============================================================================
# 8. FONCTION PRINCIPALE
# =============================================================================

def main():
    """Orchestrer le pipeline complet de fine-tuning."""
    print("=" * 60)
    print("  Fine-tuning NLLB : Fongbe  Français")
    print("=" * 60)
    print(f"\nConfiguration :")
    for k, v in MODEL_CONFIG.items():
        print(f"  {k:25s} = {v}")

    # Fixer la seed
    set_seed(MODEL_CONFIG["seed"])

    # Détecter le device
    device = detect_device()

    # 1. Charger et splitter le dataset
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(MODEL_CONFIG)

    # 2. Charger le tokenizer NLLB
    print(f"\nChargement du tokenizer : {MODEL_CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["model_name"],
        cache_dir=MODEL_CONFIG["cache_dir"],
        src_lang=MODEL_CONFIG["source_lang"],
        tgt_lang=MODEL_CONFIG["target_lang"],
    )

    # 3. Tokeniser les données
    train_tokenized, val_tokenized, test_tokenized = prepare_datasets(
        train_dataset, val_dataset, test_dataset, tokenizer, MODEL_CONFIG
    )

    # 4. Charger le modèle NLLB
    print(f"\nChargement du modèle : {MODEL_CONFIG['model_name']}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_CONFIG["model_name"],
        cache_dir=MODEL_CONFIG["cache_dir"],
    )
    model = model.to(device)
    print(f"  -> Paramètres : {model.num_parameters():,}")

    # 5. Fine-tuner
    trainer = train_model(train_tokenized, val_tokenized, tokenizer, model, MODEL_CONFIG, device)

    # 6. Évaluer
    evaluate_test(trainer, test_tokenized, tokenizer, MODEL_CONFIG)

    # 7. Sauvegarder
    save_model(trainer, tokenizer, MODEL_CONFIG)

    print("\n" + "=" * 60)
    print("  Fine-tuning terminé avec succès !")
    print("=" * 60)


if __name__ == "__main__":
    main()
