# XoNet — Sentiment Analysis for Fongbe | Analyse de Sentiment en Fongbé

<p align="center">
  <b>Classification de sentiment pour les textes en langue Fongbé (Bénin)</b><br>
  <b>Sentiment classification for Fongbe language texts (Benin)</b>
</p>

<p align="center">
  <a href="#-français">Français</a> •
  <a href="#-english">English</a>
</p>

---

## 🇫🇷 Français

### À propos de XoNet

**XoNet** est une solution d'analyse et de classification de sentiment pour le **fongbé** (langue nationale parlée au Bénin, au Togo et au Nigeria), considérée comme une langue à faibles ressources (low-resource language) en traitement automatique du langage naturel (TALN / NLP).

Le modèle classe les phrases selon trois polarités :

- **`0` : Neutre** (faits bruts, descriptions, questions usuelles)
- **`1` : Négatif** (colère, tristesse, refus, douleur, réprobation)
- **`2` : Positif** (joie, accord, satisfaction, bénédiction)

---

### Corpus & Méthodologie

Le corpus totalise **87 126 phrases** annotées en fongbé issues de la fusion et du nettoyage de plusieurs sources :

1. **Parallel text dataset for Neural Machine Translation (French → Fongbe)** (Degila et al., 2020) ;
2. **Fon French Daily Dialogues (FFR)** (Dossou et al., 2021) ;
3. **Corpus extrait du site Fongbé Bénin** (textes bibliques et littéraires).

Le fichier annoté de référence est `data/corpus_final.csv`.

#### Partitionnement des données (Stratifié)

- **Train (70 %)** : 60 988 phrases
- **Validation (15 %)** : 13 069 phrases
- **Test indépendant (15 %)** : 13 069 phrases (répartition : 60,8 % Neutre, 22,7 % Négatif, 16,4 % Positif)

---

### Résultats des modèles (Évaluation sur le jeu de Test — 13 069 phrases)

| Génération            | Modèle                      | Architecture                                                                                             |     Accuracy     |     F1 Macro     | F1 Neutre / Négatif / Positif |
| :---------------------- | :--------------------------- | :------------------------------------------------------------------------------------------------------- | :---------------: | :---------------: | :----------------------------: |
| **v1**            | **Soft Voting**        | TF-IDF caractères (3,5) + SMOTE + MNB + Random Forest                                                   |      71,90 %      |      65,20 %      |       0,80 / 0,59 / 0,57       |
| **v2**            | **LinearSVC Calibré** | TF-IDF caractères (2,6) [80k features] + LinearSVC calibré                                             |      72,28 %      |      66,59 %      |       0,80 / 0,62 / 0,58       |
| **v3 (Meilleur)** | **SVM Hybride**        | **TF-IDF Hybride Caractères (2,6) + Mots (1,5) [150k features] + LinearSVC ($C=0.5$) calibré** | **75,39 %** | **70,51 %** |  **0,82 / 0,68 / 0,62**  |

> **Gain clé :** La combinaison sous-mots/caractères (qui capture les diacritiques, accents et tons propres au fongbé) et n-grammes de mots permet de passer de 65,20 % à **70,51 % de F1 Macro** sans modifier le corpus.

---

### Démarrage rapide

#### 1. Installation des dépendances

```bash
git clone https://github.com/IFRI-AI-Classes/XoNet.git
cd XoNet
pip install -e .
```

#### 2. Lancement de l'application Web

```bash
python app.py
```

L'application démarre sur `http://localhost:5000` :

- Interface soignée avec clavier virtuel des caractères spéciaux fongbé (`ɛ`, `ɔ`, `ɖ`, `ɣ`, `ɲ`, `Ɛ`, `Ɔ`, `Ɖ`).
- Moteur par défaut : **SVM Hybride v3** (avec sélecteur permettant de basculer vers le **Soft Voting** ou la **v2**).
- Affichage de la classe, du score de décision et de la répartition par classe.

#### 3. API REST

- **`GET /api/models`** : Liste les modèles disponibles et leurs métriques.
- **`POST /api/predict`** : Analyse le sentiment d'une phrase.

  ```json
  // Requête
  {
    "text": "A ɖò ganji",
    "model_key": "v3_tuned" // optionnel : "v3_tuned" | "soft_voting" | "v2"
  }
  ```

  ```json
  // Réponse
  {
    "sentiment": "Positif",
    "label_id": 2,
    "confidence": 0.733,
    "probabilities": {
      "Positif": 0.733,
      "Neutre": 0.231,
      "Négatif": 0.036
    },
    "model_label": "SVM Hybride v3 (Char+Word)"
  }
  ```

---

### Ré-entraînement et Reproductibilité

```bash
# Entraîner et comparer les modèles v2
python -m src.training.train_sentiment_v2

# Entraîner le benchmark hybride v3
python -m src.training.train_sentiment_v3

# Optimiser les hyperparamètres (C, seuils) du modèle v3
python -m src.training.tune_sentiment_v3
```

---

## 🇬🇧 English

### About XoNet

**XoNet** is a machine learning solution for sentiment analysis and polarity classification in **Fongbe** (a major language spoken in Benin, Togo, and Nigeria), addressing low-resource natural language processing (NLP) challenges.

The system classifies inputs into three sentiment categories:

- **`0`: Neutral** (factual statements, questions, everyday descriptions)
- **`1`: Negative** (anger, sadness, refusal, pain, disapproval)
- **`2`: Positive** (joy, agreement, satisfaction, blessing)

---

### Dataset & Methodology

The dataset comprises **87,126 annotated sentences** in Fongbe curated from multiple sources:

1. **Parallel text dataset for Neural Machine Translation (French → Fongbe)** (Degila et al., 2020);
2. **Fon French Daily Dialogues (FFR)** (Dossou et al., 2021);
3. **Scraped corpus from Fongbé Bénin** (biblical and literary texts).

The primary annotated file is `data/corpus_final.csv`.

#### Stratified Split

- **Train (70%)**: 60,988 sentences
- **Validation (15%)**: 13,069 sentences
- **Hold-out Test (15%)**: 13,069 sentences (distribution: 60.8% Neutral, 22.7% Negative, 16.4% Positive)

---

### Model Benchmark (Evaluation on Test Set — 13,069 sentences)

| Generation          | Model                          | Architecture                                                                                       |     Accuracy     |     Macro F1     | F1 Neutral / Negative / Positive |
| :------------------ | :----------------------------- | :------------------------------------------------------------------------------------------------- | :--------------: | :--------------: | :------------------------------: |
| **v1**        | **Soft Voting**          | Char n-grams (3,5) TF-IDF + SMOTE + MNB + Random Forest                                            |      71.90%      |      65.20%      |        0.80 / 0.59 / 0.57        |
| **v2**        | **Calibrated LinearSVC** | Char n-grams (2,6) [80k features] TF-IDF + Calibrated LinearSVC                                    |      72.28%      |      66.59%      |        0.80 / 0.62 / 0.58        |
| **v3 (Best)** | **Hybrid SVM**           | **Hybrid Char (2,6) + Word (1,5) [150k features] TF-IDF + Calibrated LinearSVC ($C=0.5$)** | **75.39%** | **70.51%** |   **0.82 / 0.68 / 0.62**   |

> **Key takeaway:** Combining character subwords (robust to Fongbe tonal diacritics and orthographic variation) with word n-grams improved performance from 65.20% to **70.51% Macro F1** without requiring additional labeled data.

---

### Quickstart

#### 1. Installation

```bash
git clone https://github.com/IFRI-AI-Classes/XoNet.git
cd XoNet
pip install -e .
```

#### 2. Run the Web Application

```bash
python app.py
```

Access the interface at `http://localhost:5000`:

- Interactive UI with an integrated Fongbe special characters keyboard (`ɛ`, `ɔ`, `ɖ`, `ɣ`, `ɲ`, `Ɛ`, `Ɔ`, `Ɖ`).
- Powered by default by the **Hybrid SVM v3** pipeline (with support for switching to **Soft Voting** or **v2**).
- Displays predicted sentiment, confidence score, and class probability distributions.

#### 3. REST API

- **`GET /api/models`**: Returns active and available models with benchmark metrics.
- **`POST /api/predict`**: Predicts the sentiment of a Fongbe string.

  ```json
  // Request
  {
    "text": "A ɖò ganji",
    "model_key": "v3_tuned" // optional: "v3_tuned" | "soft_voting" | "v2"
  }
  ```

  ```json
  // Response
  {
    "sentiment": "Positif",
    "label_id": 2,
    "confidence": 0.733,
    "probabilities": {
      "Positif": 0.733,
      "Neutre": 0.231,
      "Négatif": 0.036
    },
    "model_label": "SVM Hybride v3 (Char+Word)"
  }
  ```

---

### Training & Reproducibility

```bash
# Train and benchmark v2 linear models
python -m src.training.train_sentiment_v2

# Train the hybrid v3 candidate architectures
python -m src.training.train_sentiment_v3

# Fine-tune regularization (C) and decision thresholds for v3
python -m src.training.tune_sentiment_v3
```

---

## 📁 Repository Structure / Structure du Dépôt

```text
XoNet/
├── app.py                      # Flask backend & REST API / Serveur Flask & API REST
├── web/
│   └── index.html              # Frontend application / Interface web
├── src/
│   ├── training/
│   │   ├── sentiment_pipeline.py   # Wrapped decision pipeline / Pipeline de décision pondérée
│   │   ├── train_sentiment_v2.py   # v2 baseline training / Entraînement v2
│   │   ├── train_sentiment_v3.py   # v3 hybrid training / Entraînement hybride v3
│   │   └── tune_sentiment_v3.py    # v3 hyperparameter tuning / Optimisation v3
│   └── ingestion/                  # Data scraping and merging scripts / Scripts d'ingestion
├── models/                         # Serialized models & metrics / Modèles sauvegardés & métriques
├── notebooks/                      # Exploratory notebooks / Carnets Jupyter d'expérimentation
├── data/                           # Datasets & corpus documentation / Données & documentation
└── pyproject.toml                  # Python package configuration / Configuration du projet
```

---

## 📚 References & Acknowledgements / Références

```text
[1] Kevin Degila, Godson Kalipe, Jamiil Touré Ali, and Momboladji Balogoun.
    Parallel text dataset for Neural Machine Translation (French -> Fongbe, French -> Ewe).
    Zenodo, 2020. DOI: 10.5281/zenodo.4266935

[2] Bonaventure Dossou, Fabroni Yoclounon, Ricardo Ahounvlamè, and Chris Emezue.
    Fon French Daily Dialogues Parallel Data.
    Zenodo, 2021. DOI: 10.5281/zenodo.4432712

[3] Fongbé Bénin.
    Bible en fongbé.
    https://fongbebenin.com/bible/bible-fongbe.html
```

---

## 📄 License / Licence

This project is licensed under the [MIT License](LICENSE).
Ce projet est sous licence [MIT](LICENSE).
